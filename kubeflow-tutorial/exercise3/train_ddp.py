"""Exercise 3: full-length training of the CropGuard model on one or more GPUs (PyTorch DDP), then save it for Exercises 4 and 5.

    python exercise3/train_ddp.py --lr 3e-4 --epochs 2                        # one process (also fine on a CPU for a smoke test with --limit-train)
    torchrun --nproc_per_node=2 exercise3/train_ddp.py --lr 3e-4 --epochs 2   # two GPUs on one machine
    (Kubeflow Trainer runs `torchrun` for you and sets the rank variables: see exercise3/make_trainjob.py)

Each GPU trains on its own half of the photos and the gradients are averaged, so two GPUs do one epoch in about half the time.
The learning rate is NOT rescaled for more GPUs: batch size per GPU stays the same, so the effective batch doubles (compare 1 and 2 GPUs fairly: same --lr, same --epochs).

Output, in <out>/<name>/ :  model.pt (weights), metadata.json (classes, settings, scores, the cut-off tuned on the dev half)  and  <out>/latest.json (which run is the newest).
The locked half of the coffee photos is NOT scored here: it is the final exam, taken once by the Exercise 4 pipeline's gate.
"""
import argparse, json, math, os, sys, time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np, torch, torch.nn as nn, torch.nn.functional as F
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'common'))
import cropguard_lib as cg

ap = argparse.ArgumentParser()
ap.add_argument('--data', default=os.environ.get('CROPGUARD_DATA', '/data/cropguard'))
ap.add_argument('--out', default='/data/models/cropguard', help='folder that collects model runs')
ap.add_argument('--name', default=None, help='run name (default: timestamp + learning rate)')
ap.add_argument('--name-prefix', default=None, help='with --run-id: the run is named <prefix>-<run-id> (the pipeline uses this)')
ap.add_argument('--run-id', default=None)
ap.add_argument('--lr', type=float, default=3e-4, help='the winner of the Katib search')
ap.add_argument('--epochs', type=int, default=2, help='2 epochs is what the Katib winner used; more rarely helps this dataset')
ap.add_argument('--batch-size', type=int, default=64, help='per GPU')
ap.add_argument('--weight-decay', type=float, default=0.05)
ap.add_argument('--label-smoothing', type=float, default=0.1)
ap.add_argument('--drop-path', type=float, default=0.1)
ap.add_argument('--freeze-stages', type=int, default=0, choices=[0, 1, 2, 3, 4])
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--workers', type=int, default=4, help='data-loader processes per GPU')
ap.add_argument('--backend', default='auto', choices=['auto', 'nccl', 'gloo'])
ap.add_argument('--benchmark', default='auto', choices=['auto', 'on', 'off'], help='cuDNN auto-tuning: +8 s at the start, ~15%% faster steps; auto = on from 5 epochs')
ap.add_argument('--allow-multi-node', action='store_true', help='let the workers run on different machines (by default they must share one: the 2-GPU job is meant to use the fast link inside a single node)')
ap.add_argument('--limit-train', type=int, default=None, help='train on only this many photos (smoke tests)')
a = ap.parse_args()

RANK, WORLD, LOCAL = int(os.environ.get('RANK', 0)), int(os.environ.get('WORLD_SIZE', 1)), int(os.environ.get('LOCAL_RANK', 0))
CUDA = torch.cuda.is_available()
device = torch.device('cuda', LOCAL % torch.cuda.device_count()) if CUDA else torch.device('cpu')
if CUDA: torch.cuda.set_device(device)
torch.backends.cudnn.benchmark = CUDA and (a.benchmark == 'on' or (a.benchmark == 'auto' and a.epochs >= 5))
if WORLD > 1: dist.init_process_group(backend=('nccl' if CUDA else 'gloo') if a.backend == 'auto' else a.backend, timeout=timedelta(minutes=30))
torch.manual_seed(a.seed); np.random.seed(a.seed)

def log(msg): print(f'[rank {RANK}/{WORLD}] {msg}', flush=True)
def say(msg): print(msg, flush=True) if RANK == 0 else None             # plain lines, one copy: scores that tools can parse (name=value)
def reduce_sum(t):
    if WORLD > 1: dist.all_reduce(t)
    return t

T0 = time.time()
if WORLD > 1:                                                          # the safety net behind the manifest's same-node rule
    where = [None] * WORLD; dist.all_gather_object(where, cg.node_name())
    if len(set(where)) > 1 and not a.allow_multi_node:
        if RANK == 0: print(f'ERROR: the {WORLD} workers run on different machines {sorted(set(where))}. The 2-GPU job must use one node: check the TrainJob (numNodes 1, or the pod affinity). Stopping.', flush=True)
        dist.barrier(); sys.exit(2)
    if RANK == 0: print(f'all {WORLD} workers run on node {where[0]}', flush=True)
data = cg.Data(a.data)
if RANK == 0: say(f'dataset check: {data.check()}')
train_idx = np.random.RandomState(a.seed).permutation(data.train)[:a.limit_train] if a.limit_train else data.train
log(f'device {device} ({torch.cuda.get_device_name(device) if CUDA else "cpu"}) | {len(train_idx):,} training photos, about {len(train_idx) // WORLD:,} per rank | batch {a.batch_size} per rank | torch {torch.__version__}')

class Photos(torch.utils.data.Dataset):
    def __len__(self): return len(train_idx)
    def __getitem__(self, k): i = int(train_idx[k]); return torch.from_numpy(np.array(data.pv_img[i])), int(data.y[i])

sampler = torch.utils.data.distributed.DistributedSampler(Photos(), num_replicas=WORLD, rank=RANK, shuffle=True, seed=a.seed, drop_last=True)
dl = torch.utils.data.DataLoader(Photos(), batch_size=a.batch_size, sampler=sampler, drop_last=True, num_workers=a.workers, pin_memory=CUDA, prefetch_factor=4 if a.workers else None)

model, n_frozen = cg.build_model(len(data.classes), weights=data.weights, drop_path=a.drop_path, freeze_stages=a.freeze_stages)
model = model.to(device).to(memory_format=torch.channels_last)
core = model                                                           # the plain model (no DDP wrapper) for evaluation and saving
if WORLD > 1: model = DDP(model, device_ids=[device.index] if CUDA else None)
opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
total, warm = a.epochs * len(dl), max(10, int(0.1 * a.epochs * len(dl)))
sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1, s / total))))
scaler = torch.amp.GradScaler(enabled=CUDA)
if RANK == 0:
    say(f'training {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.1f} M parameters | {a.epochs} epochs x {len(dl)} steps | lr {a.lr} | global batch {a.batch_size * WORLD} | setup {time.time() - T0:.0f}s')

t_train, history = time.time(), []
for ep in range(a.epochs):
    sampler.set_epoch(ep); model.train(); core.features[:n_frozen].eval(); t0 = time.time()
    loss_sum, seen = torch.zeros((), device=device), torch.zeros((), device=device)
    for xb, yb in dl:
        x, t = cg.to_device(xb, device), yb.to(device, non_blocking=True)
        with torch.autocast(device.type, dtype=torch.float16, enabled=CUDA): loss = F.cross_entropy(model(x), t, label_smoothing=a.label_smoothing)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); sched.step()
        loss_sum += loss.detach() * len(t); seen += len(t)
    if CUDA: torch.cuda.synchronize()
    secs = time.time() - t0; loss_sum, seen = reduce_sum(loss_sum), reduce_sum(seen)
    row = {'epoch': ep + 1, 'train_loss': float(loss_sum / seen), 'seconds': secs, 'photos_per_second': float(seen) / secs}
    if RANK == 0:
        row['val_acc'] = float((cg.predict_proba(core, data.pv_img[data.val], device).argmax(1) == data.y[data.val]).mean())      # rank 0 checks progress; the others wait at the next step
        say(f'epoch {ep + 1}/{a.epochs}  train loss {row["train_loss"]:.3f}  lab val acc {row["val_acc"]:.3f}  {secs:5.1f}s  ({row["photos_per_second"]:,.0f} photos/s across {WORLD} GPU(s))')
    history.append(row)
if CUDA: torch.cuda.synchronize()
train_seconds = time.time() - t_train
if WORLD > 1: dist.barrier()

if RANK == 0:
    scores = cg.evaluate_model(core, data, device, include_locked=False)
    name = a.name or (f'{a.name_prefix}-{a.run_id}' if a.name_prefix and a.run_id else f'{datetime.now():%Y%m%d-%H%M%S}-lr{a.lr:g}-{WORLD}gpu')
    final = Path(a.out) / name; tmp = Path(a.out) / f'.{name}.partial'; tmp.mkdir(parents=True, exist_ok=True)
    torch.save({k: v.detach().cpu() for k, v in core.state_dict().items()}, tmp / 'model.pt')
    meta = {'name': name, 'created': datetime.now().isoformat(timespec='seconds'), 'arch': 'convnext_tiny', 'image_size': cg.IMG, 'classes': data.classes,
            'healthy_class_ids': data.healthy.tolist(), 'preprocess': 'centre square crop, resize 224, ImageNet normalisation',
            'cutoff': scores['cutoff'], 'cutoff_note': 'tuned on the dev half of the coffee photos: call a leaf sick if its sick score >= cutoff (an inference-time setting)',
            'confidence_threshold': 0.5, 'scores': scores,
            'training': {'lr': a.lr, 'epochs': a.epochs, 'batch_size_per_gpu': a.batch_size, 'gpus': WORLD, 'global_batch': a.batch_size * WORLD, 'weight_decay': a.weight_decay,
                         'label_smoothing': a.label_smoothing, 'drop_path': a.drop_path, 'freeze_stages': a.freeze_stages, 'seed': a.seed, 'train_photos': int(len(train_idx)),
                         'train_seconds': round(train_seconds, 1), 'mean_photos_per_second': round(float(np.mean([h['photos_per_second'] for h in history])), 1),
                         'history': history, 'torch': torch.__version__, 'gpu': torch.cuda.get_device_name(device) if CUDA else 'cpu'},
            'locked_half_scored': False, 'dataset': {'name': data.manifest.get('name'), 'built': data.manifest.get('built')}}
    (tmp / 'metadata.json').write_text(json.dumps(meta, indent=1))
    if final.exists(): raise SystemExit(f'{final} already exists: pick another --name')
    tmp.rename(final)                                                   # the folder appears all at once, never half written
    latest = Path(a.out) / 'latest.json.partial'; latest.write_text(json.dumps({'name': name, 'path': str(final), 'created': meta['created']}, indent=1)); latest.replace(Path(a.out) / 'latest.json')
    say(f'photos_per_second={meta["training"]["mean_photos_per_second"]}'); say(f'train_seconds={train_seconds:.1f}'); say(f'total_seconds={time.time() - T0:.1f}')
    say(f'val_acc={scores["val_acc"]:.4f}'); say(f'test_acc={scores["test_acc"]:.4f}'); say(f'dev_auc={scores["dev_auc"]:.4f}'); say(f'cutoff={scores["cutoff"]:.2f}')
    say(f'saved={final}')
if WORLD > 1: dist.barrier(); dist.destroy_process_group()
