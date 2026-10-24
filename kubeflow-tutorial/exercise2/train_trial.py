"""One Katib trial: fine-tune ConvNeXt-Tiny on lab photos with the given learning rate and print the scores Katib reads.

    python train_trial.py --lr 3e-4 --epochs 2 --seed 42

Code is lifted from notebooks/exercise1_workspace_and_baseline.ipynb (same model, data, fp16 recipe). No augmentation.
Katib's StdOut collector reads `name=value` lines, so the final lines are:
    dev_auc=0.8123      ranking score on the dev half of the coffee photos (the objective)
    val_acc=0.9634      lab accuracy on held-out PlantVillage leaves (the safety check)
The locked half is never scored here: it is the final exam, not something a search may look at.

Speed and early-stopping options (all off unless given; defaults reproduce the plain 2-epoch run):
    --max-seconds 33.5          training-time cap; the learning-rate schedule follows the clock if it runs behind (calibrate: 1.1 x the reference run's train_seconds)
    --stop-rule 0.5:0.88        at 50% of the schedule, stop if lab accuracy on 500 held-out leaves is below 0.88
    --stop-auc  0.5:0.78        at 50% of the schedule, stop if the dev ranking score is below 0.78
    --freeze-stages N           keep the stem and N stages at pretrained weights (1 or 2 is free and 20 to 32% faster)
    --benchmark                 cuDNN auto-tuning ON: +8 s at the start for ~15% faster steps; off is better for 2-epoch trials
"""
import argparse, json, math, os, time
from pathlib import Path

import numpy as np, pandas as pd, torch, torch.nn as nn, torch.nn.functional as F, torchvision
from sklearn.metrics import roc_auc_score

ap = argparse.ArgumentParser()
ap.add_argument('--lr', type=float, default=2e-4)
ap.add_argument('--log10-lr', type=float, default=None, help='learning rate as a power of ten (-3.5 = 3.2e-4); overrides --lr. Lets a search sample the rate on a log scale')
ap.add_argument('--epochs', type=int, default=2)
ap.add_argument('--seed', type=int, default=42)
ap.add_argument('--batch-size', type=int, default=64)
ap.add_argument('--weight-decay', type=float, default=0.05)
ap.add_argument('--label-smoothing', type=float, default=0.1)
ap.add_argument('--workers', type=int, default=8)
ap.add_argument('--drop-path', type=float, default=0.1, help='stochastic depth: chance of skipping the last block while training (ramps up from 0 in the first block). torchvision default is 0.1')
ap.add_argument('--freeze-stages', type=int, default=0, choices=[0, 1, 2, 3, 4], help='0 = train everything; 1..3 = keep the stem and that many stages at their pretrained weights; 4 = train only the new classifier')
ap.add_argument('--max-seconds', type=float, default=None, help='training time budget in seconds (excludes checkpoint evaluations). The learning-rate schedule follows whichever is further along, steps or clock, so the run always finishes annealed by the deadline')
ap.add_argument('--stop-rule', action='append', default=[], metavar='FRAC:MIN_VAL_ACC', help='early stopping: when training reaches FRAC (0-1) of the schedule, score 500 held-out lab photos and stop if accuracy is below MIN_VAL_ACC. Repeatable, e.g. --stop-rule 0.5:0.90')
ap.add_argument('--stop-auc', action='append', default=[], metavar='FRAC:MIN_AUC', help='early stopping on the dev ranking score: at FRAC of the schedule, stop if it is below MIN_AUC. Repeatable')
ap.add_argument('--report', action='store_true', help='for Katib early stopping: also print `dev_auc=...` (the objective) at every --curve checkpoint, so the search can compare a running trial with finished ones')
ap.add_argument('--curve', type=float, nargs='*', default=[], metavar='FRAC', help='calibration mode: at these fractions of the schedule also print lab accuracy and dev ranking score (costs a few seconds each)')
ap.add_argument('--benchmark', action='store_true', help='turn ON cuDNN auto-tuning: about 8 s extra at the start for about 15% faster steps, so only worth it for runs longer than about a minute')
ap.add_argument('--data', default=os.environ.get('CROPGUARD_DATA', '/data/cropguard'))
a = ap.parse_args()
if a.log10_lr is not None: a.lr = 10 ** a.log10_lr
T0 = time.time()
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'; CUDA_OK = DEVICE == 'cuda'
torch.backends.cudnn.benchmark = CUDA_OK and a.benchmark; torch.manual_seed(a.seed); np.random.seed(a.seed)

DATA = Path(a.data); MANIFEST = json.loads((DATA / 'MANIFEST.json').read_text())
pv = pd.read_csv(DATA / 'plantvillage/meta.csv', index_col='idx'); ro = pd.read_csv(DATA / 'rocole/meta.csv', index_col='idx')
pv_img = np.load(DATA / 'plantvillage/images.npy'); ro_img = np.load(DATA / 'rocole/images.npy')
CLASSES = MANIFEST['classes']; HEALTHY = np.array(MANIFEST['healthy_class_ids'])
y = pv['cls'].map({c: i for i, c in enumerate(CLASSES)}).values
TRAIN, VAL = [np.where(pv['split'] == s)[0] for s in ('train', 'val')]
DEV = (ro['half'] == 'dev').values; ro_sick = ro['sick'].values.astype(int)

MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1) * 255
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1) * 255
def to_gpu(b): return ((b.to(DEVICE, non_blocking=True).permute(0, 3, 1, 2).float() - MEAN) / STD).contiguous(memory_format=torch.channels_last)

class LeafDataset(torch.utils.data.Dataset):
    def __init__(self, idx): self.idx = idx
    def __len__(self): return len(self.idx)
    def __getitem__(self, k): i = int(self.idx[k]); return torch.from_numpy(np.ascontiguousarray(pv_img[i])), int(y[i])

@torch.inference_mode()
def predict_proba(model, images, bs=128):
    model.eval(); out = []
    for i in range(0, len(images), bs):
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=CUDA_OK):
            out.append(F.softmax(model(to_gpu(torch.from_numpy(np.ascontiguousarray(images[i:i + bs])))).float(), 1).cpu())
    return torch.cat(out).numpy()

model = torchvision.models.convnext_tiny(weights=None, stochastic_depth_prob=a.drop_path)
model.load_state_dict(torch.load(DATA / 'weights/convnext_tiny_imagenet1k_v1.pth', map_location='cpu', weights_only=True))
model.classifier[2] = nn.Linear(768, len(CLASSES)); model = model.to(DEVICE).to(memory_format=torch.channels_last)
N_FROZEN = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}[a.freeze_stages]            # model.features = stem, stage1, down, stage2, down, stage3, down, stage4
for p_ in model.features[:N_FROZEN].parameters(): p_.requires_grad = False
n_train = sum(p_.numel() for p_ in model.parameters() if p_.requires_grad)

dl = torch.utils.data.DataLoader(LeafDataset(TRAIN), batch_size=a.batch_size, shuffle=True, drop_last=True, num_workers=a.workers, pin_memory=CUDA_OK,
                                 prefetch_factor=4 if a.workers else None, generator=torch.Generator().manual_seed(a.seed))
opt = torch.optim.AdamW([p_ for p_ in model.parameters() if p_.requires_grad], lr=a.lr, weight_decay=a.weight_decay)
total, warm = a.epochs * len(dl), max(10, int(0.1 * a.epochs * len(dl)))
scaler = torch.amp.GradScaler(enabled=CUDA_OK)

VAL_SUB = np.random.RandomState(0).choice(VAL, 500, replace=False)         # small fixed lab check used at checkpoints
RULES = sorted((float(r.split(':')[0]), float(r.split(':')[1])) for r in a.stop_rule)
AUC_RULES = sorted((float(r.split(':')[0]), float(r.split(':')[1])) for r in a.stop_auc)
CHECKS = sorted({f for f, _ in RULES} | {f for f, _ in AUC_RULES} | set(a.curve))
def to_train(): model.train(); model.features[:N_FROZEN].eval()           # frozen layers also stay out of train mode (no stochastic depth)

t_train, t_eval, step, stopped, finished, nxt = time.time(), 0.0, 0, None, False, 0
clock = lambda: time.time() - t_train - t_eval                               # training time so far, not counting checkpoint evaluations
def progress(): return min(1.0, max(step / total, clock() / a.max_seconds if a.max_seconds else 0.0))
for ep in range(a.epochs):
    to_train()
    for xb, yb in dl:
        pr = progress()                                                       # learning rate: warm-up, then cosine over the schedule
        for g in opt.param_groups: g['lr'] = a.lr * min(1, (step + 1) / warm) * 0.5 * (1 + math.cos(math.pi * pr))
        x, t = to_gpu(xb), yb.to(DEVICE, non_blocking=True)
        with torch.autocast(DEVICE, dtype=torch.float16, enabled=CUDA_OK): loss = F.cross_entropy(model(x), t, label_smoothing=a.label_smoothing)
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        scaler.step(opt); scaler.update(); step += 1
        pr = progress()
        while nxt < len(CHECKS) and pr >= CHECKS[nxt]:
            if CUDA_OK: torch.cuda.synchronize()
            te = time.time(); sub = float((predict_proba(model, pv_img[VAL_SUB]).argmax(1) == y[VAL_SUB]).mean()); line = f'checkpoint progress={CHECKS[nxt]:.2f} clock_s={clock():.1f} val_sub={sub:.3f}'
            auc_now = None
            if CHECKS[nxt] in a.curve or any(f == CHECKS[nxt] for f, _ in AUC_RULES): auc_now = float(roc_auc_score(ro_sick[DEV], 1 - predict_proba(model, ro_img[DEV])[:, HEALTHY].sum(1))); line += f' dev_auc_now={auc_now:.3f}'
            print(line, flush=True)
            if a.report and auc_now is not None: print(f'dev_auc={auc_now:.4f}\nval_acc={sub:.4f}', flush=True)     # Katib needs EVERY tracked metric reported before it can stop a trial cleanly
            t_eval += time.time() - te; to_train()
            hit = [f'lab {sub:.3f} < {mn}' for f, mn in RULES if f == CHECKS[nxt] and sub < mn] + [f'dev score {auc_now:.3f} < {mn}' for f, mn in AUC_RULES if f == CHECKS[nxt] and auc_now < mn]
            if hit: stopped = (CHECKS[nxt], sub, '; '.join(hit)); break
            nxt += 1
        if stopped or pr >= 1.0: finished = True; break
    print(f'stopped after {step} of {total} steps' if finished and step < total else f'epoch {ep + 1}/{a.epochs} done', flush=True)
    if finished: break
if CUDA_OK: torch.cuda.synchronize()
train_s = time.time() - t_train - t_eval

val_acc = float((predict_proba(model, pv_img[VAL]).argmax(1) == y[VAL]).mean())
sick_score = 1 - predict_proba(model, ro_img[DEV])[:, HEALTHY].sum(1)       # derived "how sick?" score, dev half only
dev_auc = float(roc_auc_score(ro_sick[DEV], sick_score))
print(f'lr={a.lr} drop_path={a.drop_path} freeze_stages={a.freeze_stages} trainable_m={n_train / 1e6:.1f} seed={a.seed} train_seconds={train_s:.1f} total_seconds={time.time() - T0:.1f} peak_gpu_gib={torch.cuda.max_memory_allocated() / 2**30 if CUDA_OK else 0:.1f}')
print(f'steps_done={step} steps_planned={total} early_stopped={int(stopped is not None)}' + (f' stop_progress={stopped[0]} stop_reason="{stopped[2]}"' if stopped else ''))
print(f'val_acc={val_acc:.4f}'); print(f'dev_auc={dev_auc:.4f}')
