"""Shared helpers for the CropGuard exercises 3 to 5 (training, pipeline steps, serving).

Lives in `common/`; the scripts in exercise3/, exercise4/ and exercise5/ add that folder to their import path (the repository is cloned into the home volume).
Everything here is lifted from the Exercise 1 notebook: same model, same preprocessing, same "how sick?" score.
"""
import io, json, math, os, socket
from pathlib import Path

import numpy as np
import pandas as pd
import torch, torch.nn as nn, torch.nn.functional as F, torchvision
from PIL import Image
from sklearn.metrics import balanced_accuracy_score, roc_auc_score

IMG = 224
MEAN, STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)        # ImageNet statistics, what the pretrained network expects


# ---------------------------------------------------------------- data
class Data:
    """The packed CropGuard dataset (see data_prep/build_cropguard_dataset.py). Photos are memory-mapped, so only what is read is loaded."""
    def __init__(self, data_dir, mmap=True):
        d = Path(data_dir); self.dir = d
        self.manifest = json.loads((d / 'MANIFEST.json').read_text())
        self.pv = pd.read_csv(d / 'plantvillage/meta.csv', index_col='idx'); self.ro = pd.read_csv(d / 'rocole/meta.csv', index_col='idx')
        mode = 'r' if mmap else None
        self.pv_img = np.load(d / 'plantvillage/images.npy', mmap_mode=mode); self.ro_img = np.load(d / 'rocole/images.npy', mmap_mode=mode)
        self.classes = self.manifest['classes']; self.healthy = np.array(self.manifest['healthy_class_ids'])
        self.y = self.pv['cls'].map({c: i for i, c in enumerate(self.classes)}).values
        self.train, self.val, self.test = [np.where(self.pv['split'] == s)[0] for s in ('train', 'val', 'test')]
        self.dev = (self.ro['half'] == 'dev').values; self.locked = ~self.dev
        self.ro_sick = self.ro['sick'].values.astype(int); self.plants = self.ro['plant'].values
        self.weights = d / 'weights/convnext_tiny_imagenet1k_v1.pth'

    def check(self):
        """The same sanity checks as Exercise 1. Returns a dict (raises if the splits leak)."""
        g = self.pv['gid'].values
        leaf_overlap = len(set(g[self.train]) & set(g[self.test])) + len(set(g[self.train]) & set(g[self.val]))
        plant_overlap = len(set(self.plants[self.dev]) & set(self.plants[self.locked]))
        ok = (self.pv_img.shape[0] == len(self.pv) and self.ro_img.shape[0] == len(self.ro) and self.pv_img.shape[1:] == (IMG, IMG, 3)
              and leaf_overlap == 0 and plant_overlap == 0)
        report = {'ok': bool(ok), 'plantvillage_photos': len(self.pv), 'rocole_photos': len(self.ro), 'classes': len(self.classes),
                  'train': len(self.train), 'val': len(self.val), 'test': len(self.test), 'rocole_dev': int(self.dev.sum()), 'rocole_locked': int(self.locked.sum()),
                  'leaves_shared_train_vs_valtest': leaf_overlap, 'plants_shared_dev_vs_locked': plant_overlap, 'dataset_gb': self.manifest.get('total_gb')}
        if not ok: raise RuntimeError(f'dataset check failed: {report}')
        return report


# ---------------------------------------------------------------- where am I running
def node_name():
    """The machine this process runs on. Kubernetes passes it in as NODE_NAME (the TrainJob sets it); outside Kubernetes it is the host name."""
    return os.environ.get('NODE_NAME') or socket.gethostname()


# ---------------------------------------------------------------- model
def build_model(num_classes, weights=None, drop_path=0.1, freeze_stages=0):
    """ConvNeXt-Tiny with a fresh num_classes-way head. `weights` = path to the ImageNet state dict (None = random init, used when loading a saved model)."""
    m = torchvision.models.convnext_tiny(weights=None, stochastic_depth_prob=drop_path)
    if weights is not None: m.load_state_dict(torch.load(weights, map_location='cpu', weights_only=True))
    m.classifier[2] = nn.Linear(768, num_classes)
    n_frozen = {0: 0, 1: 2, 2: 4, 3: 6, 4: 8}[freeze_stages]          # features = stem, stage1, down, stage2, down, stage3, down, stage4
    for p in m.features[:n_frozen].parameters(): p.requires_grad = False
    return m, n_frozen


def to_device(batch_u8, device):
    """uint8 (B,H,W,3) -> normalised float (B,3,H,W) on the device, channels-last."""
    mean = torch.tensor(MEAN, device=device).view(1, 3, 1, 1) * 255; std = torch.tensor(STD, device=device).view(1, 3, 1, 1) * 255
    x = batch_u8.to(device, non_blocking=True).permute(0, 3, 1, 2).float()
    return ((x - mean) / std).contiguous(memory_format=torch.channels_last)


@torch.inference_mode()
def predict_proba(model, images, device, bs=128):
    model.eval(); out = []
    for i in range(0, len(images), bs):
        with torch.autocast(device.type, dtype=torch.float16, enabled=device.type == 'cuda'):
            out.append(F.softmax(model(to_device(torch.from_numpy(np.array(images[i:i + bs])), device)).float(), 1).cpu())
    return torch.cat(out).numpy()


# ---------------------------------------------------------------- scores
def triage_score(P, healthy):
    """'How sick?' for one photo: 1 minus the probability the model gives to all its healthy classes."""
    return 1 - P[:, healthy].sum(1)


def best_cutoff(score, sick, mask, grid=None):
    """The cut-off on the sick score that gives the best balanced accuracy on the photos in `mask` (choose it on the DEV half only)."""
    grid = np.linspace(.02, .98, 97) if grid is None else grid
    return round(float(grid[int(np.argmax([balanced_accuracy_score(sick[mask], score[mask] >= c) for c in grid]))]), 2)


def evaluate_model(model, data, device, include_locked=False, cutoff=None):
    """Lab accuracy (val, test) and field scores. The locked half is scored only if asked: it is the final exam, once per candidate.
    If `cutoff` is None it is tuned on the dev half and returned; otherwise the given one is applied (used when judging a saved model)."""
    r = {'val_acc': float((predict_proba(model, data.pv_img[data.val], device).argmax(1) == data.y[data.val]).mean()),
         'test_acc': float((predict_proba(model, data.pv_img[data.test], device).argmax(1) == data.y[data.test]).mean())}
    s = triage_score(predict_proba(model, data.ro_img, device), data.healthy)
    r['dev_auc'] = float(roc_auc_score(data.ro_sick[data.dev], s[data.dev]))
    r['cutoff'] = best_cutoff(s, data.ro_sick, data.dev) if cutoff is None else float(cutoff)
    r['dev_bal_acc_at_cutoff'] = float(balanced_accuracy_score(data.ro_sick[data.dev], s[data.dev] >= r['cutoff']))
    r['dev_share_called_sick_at_0.5'] = float((s[data.dev] >= .5).mean())
    if include_locked:
        r['locked_auc'] = float(roc_auc_score(data.ro_sick[data.locked], s[data.locked]))
        r['locked_bal_acc_at_cutoff'] = float(balanced_accuracy_score(data.ro_sick[data.locked], s[data.locked] >= r['cutoff']))
        r['locked_bal_acc_at_0.5'] = float(balanced_accuracy_score(data.ro_sick[data.locked], s[data.locked] >= .5))
        r['locked_share_called_sick_at_0.5'] = float((s[data.locked] >= .5).mean())
    return r


# ---------------------------------------------------------------- saved models
def load_saved_model(model_dir, device):
    """Load a model folder written by train_ddp.py: returns (model on device, metadata dict)."""
    d = Path(model_dir); meta = json.loads((d / 'metadata.json').read_text())
    m, _ = build_model(len(meta['classes']), weights=None, drop_path=0.0)
    m.load_state_dict(torch.load(d / 'model.pt', map_location='cpu', weights_only=True))
    return m.to(device).to(memory_format=torch.channels_last).eval(), meta


def square224(im):
    """Centre square crop, then shrink: the preprocessing the model was trained with (Exercise 1)."""
    w, h = im.size; s = min(w, h); l, t = (w - s) // 2, (h - s) // 2
    return im.convert('RGB').crop((l, t, l + s, t + s)).resize((IMG, IMG), Image.BICUBIC)


def photo_from_bytes(b):
    return np.asarray(square224(Image.open(io.BytesIO(b))))
