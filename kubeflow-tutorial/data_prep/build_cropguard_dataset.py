"""Instructor-side: pack the CropGuard tutorial dataset that gets copied onto the participants' PVC (target: well under 5 GB).

    python data_prep/build_cropguard_dataset.py                      # run from kubeflow-tutorial/
    python data_prep/build_cropguard_dataset.py --raw data/raw --out data/cropguard --per-class 200

Reads the raw archives (downloads them if missing: PlantVillage from Hugging Face, RoCoLe from Mendeley, ~4.4 GB) and writes:

    cropguard/MANIFEST.json                  counts, class list, licences, split recipe
    cropguard/plantvillage/images.npy        (N, 224, 224, 3) uint8, memory-mappable
    cropguard/plantvillage/masks.npy         (N, 224, 28) uint8, bit-packed leaf masks
    cropguard/plantvillage/meta.csv          cls, crop, healthy, gid (leaf id), src (capture session), split (train/val/test, BY LEAF)
    cropguard/rocole/images.npy              (1560, 224, 224, 3) uint8: centre square crop of each smartphone photo
    cropguard/rocole/meta.csv                filename, cls, sick, plant, half (dev/locked, split BY PLANT, seed 123)
    cropguard/weights/convnext_tiny_imagenet1k_v1.pth    so the tutorial needs no internet

The subset, the leaf-grouped 60/20/20 split (seed 42) and the RoCoLe plant split (seed 123) reproduce notebooks/cropguard_beta.ipynb.
"""
import argparse, hashlib, io, json, random, re, time, urllib.request, zipfile
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image, ImageFilter
from sklearn.model_selection import GroupShuffleSplit

IMG, SEED = 224, 42
PV_BASE = 'https://huggingface.co/datasets/mohanty/PlantVillage/resolve/main/'
PV_FILES = ['data.zip', 'leaf_grouping/leaf-map.json', 'splits/color_train.txt', 'splits/color_test.txt']
ROCOLE_URL = 'https://data.mendeley.com/public-api/zip/c5yvn32dzg/download/2'
ROCOLE_DIR = 'RoCoLe A robusta coffee leaf images dataset/'


def fetch(url, dest):
    dest.parent.mkdir(parents=True, exist_ok=True)
    if not dest.exists():
        print('downloading', dest.name, flush=True); urllib.request.urlretrieve(url, dest)


def plantvillage_index(raw):
    lm = json.load(open(raw / 'plantvillage/leaf_grouping/leaf-map.json'))
    rows = []
    for split in ('train', 'test'):
        for p in (raw / f'plantvillage/splits/color_{split}.txt').read_text().split('\n'):
            if not p.strip(): continue
            base = p.split('/')[-1]; key = re.sub(r'\.jpe?g$', '', base.split('___', 1)[1] if '___' in base else base, flags=re.I).lower()
            v = lm.get(key)
            rows.append({'path': p.strip(), 'cls': p.split('/')[2], 'key': key, 'leaf': v[0] if v else None, 'src': key.rsplit(' ', 1)[0]})
    meta = pd.DataFrame(rows)
    meta['gid'] = meta.leaf.where(meta.leaf.notna(), 'u' + meta.index.astype(str))     # unknown leaf -> its own group
    return meta


def mask_from_segmented(seg):
    m = Image.fromarray(((np.asarray(seg).max(axis=2) > 15) * 255).astype('uint8'))
    return np.asarray(m.filter(ImageFilter.MaxFilter(5)).filter(ImageFilter.MinFilter(5))) > 127     # close small holes


def build_plantvillage(raw, out, per_class):
    meta = plantvillage_index(raw)
    Z = zipfile.ZipFile(raw / 'plantvillage/data.zip'); rs = random.Random(SEED); segmap = {}
    for nm in Z.namelist():
        if nm.startswith('raw/segmented/') and nm.lower().endswith('.jpg') and '___' in nm.split('/')[-1]:
            k = re.sub(r'(_final_masked)?\.jpe?g$', '', nm.split('/')[-1].split('___', 1)[1], flags=re.I).lower(); segmap[(nm.split('/')[2], k)] = nm
    picked = []
    for c, df in meta.groupby('cls'):                                  # whole leaves until the class quota is reached
        gl = list(df.gid.unique()); rs.shuffle(gl); n = 0
        for gid in gl:
            sub = df[df.gid == gid]; picked.append(sub); n += len(sub)
            if n >= per_class: break
    pool = pd.concat(picked).reset_index(drop=True)
    pool = pool[[(c, k) in segmap for c, k in zip(pool.cls, pool.key)]].reset_index(drop=True)
    rd = lambda n: Image.open(io.BytesIO(Z.read(n))).convert('RGB').resize((IMG, IMG), Image.BICUBIC)
    imgs = np.lib.format.open_memmap(out / 'plantvillage/images.npy', mode='w+', dtype='uint8', shape=(len(pool), IMG, IMG, 3))
    masks = np.zeros((len(pool), IMG, IMG // 8), 'uint8'); t0 = time.time()
    for i, r in enumerate(pool.itertuples()):
        imgs[i] = np.asarray(rd(r.path)); masks[i] = np.packbits(mask_from_segmented(rd(segmap[(r.cls, r.key)])), axis=-1)
        if i % 1000 == 0: print(f'  plantvillage {i}/{len(pool)}  {time.time() - t0:.0f}s', flush=True)
    imgs.flush(); np.save(out / 'plantvillage/masks.npy', masks)
    # split BY LEAF (the shipped split leaks): 60 / 20 / 20, identical to the beta notebook
    y = pd.factorize(pool.cls, sort=True)[0]; X = np.zeros((len(pool), 1))
    trval, test_i = next(GroupShuffleSplit(1, test_size=.2, random_state=SEED).split(X, y, pool.gid))
    tr_rel, va_rel = next(GroupShuffleSplit(1, test_size=.25, random_state=SEED).split(X[trval], y[trval], pool.gid.values[trval]))
    split = np.empty(len(pool), object); split[test_i] = 'test'; split[trval[va_rel]] = 'val'; split[trval[tr_rel]] = 'train'
    pool['crop'] = pool.cls.str.split('___').str[0]; pool['healthy'] = pool.cls.str.endswith('healthy'); pool['split'] = split
    pool[['cls', 'crop', 'healthy', 'gid', 'src', 'path', 'split']].to_csv(out / 'plantvillage/meta.csv', index_label='idx')
    return pool


def square224(im):                                                      # centre square crop, then shrink
    w, h = im.size; s = min(w, h); l, t = (w - s) // 2, (h - s) // 2
    return im.convert('RGB').crop((l, t, l + s, t + s)).resize((IMG, IMG), Image.BICUBIC)


def build_rocole(raw, out):
    Z = zipfile.ZipFile(raw / 'rocole/rocole.zip')
    ann = pd.read_csv(io.BytesIO(Z.read(ROCOLE_DIR + 'Annotations/RoCoLE-csv.csv')))
    files = ann['External ID'].tolist(); cls = [json.loads(l).get('classification') for l in ann['Label']]
    imgs = np.stack([np.asarray(square224(Image.open(io.BytesIO(Z.read(ROCOLE_DIR + 'Photos/' + f))))) for f in files])
    np.save(out / 'rocole/images.npy', imgs)
    meta = pd.DataFrame({'filename': files, 'cls': cls, 'plant': [f[:-6] for f in files]}); meta['sick'] = meta.cls != 'healthy'
    up = np.unique(meta.plant); rp = np.random.RandomState(123).permutation(up); dev_pl = set(rp[:len(up) // 2])   # dev half chooses settings, locked half is the final exam
    meta['half'] = np.where(meta.plant.isin(dev_pl), 'dev', 'locked')
    meta.to_csv(out / 'rocole/meta.csv', index_label='idx')
    return meta


def save_weights(out):
    import torch, torchvision
    sd = torchvision.models.ConvNeXt_Tiny_Weights.IMAGENET1K_V1.get_state_dict(progress=True)
    torch.save(sd, out / 'weights/convnext_tiny_imagenet1k_v1.pth')


def main():
    ap = argparse.ArgumentParser(); ap.add_argument('--raw', default='data/raw'); ap.add_argument('--out', default='data/cropguard'); ap.add_argument('--per-class', type=int, default=200)
    a = ap.parse_args(); raw, out = Path(a.raw), Path(a.out)
    for d in ('plantvillage', 'rocole', 'weights'): (out / d).mkdir(parents=True, exist_ok=True)
    for f in PV_FILES: fetch(PV_BASE + f, raw / 'plantvillage' / f)
    fetch(ROCOLE_URL, raw / 'rocole/rocole.zip')
    t0 = time.time(); pv = build_plantvillage(raw, out, a.per_class); ro = build_rocole(raw, out); save_weights(out)
    classes = sorted(pv.cls.unique())
    files = {str(p.relative_to(out)): p.stat().st_size for p in sorted(out.rglob('*')) if p.is_file() and p.name != 'MANIFEST.json'}
    manifest = {'name': 'cropguard', 'version': 1, 'built': pd.Timestamp.now().isoformat(timespec='seconds'), 'image_size': IMG, 'per_class': a.per_class,
                'classes': classes, 'healthy_class_ids': [i for i, c in enumerate(classes) if c.endswith('healthy')],
                'plantvillage': {'n': len(pv), 'splits': pv.split.value_counts().to_dict(), 'leaf_groups': int(pv.gid.nunique()), 'split_rule': 'GroupShuffleSplit by leaf id, seed 42, 60/20/20',
                                 'licence': 'CC BY-SA 3.0 per the Hugging Face card (some mirrors say CC0): confirm before redistributing', 'source': 'huggingface.co/datasets/mohanty/PlantVillage'},
                'rocole': {'n': len(ro), 'plants': int(ro.plant.nunique()), 'halves': ro.half.value_counts().to_dict(), 'split_rule': 'plants shuffled with RandomState(123), first half = dev',
                           'licence': 'CC BY 4.0', 'source': 'Parraga-Alava et al., Data in Brief 2019, doi 10.17632/c5yvn32dzg.2'},
                'weights': {'convnext_tiny_imagenet1k_v1.pth': 'torchvision ConvNeXt_Tiny_Weights.IMAGENET1K_V1 state_dict'},
                'bytes': files, 'total_gb': round(sum(files.values()) / 1e9, 2)}
    (out / 'MANIFEST.json').write_text(json.dumps(manifest, indent=1))
    print(f'done in {time.time() - t0:.0f}s: {manifest["total_gb"]} GB ->', out.resolve())


if __name__ == '__main__':
    main()
