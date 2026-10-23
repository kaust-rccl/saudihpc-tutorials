# ---- candidate augmentation regimes (all take a lab leaf: uint8 HxWx3 image + boolean leaf mask)
def _flip_rot90(a, rng):
    a = np.rot90(a, int(rng.integers(0, 4)))
    return a[:, ::-1] if rng.random() < .5 else a

def random_resized_crop(img, rng, scale=(.25, 1.0), ratio=(3 / 4, 4 / 3)):
    H = W = img.shape[0]; area = H * W * rng.uniform(*scale); ar = np.exp(rng.uniform(np.log(ratio[0]), np.log(ratio[1])))
    w, h = min(W, int(round(np.sqrt(area * ar)))), min(H, int(round(np.sqrt(area / ar))))
    x0, y0 = int(rng.integers(0, W - w + 1)), int(rng.integers(0, H - h + 1))
    return np.asarray(Image.fromarray(np.ascontiguousarray(img)).crop((x0, y0, x0 + w, y0 + h)).resize((W, H), Image.BICUBIC))

def photometric(img, rng, s=1.0):
    im = Image.fromarray(np.ascontiguousarray(img))
    im = ImageEnhance.Brightness(im).enhance(1 + rng.uniform(-.45, .45) * s); im = ImageEnhance.Contrast(im).enhance(1 + rng.uniform(-.4, .4) * s)
    im = ImageEnhance.Color(im).enhance(1 + rng.uniform(-.4, .4) * s)
    h, sat, v = im.convert('HSV').split(); h = Image.fromarray(((np.asarray(h).astype(int) + int(rng.uniform(-10, 10) * s)) % 256).astype('uint8'))
    im = Image.merge('HSV', (h, sat, v)).convert('RGB')
    a = np.asarray(im).astype('float32') / 255; a = a ** rng.uniform(1 - .35 * s, 1 + .35 * s)                       # gamma (exposure)
    a *= np.array([1 + rng.uniform(-.12, .12) * s, 1, 1 + rng.uniform(-.12, .12) * s])                                  # colour temperature
    im = Image.fromarray((np.clip(a, 0, 1) * 255).astype('uint8'))
    if rng.random() < .5 * s: im = im.filter(ImageFilter.GaussianBlur(rng.uniform(.3, 1.3)))
    return np.asarray(im)

def bg_swap(img, mask, other_img, other_mask, rng):
    """Put the leaf on another photo's lab backdrop (its leaf painted out), so the backdrop no longer tells you the class."""
    bgm = np.asarray(Image.fromarray((other_mask * 255).astype('uint8')).filter(ImageFilter.MaxFilter(9))) > 0
    bg = other_img.copy(); bg[bgm] = bg[~bgm].mean(0) if (~bgm).any() else bg.mean((0, 1))
    size = img.shape[0]; ys, xs = np.where(mask)
    if len(ys) < 50: return _flip_rot90(img, rng)
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    leaf = Image.fromarray(img).crop(box); m = Image.fromarray((mask * 255).astype('uint8')).crop(box)
    ang = rng.uniform(0, 360); leaf, m = leaf.rotate(ang, expand=True, resample=Image.BICUBIC), m.rotate(ang, expand=True, resample=Image.BICUBIC)
    sc = rng.uniform(.55, .98) * size / max(leaf.size); new = (max(8, int(leaf.size[0] * sc)), max(8, int(leaf.size[1] * sc)))
    leaf, m = leaf.resize(new, Image.BICUBIC), m.resize(new, Image.BICUBIC).filter(ImageFilter.GaussianBlur(.8))
    out = Image.fromarray(bg); out.paste(leaf, (int(rng.uniform(0, size - new[0])), int(rng.uniform(0, size - new[1]))), m); return np.asarray(out)

LEGACY = {'photo': 'flip+photo1', 'rrc': 'flip+rrc0.25+photo1', 'bgswap': 'bgswap', 'bgswap+rrc': 'bgswap+rrc0.25+photo1'}

def make_regime(name, i, idx_pool, rng_seed):
    """One augmented copy (uint8 224x224x3) of training image i. `name` is a recipe of ops joined by '+':
    flip (rot90/mirror) | rrcS (random resized crop, min area S) | photoS (photometric jitter, strength S) | bgswapX | synthS (field synthesis, strength S)."""
    rng = np.random.default_rng(rng_seed); img, mask = pv['img'][i], pv['mask'][i]
    for op in LEGACY.get(name, name).split('+'):
        if op == 'none': continue
        if op == 'flip': img = _flip_rot90(img, rng)
        elif op.startswith('rrc'): img = random_resized_crop(img, rng, scale=(float(op[3:]), 1.0))
        elif op.startswith('photo'): img = photometric(img, rng, float(op[5:])) if float(op[5:]) > 0 else img
        elif op.startswith('synth'): img = synth_field(img, mask, rng_seed, float(op[5:]))
        elif op.startswith('bgswap'):
            j = int(idx_pool[rng.integers(len(idx_pool))]); img = bg_swap(img, mask, pv['img'][j], pv['mask'][j], rng)
        else: raise ValueError(op)
    return img
