import io
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw

PALETTES = {'soil': ((50, 34, 22), (140, 100, 66)), 'dry soil': ((110, 90, 65), (190, 165, 125)),
            'grass': ((28, 60, 20), (110, 160, 60)), 'gravel': ((80, 80, 78), (180, 175, 165)),
            'litter': ((40, 28, 18), (130, 95, 55)), 'weeds': ((35, 70, 25), (140, 170, 70))}
CLUTTER = np.array([[50, 110, 40], [90, 130, 50], [120, 100, 50], [70, 50, 30], [30, 70, 30]])

def _noise(rng, size, octaves):
    out = np.zeros((size, size), dtype=np.float32)
    for s, w in octaves:
        out += w * np.asarray(Image.fromarray((rng.random((s, s)) * 255).astype('uint8')).resize((size, size), Image.BICUBIC), dtype=np.float32) / 255
    out -= out.min()
    return out / (out.max() + 1e-6)

def _blob(size, rng, radius, blur):
    layer = Image.new('L', (size, size), 0)
    cx, cy = rng.integers(0, size, 2); a, b = radius * rng.uniform(.6, 1.4, 2)
    ImageDraw.Draw(layer).ellipse((cx - a, cy - b, cx + a, cy + b), fill=255)
    return layer.rotate(float(rng.uniform(0, 180)), center=(cx, cy)).filter(ImageFilter.GaussianBlur(blur))

def make_background(rng, size, strength):
    lo, hi = (np.array(v) for v in PALETTES[rng.choice(list(PALETTES))])
    n = _noise(rng, size, ((3, .25), (9, .25), (30, .25), (100, .15), (224, .10)))
    img = Image.fromarray((lo + n[..., None] * (hi - lo)).astype('uint8'))
    for _ in range(int(rng.integers(2, 3 + int(10 * strength)))):           # clutter: other leaves, stems, litter
        col = np.clip(CLUTTER[rng.integers(len(CLUTTER))] + rng.integers(-20, 20, 3), 0, 255)
        alpha = _blob(size, rng, size / 8, 1.2).point(lambda v: int(v * rng.uniform(.5, .9)))
        img.paste(Image.new('RGB', (size, size), tuple(int(c) for c in col)), (0, 0), alpha)
    return img

def synth_field(img, mask, seed, strength=1.0):
    """Paste a lab leaf (uint8 HxWx3 + boolean mask) into a randomised field-like photo. `strength` scales every effect (0..1)."""
    rng = np.random.default_rng(seed); size = img.shape[0]; s = float(strength)
    ys, xs = np.where(mask)
    if len(ys) < 50: ys, xs = np.array([0, size - 1]), np.array([0, size - 1])
    box = (xs.min(), ys.min(), xs.max() + 1, ys.max() + 1)
    leaf = Image.fromarray(img).crop(box); m = Image.fromarray((mask * 255).astype('uint8')).crop(box)
    ang = rng.uniform(0, 360)
    leaf, m = leaf.rotate(ang, expand=True, resample=Image.BICUBIC), m.rotate(ang, expand=True, resample=Image.BICUBIC)
    sc = rng.uniform(.45, .95) * size / max(leaf.size); new = (max(8, int(leaf.size[0] * sc)), max(8, int(leaf.size[1] * sc)))
    leaf, m = leaf.resize(new, Image.BICUBIC), m.resize(new, Image.BICUBIC).filter(ImageFilter.GaussianBlur(.8))
    ox, oy = (int(rng.uniform(-.05 * size, size - new[0] + .05 * size)), int(rng.uniform(-.05 * size, size - new[1] + .05 * size)))
    out = make_background(rng, size, s)
    sh = Image.new('L', (size, size), 0); sh.paste(m, (ox + int(rng.integers(-8, 9)), oy + int(rng.integers(-8, 9))))   # soft contact shadow
    out.paste(Image.new('RGB', (size, size), (0, 0, 0)), (0, 0), sh.filter(ImageFilter.GaussianBlur(4)).point(lambda v: int(v * rng.uniform(.3, .6))))
    out.paste(leaf, (ox, oy), m)
    a = np.asarray(out).astype('float32')
    if rng.random() < .6 * s:                                                  # occlusion by other foliage
        for _ in range(int(rng.integers(1, 3))):
            col = CLUTTER[rng.integers(len(CLUTTER))]; al = np.asarray(_blob(size, rng, size / 12, 1.0), dtype='float32')[..., None] / 255 * .9
            a = a * (1 - al) + col * al
    ramp = np.linspace(1 - .45 * s * rng.random(), 1 + .45 * s * rng.random(), size)                     # directional light
    a *= (ramp[None, :] if rng.random() < .5 else ramp[:, None])[..., None]
    if rng.random() < .5 * s:                                                  # cast shadow
        poly = Image.new('L', (size, size), 0)
        ImageDraw.Draw(poly).polygon([tuple(int(v) for v in rng.integers(0, size, 2)) for _ in range(4)], fill=255)
        a *= (1 - .55 * np.asarray(poly.filter(ImageFilter.GaussianBlur(10)), dtype='float32') / 255)[..., None]
    a *= np.array([1 + rng.uniform(-.1, .1) * s, 1, 1 + rng.uniform(-.1, .1) * s])                        # colour temperature
    out = Image.fromarray(np.clip(a, 0, 255).astype('uint8'))
    out = ImageEnhance.Brightness(out).enhance(1 + (rng.uniform(.6, 1.4) - 1) * s)
    out = ImageEnhance.Contrast(out).enhance(1 + (rng.uniform(.7, 1.3) - 1) * s)
    f = 1 - (1 - rng.uniform(.5, 1.0)) * s                                      # low-res camera
    out = out.resize((max(32, int(size * f)),) * 2, Image.BILINEAR).resize((size, size), Image.BILINEAR).filter(ImageFilter.GaussianBlur(rng.uniform(0, 1.4) * s))
    a = np.asarray(out).astype('float32') + rng.normal(0, rng.uniform(0, 7) * s, (size, size, 3))
    buf = io.BytesIO(); Image.fromarray(np.clip(a, 0, 255).astype('uint8')).save(buf, 'JPEG', quality=int(rng.integers(int(90 - 60 * s), 92)))
    return np.asarray(Image.open(io.BytesIO(buf.getvalue())).convert('RGB'))
