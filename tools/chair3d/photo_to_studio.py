# Put a real chair photo on the same studio backdrop as the 3D renders (comp.studio_bg): the photo's wall/floor is
# estimated as a smooth brightness field from the bright pixels, and every pixel gets a soft chair matte from how much
# darker than that field it is. Light seen through the mesh and the wall around the chair become the studio backdrop;
# small dark specks (wall seams, dust) are dropped.
# Usage: python photo_to_studio.py in.jpg out.jpg   (env: W output width (default 2880), DARK ratio (0.62))
import os, sys, numpy as np
from PIL import Image, ImageOps, ImageFilter
from scipy import ndimage
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp import studio_bg

src, dst = sys.argv[1], sys.argv[2]
W = int(os.environ.get('W', 2880)); DARK = float(os.environ.get('DARK', 0.55))
im = ImageOps.exif_transpose(Image.open(src)).convert('RGB'); H = round(im.height * W / im.width)
f = np.asarray(im.resize((W, H), Image.LANCZOS), np.float32) / 255
lum = f.mean(-1)
blur = lambda x, r: np.asarray(Image.fromarray(np.clip(x * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), np.float32) / 255
# wall: normalized convolution over clearly bright pixels (wide radius so it reaches across the chair)
wgt = (lum > np.percentile(lum, 75) * 0.9).astype(np.float32)
for _ in range(3):
    ws = blur(wgt, W / 8); wl = blur(lum * wgt, W / 8) / np.maximum(ws, 1e-3)
    wgt = (lum > wl * 0.85).astype(np.float32)
# chair matte from how much darker than the wall each pixel is: 1 = chair, 0 = wall; soft in between
t = lum / np.maximum(wl, 0.05); lo, hi = DARK, float(os.environ.get('LIGHT_T', 0.88))
a = np.clip((hi - t) / (hi - lo), 0, 1); a = a * a * (3 - 2 * a)
# keep only big dark regions (drops wall seams, dust and floor marks)
# chrome (legs, rings) is as bright as the floor but full of highlights and dark streaks: anything that differs
# clearly from the smooth wall field counts too, and closing joins its streaks into solid bars
CHROME = float(os.environ.get('CHROME', 0.10))
dev = np.abs(t - 1) > CHROME if CHROME > 0 else np.zeros_like(t, bool)
core = ndimage.binary_closing((a > 0.5) | dev, iterations=int(os.environ.get('CLOSE_IT', 6))); lab, n = ndimage.label(core); sz = ndimage.sum(core, lab, range(1, n + 1))
big = ndimage.binary_dilation(np.isin(lab, 1 + np.nonzero(sz > core.size * 0.002)[0]), iterations=8)
keep = ndimage.binary_erosion(big, iterations=8) & ndimage.binary_dilation(dev, iterations=10)   # chrome and its streaks
a = np.maximum(a, blur(keep.astype(np.float32), 2))
a = (a * blur(big.astype(np.float32), 3))[..., None]
bg = np.ones((H, W, 3), np.float32) if os.environ.get('BG') == 'white' else studio_bg(W, H).astype(np.float32)   # BG=white: product cut-out (e.g. Tripo input)
out = f * a + bg * (1 - a)
chair = a[..., 0] > 0.5
Image.fromarray((out * 255 + 0.5).astype(np.uint8)).save(dst, quality=95)
print('ok', dst, W, H, 'chair %.1f%%' % (100 * chair.mean()))
