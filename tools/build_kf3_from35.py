# Shot 3 end frame from photo 35 (rear 3/4, headrest UP): whole back of the chair from the headrest top to
# the seat, base cropped out. The crop is wider than the photo, so the sides are padded with studio grey.
# Usage: python3 tools/build_kf3_from35.py out.jpg [mirror]
import sys, cv2, numpy as np
from PIL import Image, ImageOps
im = np.array(ImageOps.exif_transpose(Image.open('chair_photos/07_rear_black_base/35.jpg')).convert('RGB')).astype(np.float32)
h, w = im.shape[:2]
y0, y1 = 950, 3700                         # headrest top (~1150) with headroom, down to the seat underside
ch = y1 - y0; cw = int(ch * 16 / 9); cx = 1650
pad = cw
P = cv2.copyMakeBorder(im, 0, 0, pad, pad, cv2.BORDER_CONSTANT, value=(235, 235, 235))
crop = P[y0:y1, cx - cw // 2 + pad:cx + cw // 2 + pad]
W4, H4 = 3840, 2160
c = cv2.resize(crop, (W4, H4), interpolation=cv2.INTER_AREA)
lum = c.mean(2)
a = np.clip((150 - lum) / 50, 0, 1)
a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))      # drop the thin wall seam
n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.2).astype(np.uint8))
for i in range(1, n):
    if st[i, 4] < 5000: a[lab == i] = 0                                     # dust specks
# keep the bright chrome ring on the spine: bright pixels sitting inside the dark chair silhouette
dark = cv2.morphologyEx((a > 0.5).astype(np.uint8), cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
chrome = (dark > 0) & (lum > 150) & (cv2.blur((a > 0.5).astype(np.float32), (41, 41)) > 0.55)
a = np.maximum(a, chrome.astype(np.float32))
a = cv2.GaussianBlur(a, (0, 0), 1.5)[..., None]
yy, xx = np.mgrid[0:H4, 0:W4]
d = np.sqrt(((xx - W4 * 0.5) / W4) ** 2 + ((yy - H4 * 0.35) / H4) ** 2)
bg = np.clip(236 - 40 * d, 185, 236)[..., None] * np.array([1, 0.985, 0.97])
out = np.clip(c * a + bg * (1 - a), 0, 255).astype(np.uint8)
if 'mirror' in sys.argv[2:]: out = out[:, ::-1]
Image.fromarray(out).save(sys.argv[1], quality=95)
