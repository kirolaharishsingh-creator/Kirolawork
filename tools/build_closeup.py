# 4K studio keyframe from a full-frame 16:9 close-up photo: replace the white wall (and the wall seen
# through the mesh) with the light grey studio gradient, using a brightness mask.
# Usage: python3 tools/build_closeup.py photo.jpg out.jpg
import sys, cv2, numpy as np
from PIL import Image, ImageOps
im = np.array(ImageOps.exif_transpose(Image.open(sys.argv[1])).convert('RGB')).astype(np.float32)
W, H = 3840, 2160
im = cv2.resize(im, (W, H), interpolation=cv2.INTER_AREA)
lum = im.mean(2)
a = np.clip((185 - lum) / 45, 0, 1)
n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.2).astype(np.uint8))
for i in range(1, n):
    if st[i, 4] < 400: a[lab == i] = 0              # dust specks on the wall
a = cv2.GaussianBlur(a, (0, 0), 1.2)[..., None]
yy, xx = np.mgrid[0:H, 0:W]
d = np.sqrt(((xx - W * 0.5) / W) ** 2 + ((yy - H * 0.35) / H) ** 2)
bg = np.clip(236 - 40 * d, 185, 236)[..., None] * np.array([1, 0.985, 0.97])
out = im * a + bg * (1 - a)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save(sys.argv[2], quality=95)
