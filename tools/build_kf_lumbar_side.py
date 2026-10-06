# Lumbar end frame for the headrest -> lumbar tilt-down: same photo (23), same zoom and wall treatment as
# kf2C_headrest_side_down_4k, so the camera only tilts down the spine onto the C-shaped lumbar pad.
import glob, cv2, numpy as np
from PIL import Image, ImageOps
A = np.array(ImageOps.exif_transpose(Image.open(glob.glob('chair_photos/*/23.jpg')[0])).convert('RGB'))
ch = 1250; cw = int(ch * 16 / 9)
cx, cy = 800, 2980                         # lumbar pad + pivot arm, slightly right of the kf2C centre (720)
x0, y0 = cx - int(cw * 0.5), cy - int(ch * 0.5)
pad = 1500
W4, H4 = 3840, 2160
yy, xx = np.mgrid[0:H4, 0:W4]
d = np.sqrt(((xx - W4 * 0.5) / W4) ** 2 + ((yy - H4 * 0.35) / H4) ** 2)
bg = np.clip(236 - 40 * d, 185, 236)[..., None] * np.array([1, 0.985, 0.97])
P = cv2.copyMakeBorder(A, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(230, 230, 230))
c = cv2.resize(P[y0 + pad:y0 + pad + ch, x0 + pad:x0 + pad + cw], (W4, H4), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
a = np.clip((150 - c.mean(2)) / 50, 0, 1)
a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))
n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.2).astype(np.uint8))
for i in range(1, n):
    if st[i, 4] < 5000: a[lab == i] = 0
a = cv2.GaussianBlur(a, (0, 0), 1.5)[..., None]
Image.fromarray(np.clip(c * a + bg * (1 - a), 0, 255).astype(np.uint8)).save('real_photo_keyframes/kf4C_lumbar_side_right_4k.jpg', quality=95)
