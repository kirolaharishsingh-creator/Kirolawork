# Keyframe 2A (headrest down, photo 07) aligned onto 2B (headrest up, photo 37) using hand-picked
# backrest-frame landmarks, so the backrest stays still between the two Kling frames and only the
# headrest moves. The raw photo is warped first (border filled with white wall) and the wall is
# replaced afterwards, so the studio background has no seams.
import cv2, numpy as np
from PIL import Image, ImageOps
W, H = 3840, 2160
raw = np.array(ImageOps.exif_transpose(Image.open('chair_photos/08_usp_headrest/07.jpg')).convert('RGB'))
raw = cv2.resize(raw, (W, H), interpolation=cv2.INTER_AREA)
pa = np.float32([[954, 1080], [3360, 1230], [1065, 954]])     # outer frame top-left, outer frame top-right, rear frame top-left
pb = np.float32([[960, 1356], [3195, 1515], [1065, 1230]])
M, _ = cv2.estimateAffinePartial2D(pa, pb)
im = cv2.warpAffine(raw, M, (W, H), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_CONSTANT, borderValue=(235, 235, 235)).astype(np.float32)
lum = im.mean(2)
a = np.clip((185 - lum) / 45, 0, 1)
n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.2).astype(np.uint8))
for i in range(1, n):
    if st[i, 4] < 400: a[lab == i] = 0
a = cv2.GaussianBlur(a, (0, 0), 1.2)[..., None]
yy, xx = np.mgrid[0:H, 0:W]
d = np.sqrt(((xx - W * 0.5) / W) ** 2 + ((yy - H * 0.35) / H) ** 2)
bg = np.clip(236 - 40 * d, 185, 236)[..., None] * np.array([1, 0.985, 0.97])
out = im * a + bg * (1 - a)
Image.fromarray(np.clip(out, 0, 255).astype(np.uint8)).save('real_photo_keyframes/kf2A_headrest_down_4k_aligned.jpg', quality=95)
