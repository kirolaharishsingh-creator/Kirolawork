# Keyframe 7B: lever close-up from full-res photo 15 (both levers visible end to end)
import sys, cv2, numpy as np
from PIL import Image, ImageOps
sys.path.insert(0, 'tools'); from build_keyframe import cutout
src = np.array(ImageOps.exif_transpose(Image.open('chair_photos/01_front_chrome/15.jpg')).convert('RGB')).astype(np.float32)
h, w = src.shape[:2]
X0, Y0, CW = [int(v) for v in sys.argv[1:4]] if len(sys.argv) > 3 else (350, 3240, 2000)
CH = CW * 9 // 16
crop = src[Y0:Y0 + CH, X0:X0 + CW]
# dark parts (levers, column, seat) from a sharp luminance mask; chrome base from the remove.bg cutout alpha
lum = crop.mean(2)
dark = np.clip((165 - lum) / 45, 0, 1)
n, lab, st, _ = cv2.connectedComponentsWithStats((dark > 0.2).astype(np.uint8))
for i in range(1, n):
    if st[i, 4] < 300: dark[lab == i] = 0                                     # dust specks on the wall
_, a = cutout('real_photo_keyframes/photo15_cutout_white.jpg')
a = cv2.resize(a, (w, h), interpolation=cv2.INTER_CUBIC)[Y0:Y0 + CH, X0:X0 + CW]
a = (a > 0.5).astype(np.float32)                                           # firm chrome mask
hub = int(float(sys.argv[4]) * CH) if len(sys.argv) > 4 else int(0.62 * CH)  # chrome starts below this row
a[:hub] = 0
alpha = np.maximum(dark, a)
W, H = 3840, 2160
crop = cv2.resize(crop, (W, H), interpolation=cv2.INTER_LANCZOS4)
alpha = cv2.GaussianBlur(cv2.resize(alpha, (W, H), interpolation=cv2.INTER_CUBIC), (0, 0), 1.0)[..., None]
yy, xx = np.mgrid[0:H, 0:W]
d = np.sqrt(((xx - W * 0.5) / W) ** 2 + ((yy - H * 0.35) / H) ** 2)
bg = np.clip(236 - 40 * d, 185, 236)[..., None] * np.array([1, 0.985, 0.97])
out = crop * alpha + bg * (1 - alpha)
out = np.clip(out, 0, 255).astype(np.uint8)
Image.fromarray(out).save('real_photo_keyframes/kf7B_levers_4k.jpg', quality=95)
