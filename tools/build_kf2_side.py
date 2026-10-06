# Side-angle headrest close-ups (facing right): photo 23 = headrest down, photo 22 = headrest up.
# 22 is shifted onto 23 using hand-picked anchors fixed to the spine (grey bracket nub, top of the
# spine cap), so the spine stays still and only the headrest rises between the two frames.
import glob, cv2, numpy as np
from PIL import Image, ImageOps
def load(n):
    return np.array(ImageOps.exif_transpose(Image.open(glob.glob(f'chair_photos/*/{n}.jpg')[0])).convert('RGB'))
A, B = load('23'), load('22')
h, w = A.shape[:2]
pa = np.float32([[575, 1708], [850, 1655]]); pb = np.float32([[625, 1745], [880, 1695]])
dx, dy = (pa - pb).mean(0)
print('shift 22 by', round(dx, 1), round(dy, 1))
B = cv2.warpAffine(B, np.float32([[1, 0, dx], [0, 1, dy]]), (w, h), flags=cv2.INTER_LANCZOS4, borderMode=cv2.BORDER_REPLICATE)
# 16:9 crop centred on the headrest, with headroom for the raised position; pad with wall where needed
ch = 1250; cw = int(ch * 16 / 9)
cx, cy = 720, 1640                         # between the rear rail and the loop, at the bracket
x0, y0 = cx - int(cw * 0.5), cy - int(ch * 0.55)
pad = 1500
W4, H4 = 3840, 2160
yy, xx = np.mgrid[0:H4, 0:W4]
d = np.sqrt(((xx - W4 * 0.5) / W4) ** 2 + ((yy - H4 * 0.35) / H4) ** 2)
bg = np.clip(236 - 40 * d, 185, 236)[..., None] * np.array([1, 0.985, 0.97])
for im, out in [(A, 'kf2C_headrest_side_down_4k.jpg'), (B, 'kf2D_headrest_side_up_4k.jpg')]:
    P = cv2.copyMakeBorder(im, pad, pad, pad, pad, cv2.BORDER_CONSTANT, value=(230, 230, 230))
    c = cv2.resize(P[y0 + pad:y0 + pad + ch, x0 + pad:x0 + pad + cw], (W4, H4), interpolation=cv2.INTER_LANCZOS4).astype(np.float32)
    a = np.clip((150 - c.mean(2)) / 50, 0, 1)
    a = cv2.morphologyEx(a, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))       # drop the thin wall seam
    n, lab, st, _ = cv2.connectedComponentsWithStats((a > 0.2).astype(np.uint8))
    for i in range(1, n):
        if st[i, 4] < 5000: a[lab == i] = 0                                  # dust specks
    a = cv2.GaussianBlur(a, (0, 0), 1.5)[..., None]
    Image.fromarray(np.clip(c * a + bg * (1 - a), 0, 255).astype(np.uint8)).save('real_photo_keyframes/' + out, quality=95)
