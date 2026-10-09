# USP 4 end frame from photo 23 on the studio backdrop: the cushion slid forward D px (at 1600 wide); the armrest post
# and lumbar stay in front, the cushion hidden behind them is filled row by row from the fabric on either side, and
# what the cushion uncovers is inpainted from its surroundings. Usage: python3 seat_end_frame_23.py <dir with still23.png> [D]
import cv2, numpy as np, sys
S = sys.argv[1]; D = float(sys.argv[2]) if len(sys.argv) > 2 else 38
im = cv2.imread(S + '/still23.png'); H, W = im.shape[:2]; sc = W / 1600
f = im.astype(np.float32); lum = f.mean(2)
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
poly = lambda p: cv2.fillPoly(np.zeros((H, W), np.uint8), [(np.array(p, np.float32) * sc).astype(np.int32)], 1) > 0
CUSH = [(372, 640), (380, 600), (420, 570), (480, 530), (560, 505), (660, 492), (800, 482), (950, 482), (1100, 490), (1200, 520), (1250, 560), (1275, 620), (1280, 680), (1272, 740), (1250, 775), (1150, 782), (900, 784), (700, 782), (560, 780), (440, 776), (380, 770)]
POST = [(700, 352), (790, 352), (790, 412), (712, 770), (716, 850), (565, 850), (600, 800), (622, 770), (712, 420), (700, 400)]
LUMB = [(270, 590), (300, 560), (380, 520), (450, 470), (470, 440), (560, 430), (575, 480), (540, 560), (470, 600), (420, 625), (370, 640), (300, 640)]
front = (poly(POST) | poly(LUMB)) & (lum < 200)
cush = poly(CUSH)
a = cush.astype(np.float32)
# soft matte where cushion meets backdrop
bgl = np.median(lum[int(450 * sc):int(470 * sc), int(1300 * sc):int(1400 * sc)])
lm = np.clip((bgl - lum) / (bgl - 40), 0, 1)
nearbg = cv2.dilate((lum > bgl - 25).astype(np.uint8), np.ones((int(6 * sc) | 1,) * 2, np.uint8)) > 0
a = np.where(nearbg, lm * cush, a).astype(np.float32)
# cushion layer: hidden parts (behind post/lumbar) inpainted from the cushion around them
hid = (cush & front).astype(np.uint8)
hidd = (cv2.dilate(hid, np.ones((int(18 * sc) | 1,) * 2, np.uint8)) & cush).astype(np.uint8)
layer = f.copy()
rng = np.random.default_rng(3)
for y in range(H):
    xs = np.where(hidd[y] > 0)[0]
    if not len(xs): continue
    # contiguous runs
    runs = np.split(xs, np.where(np.diff(xs) > 1)[0] + 1)
    for r in runs:
        x0, x1 = r[0], r[-1]
        L = [x for x in range(max(x0 - 16, 0), x0 - 3) if cush[y, x] and not front[y, x] and not hidd[y, x]]
        R = [x for x in range(x1 + 4, min(x1 + 17, W)) if cush[y, x] and not front[y, x] and not hidd[y, x]]
        cl = np.median(f[y, L], 0) if len(L) > 2 else None
        cr = np.median(f[y, R], 0) if len(R) > 2 else None
        if cl is None and cr is None: continue
        if cl is None: cl = cr
        if cr is None: cr = cl
        t = np.linspace(0, 1, x1 - x0 + 1)[:, None]
        layer[y, x0:x1 + 1] = cl * (1 - t) + cr * t
layer = np.where((hidd > 0)[..., None], cv2.GaussianBlur(layer, (1, int(5 * sc) | 1), 0), layer)
g = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.8); g = g / g.std() * 1.0
layer = np.where((hidd > 0)[..., None], layer + g[..., None], layer)
a = np.where(hidd > 0, 1.0, a)
# plate: remove the cushion, fill from surroundings
hole = cv2.dilate((cush & ~front).astype(np.uint8), np.ones((5, 5), np.uint8))
plate = cv2.inpaint(im, hole * 255, int(12 * sc), cv2.INPAINT_TELEA).astype(np.float32)
d = D * sc
M = np.float32([[1, 0, d], [0, 1, 0]])
Lm = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
am = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
out = Lm * am + plate * (1 - am)
F = cv2.GaussianBlur(front.astype(np.float32), (0, 0), 0.6)[..., None]
out = f * F + out * (1 - F)
cv2.imwrite(S + '/end23.png', np.clip(out + 0.5, 0, 255).astype(np.uint8))
print('ok', d)
