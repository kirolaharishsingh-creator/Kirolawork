# USP 4 seat slider from the real side photo 13 (front to the right) on the studio backdrop: the seat (cushion + the
# grey seat shell under it) glides forward about 5 cm, holds, and glides back, as in the reference. The armrest, the
# lumbar pad, the back frame and the base stay put; the armrest post and the lumbar stay in front of the seat. What
# the cushion uncovers behind it is the dark top of the seat shell (at the back), the far armrest post, and the
# studio backdrop above. Frame N equals frame 0, so it loops.
# Usage: python3 seat_slide_side.py still.png outdir   (still: the photo on the studio backdrop, any width; layout
#        coordinates below are for a 1600 px wide frame and are scaled)   env: SLIDE px at 1600 (55), FPS, SECONDS (3)
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp import studio_bg
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
im = cv2.imread(src).astype(np.float32); H, W = im.shape[:2]; sc = W / 1600.0; lum = im.mean(2)
FPS = int(os.environ.get('FPS', 24)); SEC = float(os.environ.get('SECONDS', 3)); D = float(os.environ.get('SLIDE', 55)) * sc
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
poly = lambda pts: cv2.fillPoly(np.zeros((H, W), np.uint8), [(np.array(pts, np.float32) * sc).astype(np.int32)], 1) > 0
k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r * sc) + 1, 2 * int(r * sc) + 1))
bg = (studio_bg(W, H)[..., ::-1] * 255).astype(np.float32); bgl = bg.mean(2)

SEAT = [(447, 655), (447, 540), (470, 450), (520, 440), (600, 437), (650, 430), (700, 431), (850, 428), (880, 422),
        (1000, 432), (1060, 451), (1120, 495), (1148, 540), (1152, 575), (1142, 630), (1130, 658), (447, 658)]
ARM = [(698, 330), (772, 330), (768, 345), (742, 470), (738, 530), (748, 565), (742, 605), (712, 645), (707, 716), (588, 716), (605, 688), (700, 365)]   # near armrest post
LUMB = [(360, 380), (640, 380), (640, 420), (600, 438), (580, 455), (540, 490), (500, 515), (450, 536), (400, 520), (360, 540)]
front = (poly(ARM) | poly(LUMB)) & (lum < 170)
# only the cushion slides; the grey seat shell under it (below its bottom line, y ~655) stays with the base
seatpoly = poly(SEAT) & (Y < (655 + 0.006 * (470 - X / sc)) * sc)
# cushion matte: solid inside, soft only where its outline meets the backdrop; behind the post and the lumbar the
# cushion continues (it is hidden there in the photo)
a = np.clip((bgl - lum) / np.maximum(bgl - 60.0, 1), 0, 1) * seatpoly
a = np.where(front & seatpoly, 1.0, a).astype(np.float32)
a = cv2.GaussianBlur(a, (0, 0), 0.5 * sc)
un = (im - (1 - a[..., None]) * bg) / np.maximum(a[..., None], 0.25)
core = ((a > 0.6) & ~front).astype(np.float32)
ext = cv2.GaussianBlur(im * core[..., None], (0, 0), 2 * sc) / np.maximum(cv2.GaussianBlur(core, (0, 0), 2 * sc)[..., None], 1e-3)
layer = np.where(((a > 0.6) & ~front)[..., None], im, np.where(((a > 0.05) & ~front)[..., None], np.clip(un, 0, 255), ext))
# behind the post and the lumbar the cushion is hidden in the photo: the cushion is near uniform along its length,
# so each hidden row takes that row's median fabric colour from the open stretch of cushion, plus the fabric grain
vis = seatpoly & ~front & (X > 752 * sc) & (X < 830 * sc) & (a > 0.9)
rowc = np.zeros((H, 3), np.float32)
for y in range(H):
    m = vis[y]
    if m.sum() > 10: rowc[y] = np.median(im[y, m], 0)
filled = np.where(rowc.sum(1) > 0)[0]
for y in range(H):
    if rowc[y].sum() == 0 and len(filled): rowc[y] = rowc[filled[np.argmin(np.abs(filled - y))]]
rng = np.random.default_rng(4); grain = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.8 * sc)
gs = float(np.std((im - cv2.GaussianBlur(im, (0, 0), 2 * sc))[vis]))
synth = rowc[:, None, :] + (grain / grain.std() * gs * 0.35)[..., None]
behind = cv2.dilate(((poly(ARM) | poly(LUMB)) & seatpoly).astype(np.uint8), k(3)) & seatpoly.astype(np.uint8)
bb = cv2.GaussianBlur(behind.astype(np.float32), (0, 0), 1.5 * sc)[..., None] * seatpoly[..., None]
layer = layer * (1 - bb) + synth * bb
# plate behind the seat: the studio backdrop in the gap behind it, the dark mechanism under it (inpainted)
hole = cv2.dilate((a > 0.05).astype(np.uint8), k(2)) & (~front).astype(np.uint8) & seatpoly.astype(np.uint8)
below = Y > 9e9 * sc                                                     # (all backdrop)
inp = cv2.inpaint(np.clip(im, 0, 255).astype(np.uint8), (hole * below.astype(np.uint8)) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
syn = bg.copy()
# behind the cushion: the far armrest post continues down (its edge line extended); at the back the photo's own
# opening behind the cushion (backdrop above, dark frame below) and, under the armrest, the armrest's dark underside
# are continued in from around the cushion
FARPOST = [(692, 350), (722, 350), (716, 545), (611, 545)]
fp = poly(FARPOST)
syn = np.where(fp[..., None], np.median(im[int(380 * sc):int(420 * sc), int(690 * sc):int(705 * sc)].reshape(-1, 3), 0), syn)
cont = (hole > 0) & poly([(380, 530), (620, 530), (620, 670), (380, 670)]) & ~fp
inp2 = cv2.inpaint(np.clip(im, 0, 255).astype(np.uint8), cv2.dilate(hole, k(1)) * 255, int(6 * sc), cv2.INPAINT_TELEA).astype(np.float32)
# (each row of that opening continued from the photo just left of the cushion's back edge, so it has no hard edges)
xb = 441 * sc
rowfill = cv2.remap(im, np.full((H, W), xb, np.float32), Y, cv2.INTER_LINEAR)
rowfill = cv2.GaussianBlur(rowfill, (1, int(6 * sc) | 1), 0)
syn = np.where(cont[..., None], rowfill, syn)
# along the cushion's top, what lies just above it in the photo (armrest underside or backdrop) carried down a little,
# so the sliding top edge always meets the right thing
has = seatpoly.any(0); top = np.where(has, seatpoly.argmax(0), H).astype(np.float32)
topband = (hole > 0) & (Y < top[None, :] + 14 * sc) & (X > 640 * sc) & ~fp
above = cv2.remap(im, X, np.maximum(top[None, :] - 4 * sc, 0).repeat(H, 0).astype(np.float32), cv2.INTER_LINEAR)
syn = np.where(topband[..., None], above, syn)
# along the cushion's bottom, the grey seat shell just below it carried up, so the sliding bottom edge meets it
bot = np.where(has, H - 1 - seatpoly[::-1].argmax(0), 0).astype(np.float32)
botband = (hole > 0) & (Y > bot[None, :] - 14 * sc) & (X > 560 * sc)
below2 = cv2.remap(im, X, np.minimum(bot[None, :] + 5 * sc, H - 1).repeat(H, 0).astype(np.float32), cv2.INTER_LINEAR)
syn = np.where(botband[..., None], below2, syn)
Rm = cv2.GaussianBlur(hole.astype(np.float32), (0, 0), 0.8 * sc)[..., None]
plate = im * (1 - Rm) + syn * Rm
F = cv2.GaussianBlur(front.astype(np.float32), (0, 0), 0.5 * sc)[..., None]

N = int(round(SEC * FPS)); ease = lambda u: u * u * (3 - 2 * u)
def shift(i):                                                           # rest -> forward -> hold -> back -> rest
    u = i / N; t1, t2, t3 = 0.40, 0.52, 0.92
    if u < t1: return D * ease(u / t1)
    if u < t2: return D
    if u < t3: return D * (1 - ease((u - t2) / (t3 - t2)))
    return 0.0
for i in range(N):
    M = np.float32([[1, 0, shift(i)], [0, 1, 0]])
    L = cv2.warpAffine(layer, M, (W, H), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
    ra = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
    f = L * ra + plate * (1 - ra)
    f = im * F + f * (1 - F)
    if i == 0: f = im
    cv2.imwrite(f'{out}/{i:04d}.png', np.clip(f + 0.5, 0, 255).astype(np.uint8))
cv2.imwrite(f'{out}/plate.png', np.clip(plate + 0.5, 0, 255).astype(np.uint8))
print('ok', N, 'frames')
