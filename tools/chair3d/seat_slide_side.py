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

SEAT = [(447, 655), (447, 540), (470, 450), (520, 440), (600, 437), (650, 430), (700, 431), (850, 428), (880, 416),
        (1000, 422), (1070, 438), (1135, 475), (1170, 530), (1178, 580), (1168, 640), (1150, 662), (447, 662)]
ARM = [(698, 330), (772, 330), (768, 345), (760, 450), (753, 545), (750, 560), (725, 600), (698, 640), (700, 716), (588, 716), (604, 690), (620, 640), (640, 580), (660, 520), (680, 460), (700, 410), (700, 365)]   # near armrest post
LUMB = [(360, 380), (640, 380), (640, 420), (600, 438), (580, 455), (540, 490), (500, 515), (450, 536), (400, 520), (360, 540)]
front = (poly(ARM) | poly(LUMB)) & (lum < 170)
# only the cushion slides; the grey seat shell under it (below its bottom line, y ~655) stays with the base
seatpoly = poly(SEAT) & (Y < (655 + 0.006 * (470 - X / sc)) * sc)
# cushion matte: solid inside, soft only where its outline meets the backdrop; behind the post and the lumbar the
# cushion continues (it is hidden there in the photo)
a = np.clip((bgl - lum) / np.maximum(bgl - 60.0, 1), 0, 1) * seatpoly
a = np.where(front & seatpoly, 1.0, a).astype(np.float32)
# along the outline against the backdrop the matte is smoothed a touch (no speckle from the fabric's noise)
nearbg = cv2.dilate((cv2.GaussianBlur(lum, (0, 0), 1.5 * sc) > 90).astype(np.uint8), k(2)) > 0
a = np.where(nearbg & seatpoly, cv2.GaussianBlur(a, (0, 0), 0.8 * sc), a)
un = (im - (1 - a[..., None]) * bg) / np.maximum(a[..., None], 0.25)
core = ((a > 0.6) & ~front).astype(np.float32)
ext = cv2.GaussianBlur(im * core[..., None], (0, 0), 2 * sc) / np.maximum(cv2.GaussianBlur(core, (0, 0), 2 * sc)[..., None], 1e-3)
layer = np.where(((a > 0.6) & ~front)[..., None], im, np.where(((a > 0.05) & ~front)[..., None], np.clip(un, 0, 255), ext))
# behind the post and the lumbar the cushion is hidden in the photo: the cushion is near uniform along its length,
# so each hidden row takes that row's median fabric colour from the open stretch of cushion, plus the fabric grain
xl = np.interp(np.arange(H) / sc, [410, 460, 520, 580, 640, 690], [700, 680, 660, 640, 620, 605]) * sc
xr = np.interp(np.arange(H) / sc, [345, 450, 545, 560, 600, 640], [768, 760, 753, 750, 725, 698]) * sc
def side(xc, d0, d1):
    c = np.zeros((H, 3), np.float32)
    for y in range(H):
        xs = np.arange(int(xc[y] + d0 * sc), int(xc[y] + d1 * sc)); xs = xs[(xs >= 0) & (xs < W)]
        ok = xs[seatpoly[y, xs] & ~front[y, xs]] if len(xs) else xs
        c[y] = np.median(im[y, ok], 0) if len(ok) > 2 else np.nan
    return c
cl, cr = side(xl, -16, -5), side(xr, 5, 16)
cl = np.where(np.isnan(cl), cr, cl); cr = np.where(np.isnan(cr), cl, cr)
cl = np.nan_to_num(cv2.GaussianBlur(cl[:, None, :], (1, int(5 * sc) | 1), 0)[:, 0]); cr = np.nan_to_num(cv2.GaussianBlur(cr[:, None, :], (1, int(5 * sc) | 1), 0)[:, 0])
tt = np.clip((X - xl[:, None]) / np.maximum(xr - xl, 1)[:, None], 0, 1)[..., None]
rowmix = cl[:, None, :] * (1 - tt) + cr[:, None, :] * tt
vis = seatpoly & ~front & (X > 752 * sc) & (X < 830 * sc)
rng = np.random.default_rng(4); grain = cv2.GaussianBlur(rng.normal(0, 1, (H, W)).astype(np.float32), (0, 0), 0.8 * sc)
gs = float(np.std((im - cv2.GaussianBlur(im, (0, 0), 2 * sc))[vis]))
synth = rowmix + (grain / grain.std() * gs * 0.35)[..., None]
behind = cv2.dilate(((poly(ARM) | poly(LUMB)) & seatpoly).astype(np.uint8), k(3)) & seatpoly.astype(np.uint8)
bb = cv2.GaussianBlur(behind.astype(np.float32), (0, 0), 1.5 * sc)[..., None] * seatpoly[..., None]
layer = layer * (1 - bb) + synth * bb
# plate behind the seat: the studio backdrop in the gap behind it, the dark mechanism under it (inpainted)
hole = (seatpoly & ~front).astype(np.uint8)
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
# (behind the cushion's back there is open space down to the frame's lower arm: backdrop, then the arm's own colour
# carried in from below)
armfill = cv2.inpaint(np.clip(im, 0, 255).astype(np.uint8), (hole * 255).astype(np.uint8), int(4 * sc), cv2.INPAINT_TELEA).astype(np.float32)
tb = np.clip((Y / sc - 653) / 4, 0, 1)[..., None]
# looking into the opening one sees the dark far side of the back frame and the seat base in shadow, lit a little
# from above
GAPV = float(os.environ.get('GAPV', 0.18))
tv = np.clip((Y / sc - 545) / 90, 0, 1)[..., None]
gapc = (bg * GAPV * (1 - 0.5 * tv)) * (1 - tb) + armfill * tb
syn = np.where(cont[..., None], gapc, syn)
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
Rm = hole.astype(np.float32)[..., None]                                 # hard: no seam where plate meets photo
plate = im * (1 - Rm) + syn * Rm
F = cv2.GaussianBlur(front.astype(np.float32), (0, 0), 0.5 * sc)[..., None]
contf = cv2.GaussianBlur(cont.astype(np.float32), (0, 0), 2 * sc)

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
    # a soft shadow of the cushion's back edge into the opening behind it
    dsh = (447 * sc + shift(i)) - X
    shade = 1 - 0.5 * np.clip(1 - dsh / (30 * sc), 0, 1) * (dsh > 0) * contf
    f = L * ra + plate * shade[..., None] * (1 - ra)
    f = im * F + f * (1 - F)
    if i == 0: f = im
    cv2.imwrite(f'{out}/{i:04d}.png', np.clip(f + 0.5, 0, 255).astype(np.uint8))
cv2.imwrite(f'{out}/plate.png', np.clip(plate + 0.5, 0, 255).astype(np.uint8))
print('ok', N, 'frames')
