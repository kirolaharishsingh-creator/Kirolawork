# USP 4 seat slider from the real side photo 13 (front to the right) on the studio backdrop: the seat (cushion + the
# grey seat shell under it) glides forward about 5 cm, holds, and glides back, as in the reference. The armrest, the
# lumbar pad, the back frame and the base stay put; the armrest post and the lumbar stay in front of the seat. What
# the cushion uncovers behind it is the dark top of the seat shell (at the back), the far armrest post, and the
# studio backdrop above. Frame N equals frame 0, so it loops.
# Usage: python3 seat_slide_side.py still.png outdir   (still: the photo on the studio backdrop, any width; layout
#        coordinates below are for a 1600 px wide frame and are scaled)   env: SLIDE px at 1600 (42), FPS, SECONDS (3)
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp import studio_bg
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
im = cv2.imread(src).astype(np.float32); H, W = im.shape[:2]; sc = W / 1600.0; lum = im.mean(2)
FPS = int(os.environ.get('FPS', 24)); SEC = float(os.environ.get('SECONDS', 3)); D = float(os.environ.get('SLIDE', 42)) * sc
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
poly = lambda pts: cv2.fillPoly(np.zeros((H, W), np.uint8), [(np.array(pts, np.float32) * sc).astype(np.int32)], 1) > 0
k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * int(r * sc) + 1, 2 * int(r * sc) + 1))
bg = (studio_bg(W, H)[..., ::-1] * 255).astype(np.float32); bgl = bg.mean(2)

SEAT = [(447, 655), (447, 540), (452, 470), (470, 439), (520, 435), (600, 431), (650, 424), (700, 426), (850, 424), (880, 412),
        (1000, 417), (1070, 433), (1135, 470), (1170, 527), (1178, 580), (1168, 640), (1150, 662), (447, 662)]
ARM = [(698, 330), (772, 330), (768, 345), (760, 450), (753, 545), (750, 560), (725, 600), (698, 640), (700, 716), (588, 716), (604, 690), (620, 640), (640, 580), (660, 520), (680, 460), (700, 410), (700, 365)]   # near armrest post
LUMB = [(360, 380), (640, 380), (640, 420), (600, 438), (580, 455), (540, 490), (500, 515), (450, 536), (400, 520), (360, 540)]
front = (poly(ARM) | poly(LUMB)) & (lum < 170)
# only the cushion slides; the grey seat shell under it (below its bottom line, y ~655) stays with the base
seatpoly = poly(SEAT) & (Y < (655 + 0.006 * (470 - X / sc)) * sc)
# the cushion's top edge as measured where backdrop lies above it (subpixel), one smooth curve along its whole length,
# continued straight behind the lumbar; the moving cushion is cut along it with a one-pixel anti-aliased edge
TX = [447, 600, 620, 640, 650, 870, 890, 930, 970, 1010, 1050, 1090, 1110]
TY = [452, 436.2, 433.3, 430.8, 429.7, 420.8, 421.2, 423.4, 427.0, 433.4, 445.6, 464.6, 480.4]
topc = (np.interp(X / sc, TX, TY) * sc).astype(np.float32)
topzone = (X / sc >= 447) & (X / sc <= 880)
seatpoly = np.where(topzone, poly(SEAT + [(447, 400), (1110, 400)][:0]) | ((Y >= topc - 1) & poly([(447, 400), (1110, 400), (1110, 662), (447, 662)])), seatpoly)
seatpoly &= (Y < (655 + 0.006 * (470 - X / sc)) * sc)
seatpoly &= ~(topzone & (Y < topc - 1))
# cushion matte: solid inside, soft only where its outline meets the backdrop; behind the post and the lumbar the
# cushion continues (it is hidden there in the photo)
a = np.clip((bgl - lum) / np.maximum(bgl - 60.0, 1), 0, 1) * seatpoly
a = np.where(front & seatpoly, 1.0, a).astype(np.float32)
# along the outline against the backdrop the matte is smoothed a touch (no speckle from the fabric's noise)
nearbg = cv2.dilate((cv2.GaussianBlur(lum, (0, 0), 1.5 * sc) > 90).astype(np.uint8), k(2)) > 0
# away from the backdrop (under the armrest pad, against the frame) the cushion is cut along its outline, solid
spoly = cv2.GaussianBlur(seatpoly.astype(np.float32), (0, 0), 0.7 * sc)
a = np.where(nearbg, cv2.GaussianBlur(a, (0, 0), 0.8 * sc) * seatpoly, spoly).astype(np.float32)
a = np.where(a < 0.1, 0, a).astype(np.float32)                         # no faint halo along the outline
ta = np.clip(Y - topc + 0.5, 0, 1)
topband = topzone & (Y < topc + 3 * sc)
a = np.where(topband, ta, a).astype(np.float32)
un = (im - (1 - a[..., None]) * bg) / np.maximum(a[..., None], 0.25)
core = ((a > 0.6) & ~front).astype(np.float32)
ext = cv2.GaussianBlur(im * core[..., None], (0, 0), 2 * sc) / np.maximum(cv2.GaussianBlur(core, (0, 0), 2 * sc)[..., None], 1e-3)
layer = np.where(((a > 0.6) & ~front)[..., None] | ~nearbg[..., None], im, np.where(((a > 0.05) & ~front)[..., None], np.clip(un, 0, 255), ext))
# along the top edge the cushion's own colour from just below the edge (no backdrop mixed into the edge pixels)
below_edge = cv2.remap(im, X, (topc + 3 * sc).astype(np.float32), cv2.INTER_LINEAR)
layer = np.where(topband[..., None], below_edge, layer)
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
bb = cv2.GaussianBlur(behind.astype(np.float32), (0, 0), 4 * sc)[..., None] * seatpoly[..., None]
layer = layer * (1 - bb) + synth * bb
# plate behind the seat: the studio backdrop in the gap behind it, the dark mechanism under it (inpainted)
hole = (cv2.dilate(seatpoly.astype(np.uint8), k(3)) > 0) & ~front & (X / sc > 440)
hole = hole.astype(np.uint8)
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
GAPV = float(os.environ.get('GAPV', 1.0))
tv = np.clip((Y / sc - 545) / 90, 0, 1)[..., None]
gapc = (bg * GAPV * (1 - 0.25 * tv)) * (1 - tb) + armfill * tb
# the opening is filled from what the photo shows just left of the cushion's back (frame bars, backdrop), smeared
# rightward row by row with a gentle fade into shadow, so it has no box edges
left = cv2.blur(cv2.remap(im, np.full((H, W), 440 * sc, np.float32), Y, cv2.INTER_LINEAR), (1, int(7 * sc) | 1))
fade = np.clip((X / sc - 447) / 60, 0, 1)[..., None]
gapc = left * (1 - 0.45 * fade)
syn = np.where(cont[..., None], gapc, syn)
# along the cushion's top, what lies just above it in the photo (armrest underside or backdrop) carried down a little,
# so the sliding top edge always meets the right thing
has = seatpoly.any(0); top = np.where(has, seatpoly.argmax(0), H).astype(np.float32)
topband = (hole > 0) & (Y < top[None, :] + 14 * sc) & (X > 560 * sc)
above = cv2.remap(im, X, np.maximum(top[None, :] - 6 * sc, 0).repeat(H, 0).astype(np.float32), cv2.INTER_LINEAR)
syn = np.where(topband[..., None], above, syn)
# along the cushion's bottom, the grey seat shell just below it carried up, so the sliding bottom edge meets it
bot = np.where(has, H - 1 - seatpoly[::-1].argmax(0), 0).astype(np.float32)
botband = (hole > 0) & (Y > bot[None, :] - 14 * sc) & (X > 560 * sc)
below2 = cv2.remap(im, X, np.minimum(bot[None, :] + 5 * sc, H - 1).repeat(H, 0).astype(np.float32), cv2.INTER_LINEAR)
syn = np.where(botband[..., None], below2, syn)
syn = np.where((cv2.erode((lum > 185).astype(np.uint8), k(1)) > 0)[..., None], im, syn)   # real backdrop stays as shot
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
    shade = 1 - 0.55 * np.clip(1 - dsh / (26 * sc), 0, 1) ** 1.5 * (dsh > 0) * contf
    f = L * ra + plate * shade[..., None] * (1 - ra)
    f = im * F + f * (1 - F)
    if i == 0: f = im
    cv2.imwrite(f'{out}/{i:04d}.png', np.clip(f + 0.5, 0, 255).astype(np.uint8))
cv2.imwrite(f'{out}/plate.png', np.clip(plate + 0.5, 0, 255).astype(np.uint8))
print('ok', N, 'frames')
