# Lumbar USP loop from the real side photo 04 on the studio backdrop (seat left, spine right): the lumbar pad's lower
# part swings forward (toward the seat, i.e. left) on its pivot bolt, holds briefly, and swings back, as in the
# reference video (about 6 degrees at the tip, lower part forward and slightly down). The mount around the bolt stays
# put and the turn grows with distance from it, so the top barely moves. Frame N equals frame 0, so it loops.
# Behind the pad the plate is rebuilt: the studio backdrop, and at the tip the spine bracket's dark slot inside its two
# upper-left edge. The seat and the backrest above the pad stay in front.
# Usage: python3 lumbar_swing_side.py still.png outdir   (still: start_frame_lumbar_studio.jpg at 2000 x 1125)
#        env: ANGLE deg (6), FPS (24), SECONDS (3)
import os, sys, numpy as np, cv2
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp import studio_bg
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
ANG = float(os.environ.get('ANGLE', 6)); FPS = int(os.environ.get('FPS', 24)); SEC = float(os.environ.get('SECONDS', 3))
PIV = (1529.0, 316.0)                                                   # pivot bolt
im = cv2.imread(src).astype(np.float32); H, W = im.shape[:2]; lum = im.mean(2); assert (W, H) == (2000, 1125)
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
poly = lambda pts: cv2.fillPoly(np.zeros((H, W), np.uint8), [np.array(pts, np.int32)], 1) > 0
k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
bg = (studio_bg(W, H)[..., ::-1] * 255).astype(np.float32)

# the pad: outer C edge, inner edge, and along the thin backdrop gap above the seat; everything darker than the backdrop
PAD = [(1258, 330), (1281, 212), (1330, 156), (1390, 134), (1440, 144), (1485, 176), (1505, 215), (1470, 240), (1425, 258),
       (1420, 330), (1424, 420), (1446, 490), (1518, 555), (1598, 605), (1680, 646), (1712, 656), (1704, 700), (1700, 734),
       (1650, 738), (1600, 725), (1550, 700), (1500, 670), (1450, 636), (1400, 600), (1350, 560), (1300, 505), (1262, 420)]
pad = (poly(PAD) & (lum < 185) & ~((X > 1640) & (Y > 640) & (lum < 20))).astype(np.uint8)   # not the slot
pad = cv2.morphologyEx(pad, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
# the tip: its rounded end overlaps the bracket slot, so the end is outlined by hand and joined to the thresholded
# limb; the slot's anti-aliased edge above the end and the rib frame's rim below it are cut away
_ey = np.arange(600, 668, 4)                                            # the spine's inner edge, measured above the tip
_ex = np.array([1736.7, 1736.2, 1735.6, 1734.7, 1733.9, 1733.1, 1732.0, 1730.9, 1729.7, 1728.4, 1726.9, 1725.5, 1723.7,
                1722.0, 1720.4, 1718.6, 1716.6])
xin = lambda y: np.polyval(np.polyfit(_ey, _ex, 2), y)                 # ... and its continuation behind the tip
TIP = [(1693, 698), (1700, 701), (1703, 706), (1706, 713), (1706, 719), (1703, 724), (1696, 728), (1685, 731),
       (1668, 731), (1660, 728), (1660, 700)]
pad[poly(TIP)] = 1
pad[(X > xin(Y) - 4) & (Y < 697) & (X > 1675)] = 0
pad[(X > 1688 + 0.737 * (Y - 681)) & (Y < 700) & (Y > 660)] = 0        # the end's upper edge (1686,681)-(1700,700)
pad[(Y > 732) & (X > 1655)] = 0; pad[(X + Y > 2392) & (X > 1600) & (X <= 1655)] = 0
tipbox = poly([(1630, 640), (1720, 640), (1720, 750), (1630, 750)])
sm = cv2.morphologyEx(cv2.morphologyEx(pad, cv2.MORPH_OPEN, k(3)), cv2.MORPH_CLOSE, k(3))
pad[tipbox] = sm[tipbox]
pad[(X > 1687 + 0.737 * (Y - 681)) & (Y < 700) & (Y > 660)] = 0        # again after smoothing: no knob
n, lab, st, _ = cv2.connectedComponentsWithStats(pad); pad = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
a = cv2.GaussianBlur(pad.astype(np.float32), (0, 0), 0.9)
# the moving layer: the pad's own colours carried a few pixels past its outline, so its soft edge never drags along
# what lay next to it (the slot's black at the tip, the frame's rim below it)
pf = pad.astype(np.float32)
ext = cv2.GaussianBlur(im * pf[..., None], (0, 0), 2.0) / np.maximum(cv2.GaussianBlur(pf, (0, 0), 2.0)[..., None], 1e-3)
layer = np.where((pad > 0)[..., None], im, ext)
# where the pad's edge meets the light backdrop, matte it by brightness (a soft, natural edge, no light fringe) and
# give them the pad's own colour
bgl = (studio_bg(W, H)[..., ::-1] * 255).astype(np.float32)
edge = (cv2.dilate(pad, k(2)) > 0) & (cv2.erode(pad, k(2)) == 0) & (lum > 50) & ~(tipbox & ((X > xin(Y) - 3) | (Y > 729) | ((X > 1687 + 0.737 * (Y - 681)) & (Y < 702))))
am = np.clip((bgl.mean(2) - lum) / (bgl.mean(2) - 55.0), 0, 1)
a = np.where(edge, am, a)
layer = np.where(edge[..., None], ext, layer)

# plate behind the pad: backdrop; at the tip, the bracket's dark slot right of its upper-left edge; around the bolt (thin slivers only) inpainted from the mount and arm
tipzone = (X > 1630) & (Y > 620) & (Y < 800)
# the spine's inner edge as shot: its shading profile across the edge, measured above the tip, laid along the edge's
# continuation, so the uncovered part reads as the same spine surface
_d = np.arange(-4, 24)
_prof = np.median(np.stack([np.stack([im[y, int(round(float(xin(y)))) + d] for d in _d]) for y in range(612, 664, 2)]), 0)
_slot = np.median(im[700:730, 1715:1735].reshape(-1, 3), 0)                # the slot's own black, right behind the tip
_prof[12:] = _slot + (_prof[12:] - _slot) * 0                           # deep in the slot it is that black
_prof[8:12] = _prof[8:12] * np.linspace(1, 0, 4)[:, None] + _slot * np.linspace(0, 1, 4)[:, None]
dd = np.clip(X - xin(Y), -4, 22.999) + 4; d0 = dd.astype(int); df = (dd - d0)[..., None]
edgefill = _prof[d0] * (1 - df) + _prof[np.clip(d0 + 1, 0, len(_d) - 1)] * df
syn = np.where(tipzone[..., None], np.where((X - xin(Y) < -3.5)[..., None], bg, edgefill), bg)
inp = cv2.inpaint(np.clip(im, 0, 255).astype(np.uint8), cv2.dilate(pad, k(2)) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
rr = np.hypot(X - PIV[0], Y - PIV[1])
syn = np.where((rr < 175)[..., None], inp, syn)
# under the backrest the pad's top end slides along the backrest's dark underside, not the backdrop: fill with that
# (only right of the backdrop notch: left of it, and along the headrest's lower edge above it, it is backdrop)
topz = poly([(1428, 100), (1540, 100), (1540, 235), (1428, 235)])
syn = np.where(topz[..., None], np.median(im[165:200, 1440:1500].reshape(-1, 3), 0), syn)
R = cv2.GaussianBlur(cv2.dilate(pad, k(5)).astype(np.float32), (0, 0), 1.0)[..., None]
# above the tip end the old pad edge (left in place by the cuts) is replaced too, up to the slot's edge
R = np.maximum(R, cv2.GaussianBlur((tipbox & (X < xin(Y) + 3) & (Y < 700)).astype(np.float32), (0, 0), 1.0)[..., None])
plate = im * (1 - R) + syn * R
# in front of the pad: the seat (black fabric below the gap) and the backrest piece above the pad's top end
seat = poly([(1100, 520), (1350, 565), (1400, 605), (1450, 641), (1500, 676), (1550, 706), (1600, 731), (1640, 760),
             (1640, 1125), (1100, 1125)]) & (lum < 30)
front = poly([(1380, 60), (1600, 60), (1600, 200), (1485, 176), (1440, 144), (1390, 134), (1380, 120)]) & (lum < 120) & (pad == 0)
F = cv2.GaussianBlur((seat | front).astype(np.float32), (0, 0), 0.7)[..., None]

dX, dY = X - PIV[0], Y - PIV[1]
R0, R1 = float(os.environ.get('HOLD_R', 0)), float(os.environ.get('FULL_R', 1))   # 0/1: the whole pad turns rigidly
wr = np.clip((rr - R0) / (R1 - R0), 0, 1); wr = wr * wr * (3 - 2 * wr)
N = int(round(SEC * FPS)); ease = lambda u: u * u * (3 - 2 * u)
def angle(i):                                                           # rest -> forward -> hold -> back -> rest
    u = i / N; t1, t2, t3 = 0.40, 0.52, 0.92
    if u < t1: return ANG * ease(u / t1)
    if u < t2: return ANG
    if u < t3: return ANG * (1 - ease((u - t2) / (t3 - t2)))
    return 0.0
for i in range(N):
    th = -np.radians(angle(i)) * wr                                    # clockwise on screen: lower part moves left
    c_, s_ = np.cos(th), np.sin(th)
    mx = (PIV[0] + c_ * dX - s_ * dY).astype(np.float32); my = (PIV[1] + s_ * dX + c_ * dY).astype(np.float32)
    Lr = cv2.remap(layer, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ra = cv2.remap(a, mx, my, cv2.INTER_LINEAR)[..., None]
    f = Lr * ra + plate * (1 - ra)
    f = im * F + f * (1 - F)
    if i == 0: f = im
    cv2.imwrite(f'{out}/{i:04d}.png', np.clip(f + 0.5, 0, 255).astype(np.uint8))
cv2.imwrite(f'{out}/plate.png', np.clip(plate + 0.5, 0, 255).astype(np.uint8))
print('ok', N, 'frames')
