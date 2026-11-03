# Lumbar-ONLY USP loop from ONE real studio photo (no AI): only the lumbar pad + its bracket slide forward (towards the
# chair front, level) and back; the backrest, spine and seat never move. Same lift / clean plate as
# lumbar_slide_photo_end.py, but the moving region stops at the lumbar's top edge (under the backrest's lower edge) and
# dark pixels outside the lumbar are never replaced by wall. Seamless loop (first frame = last frame).
# Usage: python lumbar_only_photo_loop.py photo.jpg outdir
#   env: DIST 30 px @2000 wide, KEYS t:p (0:0,0.6:0,1.6:1,2.6:1,3.6:0,4:0), FPS 30, ZOOM x0,y0,x1 (2000-px) for the close-up
import cv2, numpy as np, sys, os, subprocess
src, out = sys.argv[1:3]; os.makedirs(out, exist_ok=True)
I = cv2.imread(src).astype(np.float32); H, W = I.shape[:2]; k = W / 2000
P = lambda pts: (np.array(pts, np.float32) * k).astype(np.int32)
DIST = float(os.environ.get('DIST', 20)) * k
ARM_A, ARM_B = np.array([1500, 318.]) * k, np.array([1748, 465.]) * k    # bracket bolt -> spine bolt
ARM_J = np.array([1650, 400.]) * k                                       # arm crosses the rear frame bar: static from here
u = (ARM_B - ARM_A) / np.linalg.norm(ARM_B - ARM_A); S = np.array([-DIST, -DIST * float(os.environ.get('SLOPE', 0.4))])   # towards the chair front, along the pad's own top edge (stays tucked under the backrest)
# regions (2000-px coords): lumbar + bracket + backrest bottom; the arm separately
MOVE = P([(1240, 140), (1380, 138), (1406, 141), (1413, 147), (1435, 148), (1455, 148), (1475, 147), (1495, 151), (1515, 158), (1535, 168), (1555, 179), (1565, 185),
          (1580, 190), (1580, 262), (1566, 365), (1490, 375), (1540, 470), (1600, 540), (1660, 610),
          (1690, 672), (1694, 694), (1705, 704), (1708, 714), (1704, 724), (1695, 728), (1650, 735),   # tip outline (inset)
          (1600, 718), (1550, 700), (1500, 670), (1450, 636), (1400, 600), (1356, 570), (1240, 540)])          # lower edge: mid-gap lumbar|seat (measured)
ARM = P([(1530, 300), (1600, 300), (1700, 400), (1765, 450), (1760, 505), (1700, 505), (1570, 410), (1530, 360)])
STATIC = P([(1628, 0), (2000, 0), (2000, 800), (1740, 800), (1713, 700), (1703, 693), (1722, 650), (1735, 600), (1700, 520), (1688, 420),
            (1680, 370), (1660, 280), (1640, 150)])          # rear frame bar + spine: never moved, never filled
x0, y0, x1, y1 = int(1180 * k), 0, int(1830 * k), int(800 * k)              # work crop
C = I[y0:y1, x0:x1].copy(); h, w = C.shape[:2]
def poly(p): m = np.zeros((h, w), np.uint8); cv2.fillPoly(m, [p - [x0, y0]], 1); return m
L = cv2.cvtColor(C.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
dark = (L < 140).astype(np.uint8)
mv, ar = poly(MOVE) & dark, poly(ARM) & dark & (1 - poly(MOVE))
stat = poly(STATIC)
mv, ar = mv & (1 - stat), ar & (1 - stat)
movers = mv | ar
grow = cv2.dilate(movers, np.ones((int(9 * k) | 1,) * 2, np.uint8)) & (1 - stat)
# clean plate: everything uncovered is studio wall - fill it from wall pixels only (normalized convolution), so no
# dark spine / seat pixels bleed in; light grain added back
wallm = ((L > 150) & (grow == 0)).astype(np.float32); sg = 25 * k
num = cv2.GaussianBlur(C * wallm[..., None], (0, 0), sg); den = cv2.GaussianBlur(wallm, (0, 0), sg)[..., None]
wall = num / np.maximum(den, 1e-4) + np.random.default_rng(2).normal(0, 1.5, C.shape)
# never wipe the backrest (above the lumbar) or the seat (below it); loose dark bits around the bracket do go
yb = ((np.mgrid[0:h, 0:w][0] + y0) / k)
xb = ((np.mgrid[0:h, 0:w][1] + x0) / k)
keep = dark & (1 - movers) & ((yb < 195) | ((yb > 520) & (xb < 1640))).astype(np.uint8)   # (not at the lumbar tip)
hole = grow * (1 - keep)
clean = np.where(hole[..., None] > 0, wall, C)
# except under the lumbar tip: it sat in front of the spine's lower piece, whose upper edge runs (1666,733)->(1700,697)
# (2000-px); the uncovered part on the spine side of that line is spine black, not wall
yy0, xx0 = np.mgrid[0:h, 0:w].astype(np.float32); Xg, Yg = (xx0 + x0) / k, (yy0 + y0) / k
sp = ((Xg - 1666) * 36 + (Yg - 733) * 34 > 0) & (Xg > 1630) & (Xg < 1730) & (Yg > 680) & (Yg < 770)
# fill it with real spine texture cloned from further along that same edge (down-left, where the piece is visible)
v = np.array([float(q) for q in os.environ.get('SPINE_CLONE', '14,0').split(',')]) * k   # spine texture just right of the hole (same piece)
srcx = np.clip(xx0 + v[0], 0, w - 1).astype(np.float32); srcy = np.clip(yy0 + v[1], 0, h - 1).astype(np.float32)
clone = cv2.remap(C.astype(np.float32), srcx, srcy, cv2.INTER_LINEAR)
hole_sp = ((grow > 0) & sp)
dk = np.median(C[(stat > 0) & (L < 60) & (Xg > 1705) & (Xg < 1740) & (Yg > 700) & (Yg < 760)], axis=0)
clone = np.where((cv2.cvtColor(np.clip(clone, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY) > 90)[..., None], dk, clone)
clean = np.where(hole_sp[..., None], clone, clean)          # hard edge: no wall-colour rim against the spine
# the lumbar's top runs on under the backrest's lower edge: where its top edge pulls away from that edge while it slides,
# show more lumbar top face (cloned from just inside it), not wall
TOPL = np.array([(1413, 147), (1455, 148), (1475, 147), (1495, 151), (1515, 158), (1535, 168), (1555, 179), (1565, 185)], np.float32)   # under the backrest's dark seam (measured)
line = np.interp(Xg, TOPL[:, 0], TOPL[:, 1])
band = ((Xg >= 1418) & (Yg >= line - 5) & (Yg <= line + 14) & (grow > 0)).astype(np.float32)
band = band * np.clip((1566 - DIST / k - Xg) / 2.0 + 0.5, 0, 1)        # ends at the slid lumbar's right edge (1572 at rest), feathered
srcy2 = np.clip(yy0 + 18 * k, 0, h - 1).astype(np.float32)
face = cv2.remap(C.astype(np.float32), xx0.astype(np.float32), srcy2, cv2.INTER_LINEAR)
if os.environ.get('BAND', '1') == '1': clean = face * band[..., None] + clean * (1 - band[..., None])
# soft alpha + un-mixed colour
bgL = cv2.cvtColor(clean.astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
fgL = np.percentile(L[movers > 0], 20)
a = np.clip((bgL - L) / np.maximum(bgL - fgL, 1), 0, 1) * grow
# only the traced outlines (plus a hair for the anti-aliased edge against the wall) - never neighbouring spine pixels
a = a * (cv2.dilate(poly(MOVE) | poly(ARM), np.ones((int(3 * k) | 1,) * 2, np.uint8)) & (1 - stat))
a = np.where(cv2.erode(movers, np.ones((3, 3), np.uint8)) > 0, 1.0, a)
# lumbar tip: it sat on the black spine, so its edge pixels carry spine black - use a soft edge from the inset outline
tipb = ((Xg > 1655) & (Xg < 1722) & (Yg > 680) & (Yg < 745)).astype(np.float32)
soft = cv2.GaussianBlur(cv2.erode(poly(MOVE), np.ones((int(2 * k) | 1,) * 2, np.uint8)).astype(np.float32), (0, 0), 1.5 * k)
a = a * (1 - tipb) + np.minimum(a, soft) * tipb
F = (C - (1 - a[..., None]) * clean) / np.maximum(a[..., None], 0.05)
F = np.where(a[..., None] < 0.05, C, np.clip(F, 0, 255))      # no blow-up where there is no foreground (cubic warp rings)
F = np.where(a[..., None] < 0.9, np.minimum(F, np.median(C[movers > 0], axis=0) * 1.6), F)   # edge pixels: never lighter than the pad (no white fringe)
# displacement weight at OUTPUT positions: 1 on the lumbar, fades to 0 at the top (backrest hinge), linear along the arm
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); X, Y = xx + x0, yy + y0
wgt = np.ones_like(Y)                                              # the whole lumbar moves rigidly
wgt = wgt * (cv2.dilate(poly(MOVE), np.ones((int(25 * k) | 1,) * 2, np.uint8)) > 0)    # nothing else moves
armz = cv2.dilate(poly(ARM) & (1 - poly(MOVE)), np.ones((int(15 * k) | 1,) * 2, np.uint8)) > 0
ARM_E = np.array([1585., 335.]) * k                                     # where the arm leaves the lumbar bracket
t = np.clip(((X - ARM_J[0]) * -u[0] + (Y - ARM_J[1]) * -u[1]) / (np.dot(ARM_J - ARM_E, u) + 1e-6), 0, 1)   # 1 at the bracket, 0 at the spine
wgt = np.where(armz, t, wgt)
wgt = cv2.GaussianBlur(wgt.astype(np.float32), (0, 0), 4 * k)
FPS = int(os.environ.get('FPS', 30))
KEYS = [tuple(float(v) for v in q.split(':')) for q in os.environ.get('KEYS', '0:0,0.6:0,1.6:1,2.6:1,3.6:0,4:0').split(',')]
def at(t):
    for (t0, v0), (t1, v1) in zip(KEYS, KEYS[1:]):
        if t <= t1: q = min(max((t - t0) / max(t1 - t0, 1e-6), 0), 1); return v0 + (v1 - v0) * q * q * (3 - 2 * q)
    return KEYS[-1][1]
Fa = F.astype(np.float32); aa = a.astype(np.float32)
# the small lift only where the pad's top edge slopes down under the backrest (right part); the front (left) part slides level
vf = np.clip((X / k - 1440) / 60, 0, 1); vf = (vf * vf * (3 - 2 * vf)).astype(np.float32)
# backrest lower edge = a static layer drawn OVER the sliding pad (the pad tucks under it, its outline never changes)
bk = (dark & (1 - movers) & (yb < 200) & (xb > 1370) & (xb < 1600)).astype(np.uint8)
bkd = cv2.dilate(bk, np.ones((int(3 * k) | 1,) * 2, np.uint8)) & (1 - movers)
aB = np.clip((bgL - L) / np.maximum(bgL - fgL, 1), 0, 1) * bkd; aB = np.where(bk > 0, 1.0, aB).astype(np.float32)[..., None]
bcol = np.median(C[(bk > 0) & (L < 60)], axis=0)                       # its edge pixels get the backrest's own colour (no wall fringe)
FB = np.where(bk[..., None] > 0, C, bcol).astype(np.float32)
def frame(p):
    if p < 1e-4: return np.clip(I, 0, 255).astype(np.uint8)
    mx, my = (xx - S[0] * p * wgt).astype(np.float32), (yy - S[1] * p * wgt * vf).astype(np.float32)
    a2 = np.clip(cv2.remap(aa, mx, my, cv2.INTER_CUBIC, borderValue=0), 0, 1)[..., None]
    F2 = cv2.remap(Fa, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    m = np.clip(np.hypot(*S) * p * wgt / 1.5, 0, 1)[..., None]     # where nothing moves (< 1.5 px) keep the photo exactly
    comp = F2 * a2 + clean * (1 - a2); comp = FB * aB + comp * (1 - aB)
    E = I.copy(); E[y0:y1, x0:x1] = comp * m + C * (1 - m)
    return np.clip(E, 0, 255).astype(np.uint8)
zx0, zy0, zx1 = [float(v) for v in os.environ.get('ZOOM', '880,110,2000').split(',')]
Z = [int(zx0 * k), int(zy0 * k), int(zx1 * k), int(zy0 * k + (zx1 - zx0) * k * 9 / 16)]
def enc(path, size):
    return subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', '%dx%d' % size, '-r', str(FPS), '-i', '-',
        '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', path], stdin=subprocess.PIPE)
oh = W * 9 // 16; oy = (H - oh) // 2
outs = [('full_1080p', (1920, 1080), lambda f: f[oy:oy + oh]), ('full_4k', (3840, 2160), lambda f: f[oy:oy + oh]),
        ('closeup_1080p', (1920, 1080), lambda f: f[Z[1]:Z[3], Z[0]:Z[2]])]
pr = [] if os.environ.get('STILL') else [(nm, sz, cr, enc('%s/lumbar_only_photo_%s.mp4' % (out, nm), sz)) for nm, sz, cr in outs]
n = int(round(KEYS[-1][0] * FPS))
if os.environ.get('STILL'):
    f = frame(1.0); cv2.imwrite(out + '/end_closeup.png', f[Z[1]:Z[3], Z[0]:Z[2]]); sys.exit()
for i in range(n):
    p = at(i / FPS); f = frame(p)
    for nm, sz, cr, pp in pr: pp.stdin.write(cv2.resize(cr(f), sz, interpolation=cv2.INTER_AREA).tobytes())
    if i == int(1.8 * FPS):
        cv2.imwrite(out + '/end_full.jpg', f, [cv2.IMWRITE_JPEG_QUALITY, 95]); cv2.imwrite(out + '/end_closeup.png', f[Z[1]:Z[3], Z[0]:Z[2]])
        cv2.imwrite(out + '/start_closeup.png', np.clip(I, 0, 255).astype(np.uint8)[Z[1]:Z[3], Z[0]:Z[2]])
for nm, sz, cr, pp in pr: pp.stdin.close(); pp.wait()
print('photo', W, H, 'shift px', S.round(1), 'frames', n)
