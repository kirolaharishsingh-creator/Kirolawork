# Lumbar USP loop from one photo-real still (the cleaned Nano Banana image, spine left / seat right): the lumbar pad is
# cut out and rocked about its pivot bolt -- lower edge swinging forward, a short hold, back to rest -- as in the
# reference video (about 8 degrees at the tip). Frame N equals frame 0, so it loops.
# Behind the pad the plate is rebuilt, not inpainted: the studio backdrop (smooth field from the bright pixels), the
# seat below its own top-edge curve (fitted where the edge is visible) and the spine/bracket left of its outline, each
# with the shading profile measured across its visible edge. The backrest above the pad top stays in front.
# Usage: python3 lumbar_swing_photo.py still.jpg outdir   (env: ANGLE deg (8), FPS (24), SECONDS (2.5))
import os, sys, numpy as np, cv2
src, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
ANG = float(os.environ.get('ANGLE', 8)); FPS = int(os.environ.get('FPS', 24)); SEC = float(os.environ.get('SECONDS', 2.5))
PIV = (563.0, 560.0)                                                   # pivot bolt (2000 x 1125 image)
im = cv2.imread(src).astype(np.float32); H, W = im.shape[:2]; lum = im.mean(2)
Y, X = np.mgrid[0:H, 0:W].astype(np.float32)
poly = lambda pts: cv2.fillPoly(np.zeros((H, W), np.uint8), [np.array(pts, np.int32)], 1) > 0
k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))

# seat: top-edge curve from the visible edge (first dark pixel with backdrop just above it)
sx, sy = [], []
for x in range(560, 960, 10):
    col = lum[700:1125, x]; ys = [y for y in range(5, 425) if col[y] < 40 and col[y - 4] > 150]
    if ys: sx.append(x); sy.append(700 + ys[0])
seat_y = np.polyval(np.polyfit(sx, sy, 3), X)
seat = (Y >= seat_y - 0.5) & (X >= 540) & (X <= 1000)
# spine / bracket: left of the C-curve outline that runs into the bracket edge
L = lambda x: 989 + 0.78 * (x - 407)
spine = poly([(344, 860), (356, 900), (368, 925), (375, 940), (383, 953), (391, 966), (398, 977), (404, 985), (411, L(411)),
              (426, L(426)), (480, L(480)), (480, 1125), (250, 1125), (250, 860)])
# the pad: inside its rough outline, darker than the backdrop, not seat, not spine
PADPOLY = [(498, 366), (640, 360), (780, 368), (880, 400), (912, 480), (912, 620), (860, 760), (760, 840), (660, 905), (560, 965),
           (470, 1022), (400, 1022), (396, 968), (486, 880), (572, 800), (610, 700), (608, 620), (590, 598), (545, 600), (522, 575),
           (515, 520), (498, 470)]
pad = (poly(PADPOLY) & (lum < 120) & ~seat & (~spine | ((lum > 24) & (X >= 414)))).astype(np.uint8)
# at the tip a thin line of backdrop separates the pad (right of it) from the spine's edge (left of it): split there
tb = (X < 480) & (Y > 975) & (Y < 1040)
pad[tb & ((X - Y < -578) | (Y > 1009 + 0.27 * (X - 430)))] = 0
pad[tb & (X - Y >= -578) & (Y <= 1009 + 0.27 * (X - 430)) & (lum < 120) & poly(PADPOLY)] = 1
pad = cv2.morphologyEx(pad, cv2.MORPH_CLOSE, k(2))
pad[(lum > 150) & (Y > 780)] = 0                                         # no backdrop seams between pad and seat
n, lab, st, _ = cv2.connectedComponentsWithStats(pad); pad = (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)
cnts, hier = cv2.findContours(pad, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)
for i, c in enumerate(cnts):                                            # fill small holes, keep the C's opening and mesh dots
    if hier[0][i][3] >= 0 and cv2.contourArea(c) < 40: cv2.drawContours(pad, [c], -1, 1, -1)
a = cv2.GaussianBlur(pad.astype(np.float32), (0, 0), 0.7)

# plate behind the pad
bright = ((lum > 150) & ~cv2.dilate(poly(PADPOLY).astype(np.uint8), k(6)).astype(bool)).astype(np.float32); r = 30
def nconv(r):
    return cv2.GaussianBlur(im * bright[..., None], (0, 0), r), cv2.GaussianBlur(bright, (0, 0), r)[..., None]
n1, d1 = nconv(r); n2, d2 = nconv(160)                                  # wide pass reaches the middle of the pad
w = np.clip(d1 / 0.3, 0, 1); bg = w * n1 / np.maximum(d1, 1e-4) + (1 - w) * n2 / np.maximum(d2, 1e-4)
def edge_profile(depth_fn, rows):                                       # median colour vs depth below a visible edge
    return np.array([np.median([im[y, x] for (x, y) in depth_fn(d)], 0) for d in rows])
seat_prof = np.array([np.median([im[int(np.polyval(np.polyfit(sx, sy, 3), x)) + d, x] for x in range(840, 950, 3)], 0) for d in range(25)])
dseat = np.clip(Y - seat_y, 0, 24); i0 = dseat.astype(int); fr = (dseat - i0)[..., None]
seat_fill = seat_prof[i0] * (1 - fr) + seat_prof[np.clip(i0 + 1, 0, 24)] * fr
spine_dist = cv2.distanceTransform(spine.astype(np.uint8), cv2.DIST_L2, 5)
spine_prof = np.array([np.median([im[y, 320 + int(np.argmax(lum[y, 320:420] > 120)) - 1 - d] for y in range(880, 931)], 0) for d in range(16)])
ds = np.clip(spine_dist - 0.5, 0, 15); j0 = ds.astype(int); fs = (ds - j0)[..., None]
spine_fill = spine_prof[j0] * (1 - fs) + spine_prof[np.clip(j0 + 1, 0, 15)] * fs
syn = np.where(seat[..., None], seat_fill, np.where(spine[..., None], spine_fill, bg))
# around the pivot (the arm and cap sit behind) and under the backrest the pad barely moves
# (only thin slivers open up there, so they are inpainted from the arm, the backrest and the backdrop around them)
keep = (np.hypot(X - PIV[0], Y - PIV[1]) < 95) | poly([(470, 330), (900, 330), (900, 405), (470, 405)])
inp = cv2.inpaint(np.clip(im, 0, 255).astype(np.uint8), cv2.dilate(pad, k(2)) * 255, 5, cv2.INPAINT_TELEA).astype(np.float32)
syn = np.where(keep[..., None], inp, syn)
# at the pad tip the plate is the photo except under the tip itself: the bracket and its rim highlight stay as shot,
# and under the tip the spine/bracket is its dark body inside its outline, backdrop outside
tipbox = poly([(390, 975), (480, 975), (480, 1040), (390, 1040)])
tipfill = np.where((spine & (X - Y < -581))[..., None], spine_fill, bg)
syn = np.where(tipbox[..., None], np.where((cv2.dilate(pad, k(2)) > 0)[..., None], tipfill, im), syn)
R = cv2.GaussianBlur(cv2.dilate(pad, k(4)).astype(np.float32), (0, 0), 1.2)[..., None]
# the spine's edge just left of the tip (hidden by it at rest) is redrawn as its straight 45-degree edge
rim = poly([(400, 980), (446, 980), (446, 1024), (400, 1024)]) & (X - Y < -578)
de = (-(X - Y) - 581.0) / np.sqrt(2)                                  # depth inside the spine's 45-degree edge here
dq = np.clip(de, 0, 15); q0 = dq.astype(int); qf = (dq - q0)[..., None]
edgefill = spine_prof[q0] * (1 - qf) + spine_prof[np.clip(q0 + 1, 0, 15)] * qf
ca = np.clip(de + 0.5, 0, 1)[..., None]
syn = np.where(rim[..., None], edgefill * ca + bg * (1 - ca), syn)
R = np.where(tipbox[..., None], cv2.GaussianBlur(cv2.dilate(np.maximum(pad, rim.astype(np.uint8)), k(2)).astype(np.float32), (0, 0), 0.6)[..., None], R)
plate = im * (1 - R) + syn * R
# the backrest above the pad top stays in front
front = poly([(470, 250), (700, 250), (700, 372), (640, 362), (498, 366), (470, 380)]) & (lum < 120) & (pad == 0)
front = cv2.GaussianBlur(front.astype(np.float32), (0, 0), 0.8)[..., None]

dX, dY = X - PIV[0], Y - PIV[1]; rr = np.hypot(dX, dY)
R0, R1 = float(os.environ.get('HOLD_R', 110)), float(os.environ.get('FULL_R', 300))
wr = np.clip((rr - R0) / (R1 - R0), 0, 1); wr = wr * wr * (3 - 2 * wr)
N = int(round(SEC * FPS)); ease = lambda u: u * u * (3 - 2 * u)
def angle(i):                                                           # rest -> forward -> hold -> back -> rest
    u = i / N; t1, t2, t3 = 0.40, 0.52, 0.92
    if u < t1: return ANG * ease(u / t1)
    if u < t2: return ANG
    if u < t3: return ANG * (1 - ease((u - t2) / (t3 - t2)))
    return 0.0
for i in range(N):
    # the mount around the bolt stays put and the turn grows with distance from it (top barely moves, the lower
    # part swings most, as in the reference); a rotation keeps the radius, so the inverse map is direct
    th = np.radians(angle(i)) * wr
    c_, s_ = np.cos(th), np.sin(th)
    mx = (PIV[0] + c_ * dX - s_ * dY).astype(np.float32); my = (PIV[1] + s_ * dX + c_ * dY).astype(np.float32)
    Lr = cv2.remap(im, mx, my, cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    ra = cv2.remap(a, mx, my, cv2.INTER_LINEAR)[..., None]
    f = Lr * ra + plate * (1 - ra)
    f = im * front + f * (1 - front)
    if i == 0: f = im                                                  # rest frame is the still itself
    cv2.imwrite(f'{out}/{i:04d}.png', np.clip(f + 0.5, 0, 255).astype(np.uint8))
cv2.imwrite(f'{out}/plate.png', np.clip(plate + 0.5, 0, 255).astype(np.uint8))
print('ok', N, 'frames')
