# Exploded-view animation built only from real photo pixels (photo 31, three-quarter view).
# The cutout is split into parts with hand-drawn polygons (seeds) + nearest-seed fill; each part slides out
# along its own direction, holds, and slides back. First and last frames are the untouched real chair.
# Usage: python3 tools/exploded_view.py out.mp4 [debug_parts.png] [--3d]
# --3d: depth parallax, small perspective turn per part, floor shadows under floating parts, motion blur
# --spread: wider layout on the right side; --still=out.png writes only the exploded hold frame
# --glow: cool-white glow + light streaks around each part as it detaches and as it locks back in
import sys, subprocess, numpy as np, cv2
sys.path.insert(0, 'tools')
import build_keyframe as bk
from PIL import Image

SRC = 'real_photo_keyframes/photo31_cutout_white.jpg'
W, H, FPS = 1920, 1080, 24
SCALE = bk.K * 0.62 / 2            # same chair size/placement as kf_explode_threequarter (4K -> 1080p)

def d2o(pts):                      # polygon points read off the 2x grid view -> cutout pixels
    return np.array([(191 + x / 1.529, 360 + y / 1.529) for x, y in pts], np.int32)

# Back-to-front draw order. offset = (dx, dy) in cutout pixels at full explosion; start = stagger (0..1)
PARTS = [
 ('frame',        [(990,222),(1075,230),(1125,275),(1132,335),(1065,560),(1015,760),(1005,870),(1010,1000),(1025,1080),(1022,1200),(990,1290),(955,1290),(950,1000),(962,870),(985,700),(1010,560),(1040,330),(1000,280)], (150, -40), 0.10),
 ('backrest',     [(595,222),(990,222),(1040,330),(1010,560),(980,745),(560,710),(540,690),(560,500)], (40, -150), 0.05),
 ('headrest',     [(640,140),(660,60),(760,25),(900,20),(1000,30),(1030,60),(1030,215),(640,215)], (10, -260), 0.00),
 ('lumbar',       [(495,790),(520,755),(600,742),(940,780),(950,880),(952,1082),(810,1062),(705,1034),(700,960),(690,920),(560,940),(495,930),(490,860)], (190, 20), 0.15),
 ('far_armrest',  [(680,940),(700,905),(800,895),(990,890),(995,920),(960,945),(720,965)], (210, -20), 0.30),
 ('base',         [(30,1700),(270,1560),(500,1540),(620,1540),(840,1560),(1100,1720),(1100,1780),(620,1650),(600,1880),(520,1880),(515,1650),(60,1790)], (0, 240), 0.40),
 ('gas_lift',     [(520,1395),(600,1385),(605,1520),(612,1560),(508,1560),(515,1520)], (0, 150), 0.35),
 ('mechanism',    [(245,1340),(330,1352),(400,1300),(800,1355),(1000,1300),(1020,1300),(1000,1362),(800,1368),(615,1378),(600,1390),(510,1395),(495,1480),(410,1475),(400,1390),(330,1375),(250,1356)], (0, 70), 0.30),
 ('seat',         [(130,1100),(170,1060),(400,1030),(560,1025),(700,1042),(800,1075),(890,1102),(950,1100),(950,1290),(800,1352),(500,1347),(300,1308),(160,1258),(130,1200)], (-60, -50), 0.20),
 ('near_armrest', [(85,830),(110,800),(250,782),(390,795),(392,825),(345,845),(380,1000),(395,1040),(330,1045),(290,935),(230,895),(100,865)], (-190, -40), 0.25),
]
# --spread: wider layout so the spine frame, lumbar and far armrest no longer bunch up on the right
SPREAD = {'frame': (330, -70), 'lumbar': (170, 40), 'far_armrest': (420, 190)}
if '--spread' in sys.argv:
    PARTS = [(n, poly, SPREAD.get(n, off), st) for n, poly, off, st in PARTS]
WHEELS = [((300,1640),45), ((830,1640),45), ((70,1810),58), ((1050,1810),58), ((560,1930),58)]   # display coords, radius

plain4k, shadow4k, chair4k, (px4, py4) = bk.build(SRC, None, use_rim=False, scale=0.62)
THREE_D = '--3d' in sys.argv
GLOW = '--glow' in sys.argv
GLOW_COL = np.array([255, 248, 238], np.float32)    # BGR: cool white, never yellow
DEPTH = {'frame': -0.6, 'backrest': -0.4, 'headrest': -0.3, 'lumbar': -0.5, 'far_armrest': -0.9, 'base': 0.1,
         'gas_lift': 0.0, 'mechanism': 0.15, 'seat': 0.35, 'near_armrest': 0.9,
         'wheel0': -0.5, 'wheel1': -0.6, 'wheel2': 0.4, 'wheel3': 0.3, 'wheel4': 0.9}
TURN = {'frame': -1, 'backrest': 1, 'headrest': -1, 'lumbar': -1, 'far_armrest': -1, 'base': 1, 'gas_lift': 0,
        'mechanism': 1, 'seat': 1, 'near_armrest': 1}
src, alpha = bk.cutout(SRC)
alpha = alpha.astype(np.float32); h, w = alpha.shape
chair = alpha > 0.02
names = [p[0] for p in PARTS] + [f'wheel{i}' for i in range(len(WHEELS))]
seed = np.zeros((h, w), np.int32)                       # 0 = none, i+1 = part i
for i, (n, poly, _, _) in enumerate(PARTS):
    cv2.fillPoly(seed, [d2o(poly)], i + 1)
for j, ((x, y), r) in enumerate(WHEELS):
    c = d2o([(x, y)])[0]; cv2.circle(seed, tuple(int(v) for v in c), int(r / 1.529), len(PARTS) + j + 1, -1)
seed[~chair] = 0
# every chair pixel without a seed joins the nearest seeded part
lab = np.zeros((h, w), np.int32); best = np.full((h, w), 1e9, np.float32)
for k in range(1, len(names) + 1):
    if not (seed == k).any(): continue
    d = cv2.distanceTransform((seed != k).astype(np.uint8), cv2.DIST_L2, 5)
    m = d < best; best[m] = d[m]; lab[m] = k
lab[~chair] = 0
# the dark band above the white gap between seat and lumbar is the lumbar frame's lower edge, not the seat
yy0, xx0 = np.mgrid[0:h, 0:w]
seat_i, lumbar_i = names.index('seat') + 1, names.index('lumbar') + 1
lab[(lab == seat_i) & (xx0 < 722) & (yy0 < 1010 + (xx0 - 560) * 0.354)] = lumbar_i

wheel_offsets = []
for (x, y), r in WHEELS:                                  # wheels drop with the base and spread outward
    c = d2o([(x, y)])[0]; wheel_offsets.append(((c[0] - 558) * 0.35, 330))
specs = [(n, off, st) for n, _, off, st in PARTS] + [(f'wheel{i}', wheel_offsets[i], 0.45) for i in range(len(WHEELS))]
order = list(range(len(PARTS)))                           # wheels: back pair after the base, front three on top
order = order[:6] + [len(PARTS), len(PARTS) + 1] + order[6:] + [len(PARTS) + 2, len(PARTS) + 3, len(PARTS) + 4]

layers = []
for idx in range(len(names)):
    m = (lab == idx + 1)
    if not m.any(): layers.append(None); continue
    a = alpha * m
    # pixels hidden behind parts drawn later: fill them from this part's own colours so no hole opens up
    pos = order.index(idx); later = [order[q] + 1 for q in range(pos + 1, len(order))]
    near = cv2.dilate(m.astype(np.uint8), np.ones((25, 25), np.uint8)).astype(bool)
    ys, xs = np.nonzero(m); hull = np.zeros((h, w), np.uint8)
    cv2.fillPoly(hull, [cv2.convexHull(np.stack([xs, ys], 1).astype(np.int32))], 1)
    hole = near & hull.astype(bool) & np.isin(lab, later)
    col = src.astype(np.uint8).copy()
    if hole.any():
        col = cv2.inpaint(col, (hole * 255).astype(np.uint8), 7, cv2.INPAINT_TELEA)
        a = np.maximum(a, hole * alpha)
    ys, xs = np.nonzero(a > 0.02)
    layers.append((col.astype(np.float32), a, (xs.min(), ys.min(), xs.max(), ys.max())))

# background: the builder's own studio gradient; its contact shadows become a layer that follows base + wheels
to1080 = lambda im: cv2.resize(cv2.cvtColor(np.array(im), cv2.COLOR_RGB2BGR), (W, H), interpolation=cv2.INTER_AREA).astype(np.float32)
BG = to1080(plain4k); SHADOW = np.clip(to1080(shadow4k) / np.maximum(BG, 1), 0, 1)   # multiplicative shadow
cw4, ch4 = chair4k.size
sx, sy = cw4 / w / 2, ch4 / h / 2                            # exact pasted size of the chair, at 1080p
ox, oy = px4 / 2, py4 / 2
_k = shadow4k.convert('RGBA'); _k.alpha_composite(chair4k, (px4, py4)); REAL = to1080(_k.convert("RGB"))   # the exact real keyframe

def ease(t): t = np.clip(t, 0, 1); return t * t * (3 - 2 * t)
def explode(t):                                           # 0..1 overall explosion amount per part, by time
    if t < 1: return lambda st: 0.0
    if t < 4: u = (t - 1) / 3; return lambda st: ease((u - st * 0.5) / 0.5)
    if t < 6: return lambda st: 1.0
    if t < 9: u = (t - 6) / 3; return lambda st: 1 - ease((u - (0.5 - st * 0.5)) / 0.5)
    return lambda st: 0.0

def place(idx, k, t):
    # 3x3 matrix taking cutout pixels to screen for part idx at explosion amount k
    n, (dx, dy), st = specs[idx]
    x0, y0, x1, y1 = layers[idx][2]; cxp, cyp = (x0 + x1) / 2, (y0 + y1) / 2
    A = np.array([[sx, 0, ox + dx * k * sx], [0, sy, oy + dy * k * sy], [0, 0, 1]], np.float64)
    if not THREE_D or k == 0: return A
    z = DEPTH.get(n, 0)
    s = 1 + 0.10 * z * k                                   # nearer parts grow, farther ones shrink (perspective)
    orbit = np.sin(np.pi * np.clip((t - 1) / 8, 0, 1))     # slow camera drift, zero at both ends
    par = -z * 70 * k * orbit                              # parallax in screen px
    ang = np.radians(9) * k * TURN.get(n, 0)               # small turn about the vertical axis
    # perspective turn about the part's own centre: squeeze width, grow one side's height
    hw, hh = (x1 - x0) / 2, (y1 - y0) / 2
    srcq = np.float32([[x0, y0], [x1, y0], [x1, y1], [x0, y1]])
    c, g = np.cos(ang), 0.12 * np.sin(ang)
    dst = np.float32([[cxp - hw * c, cyp - hh * (1 + g)], [cxp + hw * c, cyp - hh * (1 - g)],
                      [cxp + hw * c, cyp + hh * (1 - g)], [cxp - hw * c, cyp + hh * (1 + g)]])
    P = cv2.getPerspectiveTransform(srcq, dst).astype(np.float64)
    S = np.array([[s, 0, cxp * (1 - s)], [0, s, cyp * (1 - s)], [0, 0, 1]])
    T = np.array([[1, 0, par], [0, 1, 0], [0, 0, 1]])
    return T @ A @ S @ P

FLOOR0 = oy + 1640 * sy                                    # floor line under the wheels at rest
def render(t):
    f = explode(t)
    if all(f(st) == 0 for _, _, st in specs): return REAL.copy()
    sh = cv2.warpAffine(SHADOW, np.float32([[1, 0, 0], [0, 1, f(0.45) * 330 * sy]]), (W, H), borderValue=(1, 1, 1))
    out = BG * sh
    if THREE_D:                                            # soft floor shadow under each floating part
        FLOOR = FLOOR0 + f(0.45) * 330 * sy                # the floor sits under the lowered wheels
        fs = np.zeros((H, W), np.float32)
        for idx in order:
            L = layers[idx]
            if L is None: continue
            n, _, st = specs[idx]; k = f(st)
            if k <= 0 or n.startswith('wheel') or n == 'base': continue
            M = place(idx, k, t); x0, y0, x1, y1 = L[2]
            pts = cv2.perspectiveTransform(np.float32([[[x0, y1], [x1, y1]]]), M)[0]
            xa, xb = sorted([pts[0][0], pts[1][0]]); height = max(FLOOR - pts[:, 1].max(), 0)
            cx_, rw = (xa + xb) / 2, (xb - xa) / 2 * 0.9 + 10
            op = 0.20 * k * np.exp(-height / 500)
            cv2.ellipse(fs, (int(cx_), int(FLOOR + 6)), (int(rw), int(10 + rw * 0.08)), 0, 0, 360, float(op), -1)
        fs = cv2.GaussianBlur(fs, (0, 0), 18)
        out = out * (1 - fs[..., None])
    G = np.zeros((H, W), np.float32) if GLOW else None
    for idx in order:
        L = layers[idx]
        if L is None: continue
        col, a, _ = L; n, (dx, dy), st = specs[idx]; k = f(st)
        M = place(idx, k, t)
        c = cv2.warpPerspective(col, M, (W, H), flags=cv2.INTER_LINEAR)
        aa = cv2.warpPerspective(a, M, (W, H), flags=cv2.INTER_LINEAR)[..., None]
        out = out * (1 - aa) + c * aa
        if GLOW and k > 0:
            g = 4 * k * (1 - k) + 0.22 * k                   # flares while breaking away / locking in, soft rim at hold
            m = aa[..., 0]
            rim = np.clip(m - cv2.erode(m, np.ones((3, 3), np.uint8)), 0, 1)
            outer = np.clip(cv2.GaussianBlur(m, (0, 0), 9) - m, 0, 1)
            # light streaks trail along the part's direction of travel
            ang = np.degrees(np.arctan2(dy, dx)); ln = int(31 + 70 * g)
            ker = np.zeros((ln, ln), np.float32); ker[ln // 2, :] = 1
            ker = cv2.warpAffine(ker, cv2.getRotationMatrix2D((ln / 2, ln / 2), -ang, 1), (ln, ln)); ker /= max(ker.sum(), 1e-6)
            streak = cv2.filter2D(outer + rim, -1, ker)
            G = np.maximum(G, g * (0.9 * rim + 1.6 * outer + 0.8 * streak))
    if GLOW:
        G = np.clip(G + 0.5 * cv2.GaussianBlur(G, (0, 0), 25), 0, 1)[..., None]   # bloom
        out = out + (GLOW_COL - out) * G                     # screen-like: lifts toward cool white
    return out

def frame(t):
    f = explode(t)
    if all(f(st) == 0 for _, _, st in specs): return np.clip(REAL, 0, 255).astype(np.uint8)
    if not THREE_D: return np.clip(render(t), 0, 255).astype(np.uint8)
    acc = sum(render(t + d / FPS) for d in (-0.33, 0, 0.33)) / 3      # motion blur over ~2/3 of a frame
    return np.clip(acc, 0, 255).astype(np.uint8)

if len(sys.argv) > 2 and not sys.argv[2].startswith('--'):                                     # colour-coded part map for checking the split
    pal = np.random.RandomState(3).randint(40, 255, (len(names) + 1, 3)); pal[0] = 255
    cv2.imwrite(sys.argv[2], pal[lab].astype(np.uint8))
STILL = next((a[8:] for a in sys.argv if a.startswith('--still=')), None)
if STILL:                                                  # just the fully exploded hold frame, for checking or as a keyframe
    cv2.imwrite(STILL, frame(5.0)); sys.exit()
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                      '-vf', 'scale=flags=accurate_rnd+full_chroma_int:out_color_matrix=bt601:out_range=tv',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', sys.argv[1]], stdin=subprocess.PIPE)
for i in range(10 * FPS + 1):
    p.stdin.write(frame(i / FPS).tobytes())
p.stdin.close(); p.wait()
print('done', names)
