# Exploded-view animation built only from real photo pixels (photo 31, three-quarter view).
# The cutout is split into parts with hand-drawn polygons (seeds) + nearest-seed fill; each part slides out
# along its own direction, holds, and slides back. First and last frames are the untouched real chair.
# Usage: python3 tools/exploded_view.py out.mp4 [debug_parts.png]
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
WHEELS = [((300,1640),45), ((830,1640),45), ((70,1810),58), ((1050,1810),58), ((560,1930),58)]   # display coords, radius

plain4k, shadow4k, chair4k, (px4, py4) = bk.build(SRC, None, use_rim=False, scale=0.62)
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
    layers.append((col.astype(np.float32), a))

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

def frame(t):
    f = explode(t)
    if all(f(st) == 0 for _, _, st in specs): return np.clip(REAL, 0, 255).astype(np.uint8)   # assembled: the real photo
    sh = cv2.warpAffine(SHADOW, np.float32([[1, 0, 0], [0, 1, f(0.45) * 330 * sy]]), (W, H), borderValue=(1, 1, 1))
    out = BG * sh
    for idx in order:
        L = layers[idx]
        if L is None: continue
        col, a = L; n, (dx, dy), st = specs[idx]; k = f(st)
        M = np.float32([[sx, 0, ox + dx * k * sx], [0, sy, oy + dy * k * sy]])
        c = cv2.warpAffine(col, M, (W, H), flags=cv2.INTER_AREA)
        aa = cv2.warpAffine(a, M, (W, H), flags=cv2.INTER_AREA)[..., None]
        out = out * (1 - aa) + c * aa
    return np.clip(out, 0, 255).astype(np.uint8)

if len(sys.argv) > 2:                                     # colour-coded part map for checking the split
    pal = np.random.RandomState(3).randint(40, 255, (len(names) + 1, 3)); pal[0] = 255
    cv2.imwrite(sys.argv[2], pal[lab].astype(np.uint8))
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                      '-vf', 'scale=flags=accurate_rnd+full_chroma_int:out_color_matrix=bt601:out_range=tv',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', sys.argv[1]], stdin=subprocess.PIPE)
for i in range(10 * FPS + 1):
    p.stdin.write(frame(i / FPS).tobytes())
p.stdin.close(); p.wait()
print('done', names)
