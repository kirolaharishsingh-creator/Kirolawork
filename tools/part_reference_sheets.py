# One reference sheet per assembly part: 4 views of the isolated part from the 3D model (anim3.py parts mode, on white)
# above crops of the real product photos that show that part. Used to brief separate per-part modelling / AI 3D.
# Usage: python3 tools/part_reference_sheets.py parts_dir out_dir      (run from the repo root)
import os, sys, numpy as np, cv2
from PIL import Image, ImageOps

pdir, odir = sys.argv[1:3]; os.makedirs(odir, exist_ok=True)
P = 'chair_photos/'
# step, name, 3D objects, real photos with crop (x0, y0, x1, y1 as fractions of the photo)
PARTS = [
    (1, 'base_and_castors', ['base', 'wheel0'], [('01_front_chrome/15.jpg', (.12, .70, .88, .97)), ('03_side_upright_chrome/18.jpg', (.10, .68, .90, .97)), ('06_rear_chrome_Y_spine/21.jpg', (.15, .66, .90, .95))]),
    (2, 'gas_lift', ['gas_lift'], [('14_usp_underside_mechanism/20.jpg', (.25, .55, .75, 1.0)), ('14_usp_underside_mechanism/43.jpg', (.25, .55, .75, 1.0))]),
    (3, 'mechanism_knob_levers', ['mechanism'], [('14_usp_underside_mechanism/20.jpg', (0, .25, 1, .85)), ('14_usp_underside_mechanism/43.jpg', (0, .25, 1, .85))]),
    (4, 'seat', ['seat'], [('13_usp_seat_top_view/33.jpg', (0, 0, 1, 1)), ('13_usp_seat_top_view/38.jpg', (0, 0, 1, 1)), ('09_usp_lumbar_pivot/04.jpg', (0, .45, 1, 1))]),
    (5, 'spine_back_frame', ['frame'], [('06_rear_chrome_Y_spine/28.jpg', (.15, .12, .85, .70)), ('06_rear_chrome_Y_spine/29.jpg', (.15, .12, .85, .62)), ('09_usp_lumbar_pivot/41.jpg', (.45, 0, 1, 1))]),
    (6, 'mesh_backrest', ['backrest'], [('01_front_chrome/15.jpg', (.20, .15, .80, .55)), ('06_rear_chrome_Y_spine/29.jpg', (.20, .12, .80, .55)), ('03_side_upright_chrome/18.jpg', (.15, .10, .65, .55))]),
    (7, 'lumbar_support', ['lumbar'], [('09_usp_lumbar_pivot/04.jpg', (.45, 0, 1, .75)), ('09_usp_lumbar_pivot/41.jpg', (.45, .05, 1, .80)), ('10_usp_lumbar_rear_handle/09.jpg', (0, 0, .75, .80))]),
    (8, 'armrest_left_and_right', ['arm_l'], [('11_usp_armrest_closeups/02.jpg', (0, 0, 1, 1)), ('11_usp_armrest_closeups/26.jpg', (0, 0, 1, 1)), ('12_usp_armrest_4D_positions/34.jpg', (0, .2, 1, .8))]),
    (9, 'headrest', ['headrest'], [('08_usp_headrest/07.jpg', (0, 0, 1, 1)), ('08_usp_headrest/37.jpg', (0, 0, 1, 1))]),
]
H3, HP, W = 420, 520, 1800

def render_views(names):
    tiles = []
    for v in range(4):
        ims = []
        for n in names:
            a = cv2.imread(f'{pdir}/{n}_v{v}.png', cv2.IMREAD_UNCHANGED).astype(np.float32)
            al = a[..., 3:] / 255; im = a[..., :3] * al + 255 * (1 - al)
            ys, xs = np.where(a[..., 3] > 10)
            if len(ys): im = im[max(ys.min() - 20, 0):ys.max() + 20, max(xs.min() - 20, 0):xs.max() + 20]
            ims.append(cv2.resize(im, (int(im.shape[1] * H3 / im.shape[0]), H3)))
        tiles.append(np.hstack(ims) if len(ims) > 1 else ims[0])
    return tiles

def photo(f, c):
    a = np.asarray(ImageOps.exif_transpose(Image.open(P + f)).convert('RGB'))[..., ::-1].astype(np.float32)
    h, w = a.shape[:2]; a = a[int(c[1] * h):int(c[3] * h), int(c[0] * w):int(c[2] * w)]
    return cv2.resize(a, (int(a.shape[1] * HP / a.shape[0]), HP), interpolation=cv2.INTER_AREA)

def row(tiles, h):
    gap = np.full((h, 16, 3), 255, np.float32); r = tiles[0]
    for t in tiles[1:]: r = np.hstack([r, gap, t])
    if r.shape[1] > W: r = cv2.resize(r, (W, int(r.shape[0] * W / r.shape[1])), interpolation=cv2.INTER_AREA)
    return np.hstack([r, np.full((r.shape[0], W - r.shape[1], 3), 255, np.float32)])

def label(t, h=60):
    b = np.full((h, W, 3), 255, np.uint8); cv2.putText(b, t, (16, 42), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (30, 30, 30), 2, cv2.LINE_AA); return b.astype(np.float32)

for step, name, objs, photos in PARTS:
    sheet = np.vstack([label(f'Part {step}: {name.replace("_", " ")}  -  3D model views (front, back, side, top)'), row(render_views(objs), H3),
                       label('Real product photos'), row([photo(f, c) for f, c in photos], HP)])
    cv2.imwrite(f'{odir}/{step:02d}_{name}.jpg', sheet.astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 92]); print(step, name)
