# End frame = start frame with only the armrest PAD swapped for the pad of the end photo. The two source photos were
# shot with the chair turned a hair (spine behind the lumbar mesh, backdrop brightness differ), so nothing else may
# come from the end photo.
#  - start: pad removed (soft mask, inpainted wall, post continued up under the pad) -> clean plate
#  - end:   pad lifted with a soft alpha from its darkness against its own (inpainted) wall, colour un-mixed
#  - out = end pad over the clean start plate
# Box/seam/seed are given at 1920x1080 and scaled to the input size.
# Usage: python armside_pad_swap.py start.png end.png out.png   (env: BOX x0,y0,x1,y1  SEAM 521  SEED x,y)
import cv2, numpy as np, sys, os
s_p, e_p, o_p = sys.argv[1:4]
S, E = cv2.imread(s_p).astype(np.float32), cv2.imread(e_p).astype(np.float32); h, w = S.shape[:2]; k = w / 1920
sc = lambda v: int(round(float(v) * k))
x0, y0, x1, _ = [sc(v) for v in os.environ.get('BOX', '388,220,820,0').split(',')]
y1 = sc(os.environ.get('SEAM', 521)); sx, sy = [sc(v) for v in os.environ.get('SEED', '560,380').split(',')]
gray = lambda im: cv2.cvtColor(np.clip(im, 0, 255).astype(np.uint8), cv2.COLOR_BGR2GRAY).astype(np.float32)
LS, LE = gray(S), gray(E)
row = LS[y1 + sc(6)]; cols = np.where(row[x0:x1] < 90)[0] + x0; px0, px1 = cols.min(), cols.max()   # post just below seam
r3 = np.ones((sc(3) | 1, sc(3) | 1), np.uint8)

def lift(I, L):
    dark = np.zeros((h, w), np.uint8); dark[y0:y1 + sc(8), x0:x1] = L[y0:y1 + sc(8), x0:x1] < 110
    dark[y1:, px0 - sc(3):px1 + sc(4)] = 0                                     # the post below the seam stays
    n, lab = cv2.connectedComponents(dark); pad = (lab == lab[sy, sx]).astype(np.uint8)
    pad = cv2.morphologyEx(pad, cv2.MORPH_CLOSE, np.ones((sc(5) | 1,) * 2, np.uint8))
    grow = cv2.dilate(pad, np.ones((sc(7) | 1,) * 2, np.uint8)); grow[y1:, px0 - sc(3):px1 + sc(4)] = 0
    clean = cv2.inpaint(np.clip(I, 0, 255).astype(np.uint8), grow * 255, sc(9), cv2.INPAINT_TELEA).astype(np.float32)
    ref = I[y1 + sc(2):y1 + sc(6), px0 - sc(2):px1 + sc(3)].mean(0)            # post continued up under the pad
    for y in range(y1 - sc(25), y1 + sc(2)):
        sl = slice(px0 - sc(2), px1 + sc(3)); clean[y, sl] = np.where(grow[y, sl, None] > 0, ref, clean[y, sl])
    padL = np.median(L[pad > 0]); bgL = gray(clean)
    a = np.clip((bgL - L) / np.maximum(bgL - padL, 1), 0, 1) * grow
    a = np.where(cv2.erode(pad, r3) > 0, 1.0, a)
    F = (I - (1 - a[..., None]) * clean) / np.maximum(a[..., None], 0.05)
    print('pad bbox', cv2.boundingRect(pad), 'px', int(pad.sum())); return clean, a, F

cleanS, _, _ = lift(S, LS)
_, aE, FE = lift(E, LE)
O = FE * aE[..., None] + cleanS * (1 - aE[..., None])
band = (slice(y1 - sc(8), y1 + sc(10)), slice(x0, x1))                        # tidy the pad/post seam
O[band] = cv2.medianBlur(np.clip(O[band], 0, 255).astype(np.uint8), 5)
cv2.imwrite(o_p, np.clip(O, 0, 255).astype(np.uint8))
d = cv2.absdiff(O, S).max(2); print('changed vs start: %.2f%% of pixels' % (100 * (d > 6).mean()), 'post cols', px0, px1)
