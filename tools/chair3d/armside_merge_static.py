# End frame = start frame everywhere except the armrest pad. The two source photos were shot with the chair turned a
# hair, so the spine behind the lumbar mesh (and other parallax details) differ; only the pad should differ.
# Region taken from the end frame: both pads (dark, in the pad box) dilated by GROW px, only above the pad/post seam,
# feathered over the backdrop. Works at any scale (box given at 1920x1080).
# Usage: python armside_merge_static.py start.png end.png out.png   (env: BOX x0,y0,x1,y1 @1920  GROW 14  FEATHER 8)
import cv2, numpy as np, sys, os
s_p, e_p, o_p = sys.argv[1:4]
S, E = cv2.imread(s_p).astype(np.float32), cv2.imread(e_p).astype(np.float32); h, w = S.shape[:2]; k = w / 1920
x0, y0, x1, y1 = [int(round(int(v) * k)) for v in os.environ.get('BOX', '380,220,820,524').split(',')]
G, FE = int(round(int(os.environ.get('GROW', 14)) * k)), int(os.environ.get('FEATHER', 8)) * k
def pad(im):
    g = cv2.cvtColor(im.astype(np.uint8), cv2.COLOR_BGR2GRAY); m = np.zeros((h, w), np.uint8)
    m[y0:y1, x0:x1] = g[y0:y1, x0:x1] < 90
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    i = 1 + int(np.argmax(st[1:, cv2.CC_STAT_AREA])); return (lab == i).astype(np.uint8)
m = cv2.dilate(pad(S) | pad(E), np.ones((2 * G + 1, 2 * G + 1), np.uint8)); m[y1:] = 0
a = cv2.GaussianBlur(m.astype(np.float32), (0, 0), FE)[..., None]; a[y1:] = 0
O = E * a + S * (1 - a)
cv2.imwrite(o_p, np.clip(O, 0, 255).astype(np.uint8))
d = cv2.absdiff(O, S).max(2); print('pixels changed vs start: %.2f%%' % (100 * (d > 6).mean()), 'region bbox', cv2.boundingRect(m))
