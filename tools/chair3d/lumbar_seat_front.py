# Lumbar end frame touch-up: from this low side view the tilted pad's lower end overlaps the back of the seat. The
# seat's near edge is closer to the camera than the (narrower, centred) pad, so the seat must hide the pad there.
# The seat in that area is taken from the start photo, where the pad does not cover it, and laid over the end frame.
# Usage: python3 lumbar_seat_front.py start.jpg end.jpg out.jpg
import sys, numpy as np, cv2
from PIL import Image, ImageDraw

st = cv2.imread(sys.argv[1]).astype(np.float32); en = cv2.imread(sys.argv[2]).astype(np.float32)
H, W = st.shape[:2]; lum = st.mean(2)
m = Image.new('L', (W, H), 0)
ImageDraw.Draw(m).polygon([(1700, 780), (1950, 850), (2150, 920), (2310, 990), (2310, 1250), (1700, 1250)], 255)
zone = np.asarray(m) > 0
# the black seat fabric (darker than the pad's grey frame), as one solid shape
seat = (zone & (lum < 32)).astype(np.uint8)
k = lambda r: cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * r + 1, 2 * r + 1))
seat = cv2.morphologyEx(seat, cv2.MORPH_CLOSE, k(4)); seat = cv2.morphologyEx(seat, cv2.MORPH_OPEN, k(2))
n, lab, sts, _ = cv2.connectedComponentsWithStats(seat)          # only the seat itself, not dark slots in the pad
seat = (lab == 1 + np.argmax(sts[1:, cv2.CC_STAT_AREA])).astype(np.uint8)
a = cv2.GaussianBlur(seat.astype(np.float32), (0, 0), 0.9)[..., None]
out = st * a + en * (1 - a)
cv2.imwrite(sys.argv[3], np.clip(out + 0.5, 0, 255).astype(np.uint8), [cv2.IMWRITE_JPEG_QUALITY, 95])
print('ok', sys.argv[3])
