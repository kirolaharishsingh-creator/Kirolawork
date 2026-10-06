# Draw a thin, crisp, cool-white light line that travels up the outer edge of the chair's backrest.
# The studio background is plain light grey, so the chair silhouette is found per frame by brightness.
# Usage: python3 tools/rim_line.py in.mp4 out.mp4 [--start S] [--dur D] [--side right|left]
#        [--top F] [--bottom F] [--width PX] [--length F] [--inset PX]
import argparse, subprocess
import cv2, numpy as np

ap = argparse.ArgumentParser()
ap.add_argument('inp'); ap.add_argument('out')
ap.add_argument('--start', type=float, default=0.0)
ap.add_argument('--dur', type=float, default=None)
ap.add_argument('--side', default='right')
ap.add_argument('--top', type=float, default=0.0, help='trace from this fraction of the chair height (0 = top)')
ap.add_argument('--bottom', type=float, default=0.55, help='down to this fraction of the chair height')
ap.add_argument('--width', type=float, default=3.0, help='line width in px at 1080p')
ap.add_argument('--length', type=float, default=0.35, help='lit segment length, fraction of the traced edge')
ap.add_argument('--inset', type=float, default=2.0, help='px inside the silhouette, so it reads as a rim on the frame')
a = ap.parse_args()

COLOR = np.array([255, 250, 244], np.float32)          # BGR: neutral cool white

def silhouette(f):
    g = cv2.cvtColor(f, cv2.COLOR_BGR2GRAY)
    m = (g < 120).astype(np.uint8)
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))
    n, lab, st, _ = cv2.connectedComponentsWithStats(m)
    if n < 2: return None
    return (lab == 1 + np.argmax(st[1:, 4])).astype(np.uint8)

def edge_curve(m):
    ys = np.where(m.any(1))[0]
    top, bot = ys.min(), ys.max()
    y0, y1 = int(top + a.top * (bot - top)), int(top + a.bottom * (bot - top))
    pts = []
    for y in range(y0, y1):
        xs = np.where(m[y])[0]
        if len(xs): pts.append((xs.max() - a.inset if a.side == 'right' else xs.min() + a.inset, y))
    if len(pts) < 10: return None
    p = np.array(pts, np.float32)
    p[:, 0] = cv2.GaussianBlur(p[:, 0].reshape(-1, 1), (1, 0), 3).ravel()   # smooth the edge
    return p[::-1]                                      # bottom -> top, the direction the light travels

cap = cv2.VideoCapture(a.inp); fps = cap.get(cv2.CAP_PROP_FPS)
W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
cap.set(cv2.CAP_PROP_POS_FRAMES, int(round(a.start * fps)))
total = int(round((a.dur if a.dur else cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps - a.start) * fps))
scale = H / 1080
p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'bgr24', '-s', f'{W}x{H}', '-r', str(fps), '-i', '-',
                      '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', a.out], stdin=subprocess.PIPE)
for i in range(total):
    ok, f = cap.read()
    if not ok: break
    t = i / max(total - 1, 1)
    m = silhouette(f); curve = edge_curve(m) if m is not None else None
    if curve is not None:
        # head of the lit segment travels from below the start to past the end; fade in/out at the clip ends
        head = -a.length + t * (1 + 2 * a.length)
        s = np.linspace(0, 1, len(curve))
        u = (head - s) / a.length                        # 0 at the head, 1 at the tail
        inten = np.where((u >= 0) & (u <= 1), np.sin(np.pi * np.clip(u, 0, 1)) ** 0.6, 0)
        layer = np.zeros((H, W), np.float32)
        lw = max(1, int(round(a.width * scale)))
        for k in range(len(curve) - 1):
            if inten[k] > 0.02:
                cv2.line(layer, tuple(np.round(curve[k]).astype(int)), tuple(np.round(curve[k + 1]).astype(int)),
                         float(inten[k]), lw, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 0.6 * scale)[..., None] * m[..., None]   # keep it on the chair
        f = (f.astype(np.float32) * (1 - layer) + COLOR * layer).clip(0, 255).astype(np.uint8)
    p.stdin.write(f.tobytes())
p.stdin.close(); p.wait()
