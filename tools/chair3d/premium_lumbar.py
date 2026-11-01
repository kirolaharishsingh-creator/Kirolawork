# Lumbar USP in the premium look (premium_look.py): close-up RGBA frames (anim3.py LUMB_SLIDE, only the lumbar moves)
# + hold map -> graded chair on the light studio backdrop, a "LUMBAR SUPPORT" call-out whose pointer dot sits on the
# lumbar pad and moves with it, and a live depth read-out (0 -> +2 cm). Seamless loop (last frame = first).
# The pad position is measured from the frames: rest frame vs the fully-out hold frame (phase correlation in the pad box).
# Usage: python premium_lumbar.py frames_dir map.txt out.mp4   (env: LUMB_SLIDE keys as rendered, STYLE light, FPS 24)
import os, sys, subprocess, numpy as np, cv2
from PIL import Image, ImageDraw
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from premium_look import compose, backdrop, font
fdir, mpath, out = sys.argv[1:4]; FPS = int(os.environ.get('FPS', 24)); STYLE = os.environ.get('STYLE', 'light')
src = {int(a): int(b) for a, b in (p.split(':') for p in open(mpath).read().split())}
KEYS = [tuple(float(v) for v in k.split(':')) for k in os.environ.get('LUMB_SLIDE', '0:0,0.6:0,1.6:0.02,3.0:0.02,4.0:0,5:0').split(',')]
DMAX = max(k[1] for k in KEYS)
def smooth(u): u = min(max(u, 0.0), 1.0); return u * u * (3 - 2 * u)
def keyed(t):
    if t <= KEYS[0][0]: return KEYS[0][1]
    for (t0, a0), (t1, a1) in zip(KEYS, KEYS[1:]):
        if t <= t1: return a0 + (a1 - a0) * smooth((t - t0) / max(t1 - t0, 1e-6))
    return KEYS[-1][1]
load = lambda s: np.asarray(Image.open(f'{fdir}/{s:04d}.png').convert('RGBA'))
A0 = load(src[0]); W_, H_ = A0.shape[1], A0.shape[0]
fout = max(src.values(), key=lambda s: keyed(s / FPS)); A1 = load(fout)                   # a fully-out frame
moved = (np.abs(A0[..., 3].astype(int) - A1[..., 3].astype(int)) > 60).astype(np.uint8)
x, y, w, h = cv2.boundingRect(cv2.dilate(moved, np.ones((25, 25), np.uint8)))            # pad box
g0 = A0[y:y + h, x:x + w, 3].astype(np.float32); g1 = A1[y:y + h, x:x + w, 3].astype(np.float32)
(dx, dy), _ = cv2.phaseCorrelate(g0, g1)
# anchor: front-most (left-most) solid point of the pad at rest, at the pad box's mid height band
band = A0[y + h // 3: y + 2 * h // 3, x:x + w, 3] > 200; cols = np.where(band.any(0))[0]
ax0 = x + cols.min() + 18; ay0 = y + h // 3 + int(np.argmax(band[:, cols.min()]))
print('pad box', (x, y, w, h), 'full-out shift px', round(dx, 1), round(dy, 1), 'anchor', ax0, ay0)
BG = backdrop(W_, H_, STYLE)
col = (22, 23, 26)
def overlay(img, d):
    im = Image.fromarray(img).convert('RGBA'); lay = Image.new('RGBA', im.size, (0, 0, 0, 0)); dr = ImageDraw.Draw(lay)
    k = d / DMAX; px, py = ax0 + dx * k, ay0 + dy * k
    tx, ty = int(W_ * 0.085), int(H_ * 0.34)
    sm = font(500, int(H_ * 0.022)); cx = tx + 3
    for ch in 'ADJUSTABLE':
        dr.text((cx, ty), ch, font=sm, fill=col + (200,)); cx += sm.getlength(ch) + int(H_ * 0.006)
    big = font(200, int(H_ * 0.085)); dr.text((tx, ty + int(H_ * 0.03)), 'Lumbar Support', font=big, fill=col + (255,))
    ly = ty + int(H_ * 0.145); lx1 = tx + int(big.getlength('Lumbar Support')) + 2
    dr.line([(tx + 3, ly), (lx1, ly)], fill=col + (150,), width=2)
    dr.line([(lx1, ly), (px - 14, py)], fill=col + (150,), width=2)                         # leader to the pad
    r = 9; dr.ellipse([px - r, py - r, px + r, py + r], outline=col + (230,), width=2)
    dr.ellipse([px - 3, py - 3, px + 3, py + 3], fill=col + (230,))
    md = font(300, int(H_ * 0.034)); dr.text((tx + 2, ly + int(H_ * 0.02)), 'DEPTH  +%.1f cm' % (d * 100), font=md, fill=col + (215,))
    return np.asarray(Image.alpha_composite(im, lay).convert('RGB'))
p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W_}x{H_}', '-r', str(FPS), '-i', '-',
                      '-an', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
                     stdin=subprocess.PIPE)
cache = {}
for f in range(len(src)):
    s = src[f]
    if s not in cache: cache = {s: compose(load(s), STYLE, BG.copy())}
    p.stdin.write(overlay(cache[s], keyed(f / FPS)).tobytes())
p.stdin.close(); p.wait(); print('ok', out, len(src), 'frames')
