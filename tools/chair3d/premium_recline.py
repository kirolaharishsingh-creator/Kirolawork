# Recline multilock USP in the premium look (premium_look.py): RGBA frames + hold map (recline_finish.py layout) ->
# graded chair on the light or dark studio backdrop with a live angle read-out that counts with the backrest
# (recline keys -> 90 / 120 / 135 deg at the locks). Seamless loop: the last frame leads straight into the first.
# Usage: python premium_recline.py frames_dir map.txt out.mp4   (env: STYLE light|dark, FPS 24,
#        RECLINE keys as rendered, LOCKS 'deg:angle,...' 0:90,18:120,33:135)
import os, sys, subprocess, numpy as np
from PIL import Image
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from premium_look import compose, angle_label, backdrop
fdir, mpath, out = sys.argv[1:4]; FPS = int(os.environ.get('FPS', 24)); STYLE = os.environ.get('STYLE', 'light')
src = {int(a): int(b) for a, b in (p.split(':') for p in open(mpath).read().split())}
KEYS = [tuple(float(v) for v in k.split(':')) for k in os.environ.get('RECLINE', '0:0,0.6:0,1.4:18,2.6:18,3.4:33,4.8:33,5.8:0,7:0').split(',')]
LOCKS = [tuple(float(v) for v in k.split(':')) for k in os.environ.get('LOCKS', '0:90,18:120,33:135').split(',')]
def smooth(u): u = min(max(u, 0.0), 1.0); return u * u * (3 - 2 * u)
def keyed(t):                                                   # same easing as anim3.py
    if t <= KEYS[0][0]: return KEYS[0][1]
    for (t0, a0), (t1, a1) in zip(KEYS, KEYS[1:]):
        if t <= t1: return a0 + (a1 - a0) * smooth((t - t0) / max(t1 - t0, 1e-6))
    return KEYS[-1][1]
def angle(deg): return float(np.interp(deg, [l[0] for l in LOCKS], [l[1] for l in LOCKS]))
W, H = Image.open(f'{fdir}/{src[0]:04d}.png').size; BG = backdrop(W, H, STYLE)
p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                      '-an', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
                     stdin=subprocess.PIPE)
cache = {}
for f in range(len(src)):
    s = src[f]
    if s not in cache: cache = {s: compose(np.asarray(Image.open(f'{fdir}/{s:04d}.png').convert('RGBA')), STYLE, BG.copy())}
    p.stdin.write(angle_label(cache[s], angle(keyed(f / FPS)), 1.0, STYLE).tobytes())
p.stdin.close(); p.wait(); print('ok', out, len(src), 'frames')
