# Finish the recline USP: RGBA frames (anim3.py RECLINE, only the changing frames rendered) -> full sequence (hold frames
# repeated from map.txt 'frame:source ...'), studio backdrop (comp.over), a small angle label that fades in at each lock
# (90 / 120 / 135 deg) in the top-left, H.264 1080p.
# Usage: python recline_finish.py frames_dir map.txt out.mp4      (env: FPS 24, LABELS 'from_s:to_s:text;...', NOLABEL=1)
import os, sys, subprocess, numpy as np
from PIL import Image, ImageDraw, ImageFont
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from comp import over, studio_bg

fdir, mpath, out = sys.argv[1:4]; FPS = int(os.environ.get('FPS', 24))
src = {int(a): int(b) for a, b in (p.split(':') for p in open(mpath).read().split())}
LABELS = [(float(a), float(b), t) for a, b, t in (x.split(':') for x in
          os.environ.get('LABELS', '0:0.6:90°;1.4:2.6:120°;3.4:4.8:135°;5.8:7:90°').split(';'))]
first = Image.open(f'{fdir}/{src[0]:04d}.png'); W, H = first.size; BG = studio_bg(W, H)
FONT = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', int(H * 0.075))
SMALL = ImageFont.truetype('/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf', int(H * 0.024))
cache = {}
p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                      '-an', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
                     stdin=subprocess.PIPE)
for f in range(len(src)):
    s = src[f]
    if s not in cache: cache = {s: over(np.asarray(Image.open(f'{fdir}/{s:04d}.png').convert('RGBA')), BG)}
    im = Image.fromarray(cache[s]).convert('RGBA')
    if os.environ.get('NOLABEL') != '1':
        t = f / FPS
        for a, b, txt in LABELS:
            if a - 0.25 <= t <= b + 0.1:
                k = min(1, (t - (a - 0.25)) / 0.25, (b + 0.1 - t) / 0.25) if b < len(src) / FPS - 0.01 else min(1, (t - (a - 0.25)) / 0.25)
                lay = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
                x, y = int(W * 0.07), int(H * 0.12); a8 = int(255 * max(0, k))
                d.text((x, y), txt, font=FONT, fill=(28, 28, 30, a8))
                d.text((x + 4, y + int(H * 0.095)), 'MULTILOCK RECLINE', font=SMALL, fill=(70, 70, 74, int(a8 * 0.85)))
                im = Image.alpha_composite(im, lay)
    p.stdin.write(np.asarray(im.convert('RGB')).tobytes())
p.stdin.close(); p.wait(); print('ok', out)
