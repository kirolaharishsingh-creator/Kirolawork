# Finish the recline USP: RGBA frames (anim3.py RECLINE, only the changing frames rendered) -> full sequence (hold frames
# repeated from map.txt 'frame:source ...'), studio backdrop (comp.over), a small angle label that fades in at each lock
# (90 / 120 / 135 deg) in the top-left (only with LABEL=1), H.264 1080p. Thin see-through slits between the separately
# generated parts (headrest post / spine clamp, lumbar bracket / spine, armrest pad / post) are closed: transparent pixels
# that a SLIT-px closing of the silhouette fills get inpainted from the surrounding chair.
# Usage: python recline_finish.py frames_dir map.txt out.mp4      (env: FPS 24, LABEL=1, LABELS 'from_s:to_s:text;...',
#        SLIT 11, SLIT_YMAX 760: no filling below this row - the base and its shadow)
import os, sys, subprocess, numpy as np, cv2
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
cache = {}; SLIT = int(os.environ.get('SLIT', 11)); SLIT_YMAX = int(os.environ.get('SLIT_YMAX', 760))
def close_slits(a):
    if not SLIT: return a
    op = (a[..., 3] > 128).astype(np.uint8)
    slit = (cv2.morphologyEx(op, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (SLIT, SLIT))) > 0) & (op == 0)
    slit[SLIT_YMAX:] = False; a = a.copy()
    a[..., :3] = cv2.inpaint(np.ascontiguousarray(a[..., :3]), slit.astype(np.uint8), 5, cv2.INPAINT_TELEA); a[slit, 3] = 255
    return a
p = subprocess.Popen(['ffmpeg', '-y', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{W}x{H}', '-r', str(FPS), '-i', '-',
                      '-an', '-c:v', 'libx264', '-crf', '14', '-preset', 'slow', '-pix_fmt', 'yuv420p', '-movflags', '+faststart', out],
                     stdin=subprocess.PIPE)
for f in range(len(src)):
    s = src[f]
    if s not in cache: cache = {s: over(close_slits(np.asarray(Image.open(f'{fdir}/{s:04d}.png').convert('RGBA'))), BG)}
    im = Image.fromarray(cache[s]).convert('RGBA')
    if os.environ.get('LABEL') == '1':
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
