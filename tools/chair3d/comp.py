# Composite the RGBA Blender frames over the light grey studio background and encode.
# Usage: python3 comp.py frames out.mp4   |   python3 comp.py frames sheet.jpg 0,30,60,90
import sys, glob, subprocess, numpy as np
from PIL import Image
src, out = sys.argv[1], sys.argv[2]
files = sorted(glob.glob(f'{src}/*.png'))
w, h = Image.open(files[0]).size
yy, xx = np.mgrid[0:h, 0:w]
r = np.hypot((xx - w * 0.5) / (w * 0.6), (yy - h * 0.45) / (h * 0.75))
tone = np.array([202, 198, 194], np.float32) / 255           # timeline studio grey (RGB)
BG = np.clip(tone * (1.13 - 0.16 * np.clip(r, 0, 1.4)[..., None]), 0, 1)
def comp(f):
    im = np.asarray(Image.open(f).convert('RGBA'), np.float32) / 255
    a = im[..., 3:]
    return (np.clip(im[..., :3] * a + BG * (1 - a), 0, 1) * 255).astype(np.uint8)
if out.endswith('.jpg'):
    sel = [int(v) for v in sys.argv[3].split(',')]
    tiles = [comp(f'{src}/{i:04d}.png') for i in sel]
    rows = [np.hstack(tiles[i:i + 2]) for i in range(0, len(tiles), 2)]
    Image.fromarray(np.vstack(rows)).resize((w, h * len(rows) // 2)).save(out, quality=72)
else:
    p = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', '24', '-i', '-',
                          '-c:v', 'libx264', '-crf', '17', '-pix_fmt', 'yuv420p', out], stdin=subprocess.PIPE)
    for f in files: p.stdin.write(comp(f).tobytes())
    p.stdin.close(); p.wait()
print('ok', out)
