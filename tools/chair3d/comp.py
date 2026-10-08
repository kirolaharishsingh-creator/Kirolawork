# Composite the RGBA Blender frames over the light grey studio background and encode.
# Usage: python3 comp.py frames out.mp4   |   python3 comp.py frames sheet.jpg 0,30,60,90
import sys, os, glob, subprocess, numpy as np
from PIL import Image, ImageFilter

FILL = np.array([40, 40, 42], np.float32) / 255          # fallback colour for pin-holes with no surface around them

def studio_bg(w, h):
    if os.environ.get('BG') == 'ref':                 # USP look: mid grey at the top fading to near white at the floor
        t = np.clip(np.linspace(0, 1, h) / 0.85, 0, 1)[:, None, None] ** 0.8
        top, bot = np.array([112, 112, 113], np.float32) / 255, np.array([238, 238, 238], np.float32) / 255
        return np.broadcast_to(top + (bot - top) * t, (h, w, 3)).copy()
    yy, xx = np.mgrid[0:h, 0:w]
    r = np.hypot((xx - w * 0.5) / (w * 0.6), (yy - h * 0.45) / (h * 0.75))
    tone = np.array([202, 198, 194], np.float32) / 255       # timeline studio grey (RGB)
    return np.clip(tone * (1.13 - 0.16 * np.clip(r, 0, 1.4)[..., None]), 0, 1)

def close_alpha(a8, size=9):
    # morphological closing: small see-through holes inside a part become solid; gaps between parts stay open
    im = Image.fromarray(a8)
    return np.asarray(im.filter(ImageFilter.MaxFilter(size)).filter(ImageFilter.MinFilter(size)), np.float32) / 255

def neighbour_colour(rgb, a, r=6):
    # colour of the surrounding surface: blur of the premultiplied colour divided by blur of the alpha
    pm = Image.fromarray((np.clip(rgb * a, 0, 1) * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(r))
    wa = Image.fromarray((a[..., 0] * 255).astype(np.uint8)).filter(ImageFilter.BoxBlur(r))
    w = np.asarray(wa, np.float32)[..., None] / 255
    return np.where(w > 1e-3, np.asarray(pm, np.float32) / 255 / np.maximum(w, 1e-3), FILL)

def fill_pinholes(rgba):
    # small see-through specks fully enclosed by the chair (open slots Tripo left in the rims) take the colour
    # around them; real openings are far larger, and the gaps between flying parts are not enclosed
    from scipy import ndimage
    h, w = rgba.shape[:2]; a = rgba[..., 3]
    hole = ndimage.binary_fill_holes(a > 128) & (a <= 128)
    lab, n = ndimage.label(hole)
    if not n: return rgba
    size = ndimage.sum(hole, lab, index=np.arange(1, n + 1)); small = np.zeros(n + 1, bool)
    small[1:] = size < float(os.environ.get('PINHOLE_FRAC', 0.0002)) * w * h
    m = small[lab]
    if not m.any(): return rgba
    out = rgba.astype(np.float32) / 255
    fill = neighbour_colour(out[..., :3], out[..., 3:], r=4)
    out[m, :3] = fill[m]; out[m, 3] = 1.0
    return (out * 255).astype(np.uint8)

def over(rgba, bg):
    if os.environ.get('PINHOLES', '1') == '1': rgba = fill_pinholes(rgba)
    im = rgba.astype(np.float32) / 255; a = im[..., 3:]
    # CLOSE=0 for models without pin-holes (model 8): the closing would bridge small gaps between parts with grey
    if os.environ.get('CLOSE', '1') == '0': return (np.clip(im[..., :3] * a + bg * (1 - a), 0, 1) * 255).astype(np.uint8)
    ac = np.maximum(close_alpha(rgba[..., 3])[..., None], a)
    fill = neighbour_colour(im[..., :3], a)              # pin-holes take the fabric colour around them
    return (np.clip(im[..., :3] * a + fill * (ac - a) + bg * (1 - ac), 0, 1) * 255).astype(np.uint8)

if __name__ == '__main__':
    src, out = sys.argv[1], sys.argv[2]
    files = sorted(glob.glob(f'{src}/*.png'))
    w, h = Image.open(files[0]).size
    BG = studio_bg(w, h)
    comp = lambda f: over(np.asarray(Image.open(f).convert('RGBA')), BG)
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
