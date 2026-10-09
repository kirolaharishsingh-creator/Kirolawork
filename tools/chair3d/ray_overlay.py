# Add the thin light ray (the one anim3.py draws in 3D) to a finished video, e.g. an AI-generated exploded clip.
# The chair is near-black on a light studio backdrop, so each frame's chair pixels are found by brightness; a thin,
# slightly diagonal cool-white line sweeps up across them (as the parts detach), with a soft glow, and nothing is
# drawn on the backdrop or the floor shadow.
# Usage: python3 ray_overlay.py in.mp4 out.mp4
#   env: T0,T1   sweep start/end as fractions of the clip (default 0.08, 0.80)
#        SWEEPS  number of sweeps between T0 and T1 (default 2: one per wave of parts leaving)
#        WIDTH   line half-width as a fraction of the chair height (default 0.012)
#        GAIN    brightness (default 1.0), DARK  chair threshold on 0-255 luminance (default 95)
#        MODE    'ray' (a line sweeping across the parts) or 'outline' (a thin glowing outline traced around every
#                part's edge while the parts separate; fades in at T0 and out at T1)
import os, sys, subprocess, json, numpy as np
from PIL import Image, ImageFilter
from scipy import ndimage

src, dst = sys.argv[1], sys.argv[2]
T0, T1 = float(os.environ.get('T0', 0.08)), float(os.environ.get('T1', 0.80))
SWEEPS = int(os.environ.get('SWEEPS', 2)); WIDTH = float(os.environ.get('WIDTH', 0.012))
GAIN = float(os.environ.get('GAIN', 1.0)); DARK = float(os.environ.get('DARK', 95))
COLOR = np.array([0.85, 0.93, 1.0], np.float32)
MODE = os.environ.get('MODE', 'ray'); EDGE = float(os.environ.get('EDGE', 2.0))

info = json.loads(subprocess.check_output(['ffprobe', '-v', 'error', '-select_streams', 'v:0', '-count_frames', '-show_entries',
                                           'stream=width,height,r_frame_rate,nb_read_frames', '-of', 'json', src]))['streams'][0]
w, h, n = info['width'], info['height'], int(info['nb_read_frames']); fps = info['r_frame_rate']
dec = subprocess.Popen(['ffmpeg', '-v', 'error', '-i', src, '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-'], stdout=subprocess.PIPE)
enc = subprocess.Popen(['ffmpeg', '-v', 'error', '-y', '-f', 'rawvideo', '-pix_fmt', 'rgb24', '-s', f'{w}x{h}', '-r', fps, '-i', '-',
                        '-i', src, '-map', '0:v', '-map', '1:a?', '-c:v', 'libx264', '-crf', '16', '-pix_fmt', 'yuv420p', '-c:a', 'copy', dst],
                       stdin=subprocess.PIPE)
yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)

def envelope(u):
    # which sweep is running and how far along it is (0..1); None between sweeps
    if u < T0 or u > T1: return None, 0.0
    s = (u - T0) / (T1 - T0) * SWEEPS; k = int(min(s, SWEEPS - 1e-6)); return k, s - k

for i in range(n):
    f = np.frombuffer(dec.stdout.read(w * h * 3), np.uint8).reshape(h, w, 3).astype(np.float32) / 255
    u = i / max(n - 1, 1)
    if MODE == 'outline':
        a = min(max((u - T0) / 0.06, 0), 1) * min(max((T1 - u) / 0.10, 0), 1)     # quick fade in, slower fade out
        a = a * a * (3 - 2 * a) * GAIN
        if a > 0:
            lum = f.mean(-1) * 255
            dark = lum < DARK
            # chrome and bright parts: anything that stands out from the smooth backdrop (the backdrop is estimated by
            # blurring the frame with the dark parts filled in from around them)
            wgt = (~ndimage.binary_dilation(dark, iterations=3)).astype(np.float32)        # backdrop pixels only
            blur = lambda x, r: np.asarray(Image.fromarray(np.clip(x * 255, 0, 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(r)), np.float32) / 255
            ws = blur(wgt, 40)[..., None]; g = np.stack([blur(f[..., c] * wgt, 40) for c in range(3)], -1) / np.maximum(ws, 1e-3)
            bright = ((f - g).max(-1) * 255 > 30) & ndimage.binary_dilation(dark, iterations=25)   # chrome highlights; floor shadows are darker, not brighter
            part = ndimage.binary_opening(dark | bright, iterations=1)
            part = ndimage.binary_fill_holes(ndimage.binary_closing(part, iterations=1))
            lab, nl = ndimage.label(part)
            if nl:
                sz = ndimage.sum(part, lab, range(1, nl + 1)); part = np.isin(lab, 1 + np.nonzero(sz > 60)[0])
            inner = ndimage.binary_erosion(part, iterations=max(1, int(round(EDGE))))
            edge = (part & ~inner).astype(np.float32)
            edge = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(0.7)), np.float32) / 255
            glow = np.asarray(Image.fromarray((edge * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(5)), np.float32) / 255
            add = (edge * 1.0 + glow * 0.8)[..., None] * COLOR * a
            f = 1 - (1 - f) * (1 - np.clip(add, 0, 1))
        enc.stdin.write((np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes()); continue
    k, p = envelope(u)
    if k is not None:
        lum = f.mean(-1) * 255
        chair = ndimage.binary_opening(lum < DARK, iterations=1)
        chair = ndimage.binary_fill_holes(ndimage.binary_closing(chair, iterations=2))
        if chair.any():
            ys, xs = np.nonzero(chair); y0, y1, x0, x1 = ys.min(), ys.max(), xs.min(), xs.max(); H = max(y1 - y0, 1)
            # line position: bottom -> top, tilted a little (like the 3D ray), eased in and out
            e = p * p * (3 - 2 * p)
            pos = (y1 - yy) / H * 0.85 + (xx - x0) / max(x1 - x0, 1) * 0.15
            d = np.abs(pos - (-0.05 + 1.10 * e)) / WIDTH
            line = np.clip(1 - d, 0, 1) ** 1.5
            m = chair.astype(np.float32); m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(1)), np.float32) / 255
            core = line * m
            glow = np.asarray(Image.fromarray((core * 255).astype(np.uint8)).filter(ImageFilter.GaussianBlur(float(max(2.0, H * 0.012)))), np.float32) / 255
            strength = np.sin(np.pi * p) ** 0.5 * GAIN
            add = (core * 0.95 + glow * 0.6 * m)[..., None] * COLOR * strength
            f = 1 - (1 - f) * (1 - np.clip(add, 0, 1))          # screen blend: brightens the chair, never darkens
    enc.stdin.write((np.clip(f, 0, 1) * 255).astype(np.uint8).tobytes())
enc.stdin.close(); enc.wait(); dec.wait()
print('ok', dst, n, 'frames')
