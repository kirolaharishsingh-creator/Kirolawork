# Premium look for the USP renders (client: "video looks dull - premium look, light, contrast"): RGBA frame from anim3.py
# -> graded chair + studio backdrop + floor shadow (+ optional angle read-out). Two backdrops:
#   light: cool white seamless, soft key light behind the chair, gentle vignette   (Apple-style product page)
#   dark : charcoal seamless with a soft glow behind the chair for separation        (luxury / night look)
# Chair grade: S-curve contrast with deeper blacks, clarity (large-radius local contrast), fine sharpening,
# chrome highlights lifted, slightly cool neutral. The floor shadow comes from the frame's semi-transparent alpha.
import numpy as np, cv2, os
from PIL import Image, ImageDraw, ImageFont
FONT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'fonts')
def font(w, size): return ImageFont.truetype(os.path.join(FONT_DIR, 'montserrat-latin-%d-normal.woff' % w), size)

def backdrop(w, h, style):
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32); X, Y = xx / w, yy / h
    if style == 'light':
        top, hor = np.array([244, 245, 247.]), np.array([222, 224, 228.])
        t = np.clip(Y / 0.80, 0, 1)[..., None] ** 1.4; bg = top + (hor - top) * t
        floor = np.clip((Y - 0.80) / 0.20, 0, 1)[..., None]; bg = bg + floor * 10           # floor a touch brighter (bounce)
        spot = np.exp(-(((X - 0.55) / 0.30) ** 2 + ((Y - 0.45) / 0.42) ** 2))[..., None]; bg = bg + 9 * spot
        vig = np.clip(np.hypot((X - 0.5) / 0.75, (Y - 0.5) / 0.85), 0, 1.3)[..., None]; bg = bg * (1 - 0.13 * vig ** 2.2)
    else:
        c, e = np.array([66, 68, 73.]), np.array([12, 13, 15.])
        r = np.clip(np.hypot((X - 0.55) / 0.55, (Y - 0.48) / 0.70), 0, 1.4)[..., None]
        bg = e + (c - e) * np.clip(1 - r, 0, 1) ** 1.6
        floor = np.clip((Y - 0.80) / 0.20, 0, 1)[..., None]; bg = bg * (1 - 0.25 * floor)
    return np.clip(bg + np.random.default_rng(5).normal(0, 0.8, bg.shape), 0, 255)

def grade_chair(rgb):
    x = rgb.astype(np.float32) / 255
    x = x + 0.45 * (x - cv2.GaussianBlur(x, (0, 0), 14))                                 # clarity
    x = x + 0.35 * (x - cv2.GaussianBlur(x, (0, 0), 1.2))                                # fine detail
    x = np.clip(x, 0, 1)
    x = 0.5 + (x - 0.5) * 1.12; x = np.clip(x, 0, 1) ** 1.12                             # contrast, deeper blacks
    lum = x.mean(2, keepdims=True); hi = np.clip((lum - 0.45) / 0.4, 0, 1)
    x = x + hi * (1 - x) * 0.35                                                           # chrome / speculars pop
    x = x * np.array([0.985, 1.0, 1.03])                                                  # cool neutral
    return np.clip(x * 255, 0, 255)

def compose(rgba, style, bg=None):
    rgba = rgba.astype(np.float32); h, w = rgba.shape[:2]
    bg = backdrop(w, h, style) if bg is None else bg
    a = rgba[..., 3:] / 255
    solid = (a > 0.85).astype(np.float32)
    chair = grade_chair(rgba[..., :3])
    if style == 'dark':                                                                    # soft glow behind the chair
        glow = cv2.GaussianBlur(solid[..., 0], (0, 0), 55)[..., None]
        bg = bg + glow * np.array([70, 72, 76.]) * 0.55
    shadow_k = 1.25 if style == 'light' else 1.0
    a2 = np.clip(np.where(solid > 0, a, a * shadow_k), 0, 1)                               # deeper contact shadow
    out = chair * a2 + bg * (1 - a2)
    return np.clip(out, 0, 255).astype(np.uint8)

def angle_label(img, deg, alpha, style, sub='MULTILOCK RECLINE'):
    # big light-weight number with a hairline and spaced caps, left third, fades with alpha (0..1)
    im = Image.fromarray(img).convert('RGBA'); W, H = im.size
    lay = Image.new('RGBA', im.size, (0, 0, 0, 0)); d = ImageDraw.Draw(lay)
    col = (22, 23, 26) if style == 'light' else (240, 241, 243); a8 = int(255 * np.clip(alpha, 0, 1))
    x, y = int(W * 0.085), int(H * 0.30)
    big = font(200, int(H * 0.16)); txt = '%d°' % round(deg)
    d.text((x, y), txt, font=big, fill=col + (a8,))
    ly = y + int(H * 0.20); d.line([(x + 4, ly), (x + int(W * 0.11), ly)], fill=col + (int(a8 * 0.6),), width=max(1, H // 540))
    sm = font(500, int(H * 0.022)); cx = x + 4
    for ch in sub:                                                                        # letter-spaced caps
        d.text((cx, ly + int(H * 0.022)), ch, font=sm, fill=col + (int(a8 * 0.8),)); cx += sm.getlength(ch) + int(H * 0.006)
    return np.asarray(Image.alpha_composite(im, lay).convert('RGB'))
