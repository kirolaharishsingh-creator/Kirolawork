# 16:9 crop of a portrait/any photo at a given centre, optional mirror. Feed the result to build_closeup.py.
# Usage: python3 tools/crop169.py photo.jpg out.jpg cy [mirror]   (full width, cy = crop centre y in px)
import sys
from PIL import Image, ImageOps
im = ImageOps.exif_transpose(Image.open(sys.argv[1])).convert('RGB')
if 'mirror' in sys.argv[4:]: im = ImageOps.mirror(im)
w, h = im.size; ch = int(w * 9 / 16); cy = int(sys.argv[3])
y0 = max(0, min(h - ch, cy - ch // 2))
im.crop((0, y0, w, y0 + ch)).save(sys.argv[2], quality=97)
