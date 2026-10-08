# Sandbox setup: Blender module, the Tripo model, cached arrays, and the current scripts at commit $1.
cd /home/user
python3 -c "import bpy, scipy" 2>/dev/null || pip install -q bpy scipy
[ -f chair.glb ] || curl -sSf -o chair.glb "https://d8j0ntlcm91z4.cloudfront.net/user_3JAuxJYdZlc38Qcbn91XwOtqpSI/hf_20261008_012925_2a6fd930-54f6-4bab-bc92-0ede64690ed8.glb"
B=https://raw.githubusercontent.com/kirolaharishsingh-creator/Kirolawork/$1/tools/chair3d
for f in parts.py anim.py comp.py prep.py explodecheck.py; do curl -sSf -o $f $B/$f || echo FAIL $f; done
[ -f tri.npy ] || python3 prep.py
