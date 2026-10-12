# Tripo part models

Per-part GLBs generated in Tripo from `exploded/parts_reference/tripo_final` / Nano Banana part images.
The assembled chair (`chair_tripo_v5.blend`, ~140 MB, too large for GitHub) is rebuilt from these with:

    OUT=chair_tripo_v2.blend FACES=200000 PARTS="seat=04_seat.glb;backrest=<06>.glb;frame=<05>.glb;lumbar=<07>.glb;arm_r=<08>.glb;arm_l=<08>.glb;base+wheel_0+wheel_1+wheel_2+wheel_3+wheel_4=<01>.glb" \
      python tools/chair3d/fit_tripo_parts.py
    python tools/chair3d/finish_tripo_chair.py chair_tripo_v2.blend chair_tripo_v5.blend

(7z archives: extract first.) Headrest and mechanism still come from chair_m8c_cleaned.blend; the gas lift is rebuilt.
