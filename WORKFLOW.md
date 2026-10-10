# Product video workflow (9-video package)

How we make the full video set for one product, built from the ergonomic-chair project. Follow it step by step for the next product. Everything below was learned the hard way; the "Pitfalls" lines are real mistakes we already paid for.

## The package

| # | Video | Length | Method |
|---|---|---|---|
| 1 | Hero / anatomy film | 26 s | 8–10 short shots (≈3 s each) cut together, shared backdrop, light-sweep transitions |
| 2–8 | 7 movement (USP) videos | 3 s loop each | One mechanism per clip, locked camera, move → hold → return |
| 9 | Product assembly (exploded) | 6–8 s loop | 3D model or Kling: parts fly apart, hold, reassemble |

USP list for the chair (rename per product): 1 rotation 360°, 2 mesh breathability, 3 lumbar movement, 4 seat slider, 5 multilock recline, 6 armrest 4D (vertical / horizontal / diagonal), 7 headrest.

## Ground rules (never break)

1. **Credits:** never run a paid generation (Kling, Nano Banana, Seedance, etc.) until the client says **"generate"** for that exact request. Quote model, mode, length and credit cost first.
2. **No automatic retries.** One run per approval. If it fails, report and ask.
3. Show start/end frames (free) before asking for a paid run.
4. Every video sits on the **same studio backdrop**: warm light grey, RGB ≈ **202, 198, 194** (`tools/chair3d/comp.py: studio_bg`).
5. Commit and push each deliverable as soon as it is approved.

## Folder layout

```
chair_photos/NN_topic/        client photos sorted by use (README lists numbers)
usp/<usp>_stills/             start/end frames per USP
usp/uspN_<name>_final_1080p.mp4   approved finals (4K master if from photo)
kling_final/, anatomy_edit/   hero-film shots and cuts
exploded/                     assembly video work
tools/, tools/chair3d/        all scripts (reuse them)
```

## Step 0 — Intake (once per product)

1. Sort client photos into `chair_photos/NN_topic/` folders (front, side, rear, reclined, each USP close-up) and write the README table.
2. Score sharpness of every candidate: `cv2.Laplacian(gray at 1600 px wide).var()`. **Use ≥ 100. Under ~60 = blurry → Kling output will be blurry too.**
3. For each USP collect the client's reference animation (GIF/MP4). Ask for it if missing.

## Step 1 — Read the reference before anything else

Measure what moves in the reference; don't trust a glance.

- Track parts with **edges / template matching** (`cv2.matchTemplate`), not optical flow — flow fails on black, textureless parts.
- Write down: which parts move together, which stay still, travel in px, timing (move / hold / return).
- **Pitfall (seat slider):** we first assumed only the cushion moved; in fact seat + shell + both armrests move as one unit. Measuring would have saved two paid runs.
- **Pitfall (recline):** seat, armrests and backrest tilt together; base never moves.

## Step 2 — Start and end frames (free)

Pick the sharpest real photo with the right angle as the **start frame**. For the end frame, in order of preference:

1. **Real photo of the end position** from the same camera → align it to the start on the static parts (`tools/chair3d/armrest_align_pairs.py`: ORB + RANSAC on backrest/seat with the moving zone masked out; check the overlay).
2. **Digitally move the part in the start photo** when the end photo is from another camera position (e.g. `armrest_vertical_end.py`: lift the pad rigidly, stretch the real post section so the taper stays, refill the wall). Pixel-identical everywhere else → no camera drift in Kling.
3. Nano Banana Pro edit (2 credits at 2K) only if 1–2 are impossible; ask first.

Then for every frame:
- **Put it on the studio backdrop**: `W=2880 python3 tools/chair3d/photo_to_studio.py in.jpg out.jpg` (`CHROME=0` for photos without chrome). **Pitfall (armrest):** we sent raw photo walls once — cool bluish grey + hanging string + floor seam. Always convert first.
- 16:9, 1920×1080 (or 3840×2160 for a 4K master).
- Show start/end side by side (and a toggle) to the client.

## Step 3 — Generate the movement (paid, after "generate")

- Model **Kling 3.0 Omni Frames** (`kling_o3_flf`), roles `start_image` + `end_image`, **Pro, 3 s, 16:9 = 3.75 credits** (std 3 cr = 720p; 4K 18 cr).
- Upload frames by committing them and importing the GitHub raw URL **with the commit hash** (`media_import_url`).
- If Higgsfield suggests the "IN THE DARK" preset, decline it (`declined_preset_id 24bae836-2c4a-48e0-89b6-49fcc0b21612`).
- Prompt template:
  > Locked tripod camera, absolutely no camera movement or zoom. **[static parts]** stay completely still. Only **[moving unit]** **[motion verb + direction/axis]**, one rigid unit, easing in and easing out. Photorealistic product video, light grey studio backdrop.
- Start-only (no end frame) works when the motion is simple and the prompt names the whole moving unit (USP 4 photo-23 run).
- Result downloads are blocked in the cloud session: give the client the link, they attach the MP4 back.

## Step 4 — Check the clip (free)

Measure, then decide:
- Static parts (backrest, base, wall) drift ≤ 2 px over the clip.
- Moving unit is **rigid**: after removing the travel, its outline matches the start frame. **Pitfall (seat slider):** Kling made the armrests swing/stretch 15–25 px.
- Find the usable segment (Kling often idles or wobbles first; take only the clean move).
- Sharpness similar to the input frames.

## Step 5 — Post (free)

| Problem | Fix (script) |
|---|---|
| Camera drift / push-in | Stabilise on static patches → affine (`clip_stabilize.py`), crop 3–4 % |
| Part deforms / not rigid | Move the start-frame part rigidly by the measured travel; use Kling only for the revealed background (`kling_seat_rigid_arm.py`) |
| Thin black part needs exact rigid motion | Pure 2D rig from the photo (`lumbar_swing_side.py`: pad rotates on its pivot, plate rebuilt behind) |
| Background colour off | Re-key wall → `studio_bg`, or gain-match to RGB 202,198,194 |
| Loop | Ease (smootherstep) forward 1.2 s → hold → back 1.0 s, start = end (`kling_seat_loop.py`, `clip_seat_loop.py`) |
| Audio | Always strip (`-an`); Kling adds sound by default |

Encode: H.264, CRF 14, preset slow, yuv420p, `+faststart`, 30 fps, no audio. Light unsharp (1.5/−0.5, σ 1.2) only when the source is soft.

## Hero film (26 s)

1. One keyframe per feature (swivel, headrest, mesh, lumbar, side, armrest, mechanism, hero rear) from real photos on the studio backdrop (`tools/build_kf*.py`, `crop169.py`).
2. Each shot = Kling start→end or start-only with a gentle camera move (push-in, orbit). Grade and lock exposure (`lock_exposure.py`, `match_bg.py`).
3. Cut each to ≈3 s, join with light-sweep transitions and shared backdrop gain (`tools/assemble_anatomy.py`; whip pans: `assemble_whip.py`, `join_whip.py`).
4. Check every cut for colour jumps; total 26 s at 30 fps.

## Assembly / exploded video

1. 3D model (Tripo or client file) → clean in Blender (`tools/chair3d/tripo_fix.py`, `fix_m*.py`, part map `parts.py`, `partsheet.py`).
2. Option A — pure 3D: `anim3.py` (explode → hold → reassemble, turntable, `DURATION`, `EXPLODE`, `LENS`), composite with `comp.py` on `studio_bg`.
3. Option B — Kling: start = real photo on backdrop, end = 3D exploded render matched to the same angle; Kling interpolates; add the thin light outline with `ray_overlay.py`.
4. Loop: forward + reverse, seamless.

## Delivery checklist (per video)

- [ ] Backdrop RGB ≈ 202,198,194, no strings/seams/patches
- [ ] Static parts locked, moving part rigid, motion matches the reference
- [ ] Loop seamless (first frame = last frame) for USP/assembly clips
- [ ] 1080p final (+4K master where the source allows), no audio
- [ ] Committed, pushed, raw download link sent to client
