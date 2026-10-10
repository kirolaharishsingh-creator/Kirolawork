---
name: product-video-workflow
description: Make the 9-video product package (26 s hero film, 7 three-second movement/USP loops, 1 assembly/exploded video) from client photos with Kling and local post-processing. Use for any new USP clip, hero shot, assembly video or a new product.
---

Read `WORKFLOW.md` at the repo root and follow it step by step. Non-negotiable:

1. Never spend credits (Kling, Nano Banana, Seedance…) until the client says "generate" for that request; quote the cost first; never retry on your own.
2. Measure the reference animation (edge/template tracking, not optical flow) before choosing frames.
3. Start/end frames: sharpest real photos (Laplacian ≥ 100), aligned on static parts, converted to the studio backdrop (RGB ≈ 202,198,194) with `tools/chair3d/photo_to_studio.py`, shown to the client before any paid run.
4. Kling 3.0 Omni Frames Pro 3 s (3.75 credits), locked-camera prompt naming static parts and the rigid moving unit.
5. After the client returns the clip: check drift/rigidity, fix in post (stabilise, rigid re-composite, backdrop match), ease into a move-hold-return loop, strip audio, commit, push, send the raw link.
