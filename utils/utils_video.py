#!/usr/bin/env python3
import os
import glob
import re
from datetime import datetime

import numpy as np
from PIL import Image
from natsort import natsorted
import imageio
import imageio_ffmpeg

def _load_sorted_pngs(folder):
    files = natsorted(glob.glob(os.path.join(folder, "*.png")))
    assert files, f"No PNGs found in {folder}"
    return files

def _resize_to_common_height(imgs):
    """Resize PIL images to the same height (max of the three), preserving aspect ratio."""
    heights = [im.height for im in imgs]
    target_h = max(heights)
    out = []
    for im in imgs:
        if im.height != target_h:
            new_w = int(round(im.width * (target_h / im.height)))
            im = im.resize((new_w, target_h), resample=Image.BICUBIC)
        out.append(im)
    return out

def _hstack_pils(imgs):
    """Horizontally stack equally-tall PIL images."""
    widths = [im.width for im in imgs]
    height = imgs[0].height
    canvas = Image.new("RGB", (sum(widths), height))
    x = 0
    for im in imgs:
        canvas.paste(im, (x, 0))
        x += im.width
    return canvas

def generate_triptych_video(
    gt_dir,
    pred_dir,
    mse_dir,
    save_root="results_video",
    fps=8,
    codec="libx264",
    pixel_format="yuv420p"
):
    gt_paths   = _load_sorted_pngs(gt_dir)
    pred_paths = _load_sorted_pngs(pred_dir)
    mse_paths  = _load_sorted_pngs(mse_dir)

    # Use the minimum available length across all three sources
    n = min(len(gt_paths), len(pred_paths), len(mse_paths))
    assert n > 0, "No overlapping frames to combine."

    os.makedirs(save_root, exist_ok=True)
    ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    out_path = os.path.join(save_root, f"{fps}fps_{ts}.mp4")

    with imageio.get_writer(out_path, fps=fps, codec=codec, ffmpeg_params=["-pix_fmt", pixel_format]) as writer:
        for i in range(n):
            gt   = Image.open(gt_paths[i]).convert("RGB")
            pred = Image.open(pred_paths[i]).convert("RGB")
            mse  = Image.open(mse_paths[i]).convert("RGB")

            # Make heights match
            gt, pred, mse = _resize_to_common_height([gt, pred, mse])

            # Concatenate horizontally
            frame = _hstack_pils([gt, pred, mse])

            writer.append_data(np.array(frame))

    print(f"✅ Saved video: {out_path}")

def generate_video_from_pngs_single_image(src_dir, save_root, fps=8):
    frames = natsorted(glob.glob(os.path.join(src_dir, "*.png")))
    assert frames, f"No PNGs found in {src_dir}"

    ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    out_dir = os.path.join(save_root, ts)
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"out_{fps}fps.mp4")


    with imageio.get_writer(out_path, fps=fps, codec='libx264', pixelformat='yuv420p') as writer:
        for p in frames:
            rgb = Image.open(p).convert("RGB")
            writer.append_data(np.array(rgb))  # ✅ <- note: append_data()

    print(f"✅ Saved to {out_path}")

def generate_video_from_ddpm_animation(ddpm_dir, save_root="results_video", fps=1):
    """
    Nimmt PNGs aus ddpm_dir und erzeugt ein MP4 mit fps.
    Es wird pro step_XXX nur ein Bild genommen (das erste nach Sortierung).
    """
    files = natsorted(glob.glob(os.path.join(ddpm_dir, "*.png")))
    assert files, f"No PNGs found in {ddpm_dir}"

    step_re = re.compile(r"(step_(\d+))")  # matches step_000 and captures number

    # group by step number
    per_step = {}
    for p in files:
        m = step_re.search(os.path.basename(p))
        if not m:
            continue
        step_num = int(m.group(2))
        per_step.setdefault(step_num, []).append(p)

    assert per_step, f"No files with 'step_XXX' pattern found in {ddpm_dir}"

    # pick exactly one file per step (first in sorted list)
    chosen = []
    for step_num in sorted(per_step.keys()):
        candidates = natsorted(per_step[step_num])
        chosen.append(candidates[0])

    os.makedirs(save_root, exist_ok=True)
    ts = datetime.now().strftime("%d%m%Y_%H%M%S")
    out_path = os.path.join(save_root, f"ddpm_{fps}fps_{ts}.mp4")

    with imageio.get_writer(out_path, fps=fps, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"]) as writer:
        for p in chosen:
            img = Image.open(p).convert("RGB")
            writer.append_data(np.array(img))

    print(f"✅ Saved video: {out_path}")
    print(f"Frames used: {len(chosen)} (one per step)")

# if __name__ == "__main__":
#     generate_video_from_pngs_single_image("results_val/results_gt_08102025_125031", "results_video", fps=8)

if __name__ == "__main__":
    dir_name = "results_thesis/20251218_002920_K75_phys_165_3" # Enter the name of the directory in results_val
    generate_triptych_video(
        f"{dir_name}/results_gt",
        f"{dir_name}/results_pred",
        f"{dir_name}/error_map",
        save_root="results_video",
        fps=12
    )

# if __name__ == "__main__":
#     ddpm_dir = "results_targets/PHYS_TARGET/seq_0"  # oder z.B. checkpoints/.../ddpm_animation/epoch_0000
#     generate_video_from_ddpm_animation(ddpm_dir, save_root="results_targets", fps=12)