import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl
import imageio.v2 as iio
import imageio_ffmpeg

from datetime import datetime

def _uniquify(filepath: str) -> str:
    """If filepath exists, append _1, _2, ... before the extension."""
    base, ext = os.path.splitext(filepath)
    i = 1
    candidate = filepath
    while os.path.exists(candidate):
        candidate = f"{base}_{i}{ext}"
        i += 1
    return candidate

def save_temperature_plot(temp_tensor, path="results", name_prefix="temp", epoch=None):
    """
    Save a 2D temperature field as a .png image in ./results folder.
    
    Args:
        temp_tensor (torch.Tensor or np.ndarray): shape (H, W) or (1, H, W)
        name_prefix (str): prefix for filename ("pred" or "true", etc.)
    """
    # --- Ensure numpy array ---
    if hasattr(temp_tensor, "detach"):
        temp_tensor = temp_tensor.detach().cpu().numpy()

    # If shape is (1, H, W), squeeze the channel
    if temp_tensor.ndim == 3 and temp_tensor.shape[0] == 1:
        temp_tensor = temp_tensor[0]

    # Make sure shape is (H, W)
    assert temp_tensor.ndim == 2, f"Expected (H, W), got {temp_tensor.shape}"

    # --- Create results folder if it doesn't exist ---
    os.makedirs(path, exist_ok=True)
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")

    if epoch is not None:
        os.makedirs(f"{path}/{epoch}", exist_ok=True)
        filename = f"{path}/{epoch}/{name_prefix}_{timestamp}.png"
    else:
        filename = f"{path}/{name_prefix}_{timestamp}.png"

    filename = _uniquify(filename)


    # --- Plot ---
    plt.figure(figsize=(5, 4))
    im = plt.imshow(temp_tensor, cmap="inferno", origin="lower")
    plt.colorbar(im, label="Temperature (normalized)")
    plt.title(f"{name_prefix} @ {timestamp}")
    plt.tight_layout()
    plt.savefig(filename, dpi=150)
    plt.close()

    print(f"[INFO] Saved temperature plot to {filename}")


def _to_THW(seq: torch.Tensor) -> torch.Tensor:
    """
    Akzeptiert (B,T,1,H,W) | (T,1,H,W) | (T,H,W) und gibt (T,H,W) (float32, cpu) zurück.
    """
    x = torch.as_tensor(seq).detach().cpu()
    if x.ndim == 5:   # (B,T,1,H,W)
        x = x[0, :, 0]
    elif x.ndim == 4: # (T,1,H,W)
        x = x[:, 0]
    elif x.ndim == 3: # (T,H,W)
        pass
    else:
        raise ValueError(f"Unexpected shape: {tuple(x.shape)}")
    return x.float()

def animate_side_by_side_mp4(pred_seq, gt_seq, out_path, fps=8, cmap="inferno", vmin=None, vmax=None):
    """
    pred_seq, gt_seq: gleiche Länge/Größe, jeweils (B,T,1,H,W) | (T,1,H,W) | (T,H,W)
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    p = _to_THW(pred_seq)
    g = _to_THW(gt_seq)
    assert p.shape == g.shape, f"Shape mismatch: {p.shape} vs {g.shape}"
    T, H, W = p.shape

    if vmin is None: vmin = float(min(p.min(), g.min()))
    if vmax is None: vmax = float(max(p.max(), g.max()))

    norm   = mpl.colors.Normalize(vmin=vmin, vmax=vmax, clip=True)
    cmap_f = mpl.cm.get_cmap(cmap)

    with iio.get_writer(out_path, format="FFMPEG", fps=fps, codec="libx264", ffmpeg_params=["-pix_fmt", "yuv420p"]) as writer:
        for t in range(T):
            rgb_p = (cmap_f(norm(p[t].numpy()))[..., :3] * 255).astype(np.uint8)
            rgb_g = (cmap_f(norm(g[t].numpy()))[..., :3] * 255).astype(np.uint8)
            frame = np.concatenate([rgb_p, rgb_g], axis=1)  # (H, 2W, 3)
            frame = np.ascontiguousarray(frame)
            writer.append_data(frame)
    return out_path

def save_temperature_plot_sequence(temp_tensor, idx, path='results', name_prefix='temp', epoch=None):

    y = temp_tensor.detach().cpu()
    if y.ndim == 4:  # (B, 1, H, W) -> nur ein Frame
        frames = [y[0, 0]]
    elif y.ndim == 5:  # (B, T, 1, H, W)
        T = y.size(1)
        frames = [y[0, t, 0] for t in range(T)]
    else:
        raise ValueError(f"Unexpected target shape: {tuple(y.shape)}")
    os.makedirs(path, exist_ok=True)

    for t, frame in enumerate(frames):
        save_temperature_plot(frame, path=path,
                              name_prefix=f"{name_prefix}_id{idx}_t{t:03d}",
                              epoch=epoch)
        
import glob, cv2
from natsort import natsorted  # pip install natsort
from PIL import Image

# pip install imageio imageio-ffmpeg natsort pillow
import os, glob
from datetime import datetime
from natsort import natsorted
from PIL import Image
import numpy as np
import imageio

def generate_video_from_pngs(src_dir, save_root, fps=8):
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

if __name__ == "__main__":
    generate_video_from_pngs("results_val/07102025_184051", "results_video", fps=8)