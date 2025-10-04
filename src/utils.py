import h5py

import os
import random
import numpy as np
import matplotlib.pyplot as plt
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

def save_temperature_plot(temp_tensor, path, name_prefix="temp", epoch=None):
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
    os.makedirs("results", exist_ok=True)
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


def dump_h5_structure(path, out_path="h5_structure.txt"):
    """
    Walks through an HDF5 file and writes a tree-style overview
    (groups, datasets, shapes, dtypes, scalar values) to a text file.

    Args:
        path (str): path to your .h5 file
        out_path (str): path where the txt overview will be saved
    """
    lines = []

    def _collect(name, obj):
        indent = "  " * (name.count('/') - 1)
        if isinstance(obj, h5py.Group):
            lines.append(f"{indent}📁 Group: {name}/")
        elif isinstance(obj, h5py.Dataset):
            shape = obj.shape
            dtype = obj.dtype
            if shape == ():  # scalar dataset
                val = obj[()]
                lines.append(f"{indent}📄 Dataset: {name}  shape={shape}, dtype={dtype}, value={val}")
            else:
                lines.append(f"{indent}📄 Dataset: {name}  shape={shape}, dtype={dtype}")

    with h5py.File(path, "r") as f:
        lines.append(f"HDF5 Structure of: {path}\n")
        f.visititems(_collect)
        lines.append("\n✅ Done.\n")

    with open(out_path, "w") as f_out:
        f_out.write("\n".join(lines))

    print(f"✅ HDF5 structure written to: {out_path}")

# Example usage:
# inspect_h5_file("data/pattern_paths_training_44_44.h5")

if __name__ == "__main__":
    dump_h5_structure("data/pattern_paths_training_44_44.h5")