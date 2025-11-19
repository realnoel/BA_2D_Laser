import numpy as np
import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl


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

def save_temperature_plot(temp_tensor,
                          path="results", 
                          name_prefix="temp", 
                          epoch=None, 
                          label="Temperature (normalized)", 
                          scale_fix=False, 
                          max_val=None
                          ):
    """
    Save a 2D temperature field as a .png image in ./results folder.
    
    Args:
        temp_tensor (torch.Tensor or np.ndarray): shape (H, W) or (1, H, W)
        name_prefix (str): prefix for filename ("pred" or "true", etc.)
    """
    # --- Ensure numpy array ---
    if hasattr(temp_tensor, "detach"):
        temp_tensor = temp_tensor.detach().cpu().numpy()
    
    # if temp_tensor.ndim == 4 and temp_tensor.shape[0] == 1:
    #     temp_tensor = temp_tensor[0]  # (1, 1, H, W) -> (1, H, W)

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

    # --- Determine color scale if fixed ---
    vmin, vmax = None, None
    if scale_fix:
        vmin = 0.0
        vmax = max_val if max_val is not None else 1.0


    # --- Plot ---
    plt.figure(figsize=(5, 4))
    im = plt.imshow(temp_tensor, cmap="inferno", origin="lower", vmin=vmin, vmax=vmax)
    plt.colorbar(im, label=label)
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
        
def plot_error(error, path, filename="rel_l2.png", title="Error", y_axis="Error", x_axis="Step t"):
    """
    Plot relative L2 (%) vs. step t from a list like:
        rel_l2 = [(value, t), (value, t), ...]
    and save to path/filename.

    - Handles floats/ints/torch tensors for 'value'.
    - Sorts points by t before plotting.
    """
    # ensure output dir exists
    os.makedirs(path, exist_ok=True)
    out = os.path.join(path, filename)

    # normalize & sort
    pairs = []
    for v, t in error:
        try:
            # handle torch tensors
            if hasattr(v, "item") and callable(v.item):
                v = v.item()
        except Exception:
            pass
        # basic sanity: cast to float
        v = float(v)
        pairs.append((int(t), v))
    pairs.sort(key=lambda p: p[0])

    steps = [p[0] for p in pairs]
    values = [p[1] for p in pairs]

    # plot (single chart, default style/colors)
    plt.figure(figsize=(6, 4))
    plt.plot(steps, values, marker="o")
    plt.title(title)
    plt.xlabel(x_axis)
    plt.ylabel(y_axis)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    return out

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def plot_train_val_loss(csv_path: str, metric: str = "L1", out_dir: str = "results_val", start_epoch: int = 0):
    """
    Plot train and validation loss curves from a metrics CSV file (log-scaled y-axis).

    Parameters
    ----------
    csv_path : str
        Path to the metrics.csv file.
    metric : str
        Metric type: 'L1', 'L2', or 'MSE' (case-insensitive).
    out_dir : str
        Directory where the output plot will be saved.
    start_epoch : int, optional
        Ignore all epochs before this value (default = 0).
    """

    # --- Select columns based on metric ---
    metric = metric.upper()
    if metric == "L1":
        train_col = "train_l1_percent"
        val_col = "val_rel_l1_percent"
        ylabel = "L1 (%)"
    elif metric == "L2":
        train_col = "train_rel_l2_percent"
        val_col = "val_rel_l2_percent"
        ylabel = "Relative L2 (%)"
    elif metric == "MSE":
        train_col = "train_mse"
        val_col = "val_mse"
        ylabel = "MSE"
    else:
        raise ValueError("metric must be one of ['L1', 'L2', 'MSE']")

    # --- Read CSV robustly ---
    df = pd.read_csv(csv_path)
    df = df.dropna(how="all")
    df["epoch"] = pd.to_numeric(df["epoch"], errors="coerce")
    df[train_col] = pd.to_numeric(df[train_col], errors="coerce")
    df[val_col] = pd.to_numeric(df[val_col], errors="coerce")

    # --- Filter and group ---
    df = df.dropna(subset=["epoch"])
    df = df.sort_values("epoch")
    grouped = df.groupby("epoch")[[train_col, val_col]].mean()

    # --- Slice epochs ---
    grouped = grouped[grouped.index >= start_epoch]

    # --- Prepare data ---
    epochs = grouped.index.values
    train_vals = np.clip(grouped[train_col].values, 1e-12, None)
    val_vals = np.clip(grouped[val_col].values, 1e-12, None)

    # --- Plot ---
    plt.figure(figsize=(8, 5))
    plt.plot(epochs, train_vals, label="Train", linewidth=2)
    plt.plot(epochs, val_vals, label="Validation", linewidth=2)
    plt.xlabel("Epoch")
    plt.ylabel(ylabel)
    plt.yscale("log")
    plt.title(f"Training vs Validation {metric} Loss (log scale)\nStarting from epoch {start_epoch}")
    plt.legend()
    plt.grid(True, linestyle="--", alpha=0.5, which="both")

    # --- Save ---
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, f"loss_curve_{metric}_from{start_epoch}.png")
    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    print(f"✅ Saved log-scale plot (starting from epoch {start_epoch}) to {out_path}")



if __name__ == "__main__":
    plot_train_val_loss("checkpoints/F2/logs/version_0/metrics.csv", metric="L1", start_epoch=250, out_dir="results_val_F2")

    