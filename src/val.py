#!/usr/bin/env python3
import os
import argparse
import torch
import yaml

from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss

# Your project imports
from dataloader import PDEDatasetLoader_Multi
from model_fno import FNO2d, PadCropFNO
from utils import save_temperature_plot  # assumes the shape-robust version

def parse_args():
    p = argparse.ArgumentParser(
        description="Quick validation: load checkpoint, run on validation set, no training."
    )
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt (e.g., checkpoints/<run_id>/best.pt)")
    p.add_argument("--batch-size", type=int, default=16, help="Validation batch size")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    p.add_argument("--pad", type=int, default=None, help="Pad size for PadCropFNO (overrides ckpt/config).")
    p.add_argument("--in-channels", type=int, default=None, help="Override in_channels if not in ckpt config.")
    p.add_argument("--modes1", type=int, default=None, help="Override modes1 if not in ckpt config.")
    p.add_argument("--modes2", type=int, default=None, help="Override modes2 if not in ckpt config.")
    p.add_argument("--width", type=int, default=None, help="Override width if not in ckpt config.")
    p.add_argument("--save-images", type=int, default=2, help="How many (pred, gt) pairs to save (0 = none).")
    p.add_argument("--yaml", default="configs/default.yaml", help="(Optional) Fallback YAML to read dataset paths.")
    return p.parse_args()

def load_ckpt(ckpt_path, map_location="cpu"):
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    # Expected keys from your training checkpoint helper
    # 'epoch', 'val_metric', 'model_state', 'optimizer_state', 'scheduler_state', 'config', 'norm', 'run_id'
    return state

def build_model_from_config(cfg, args):
    # Pull defaults from checkpoint config if present
    model_cfg = (cfg or {}).get("model", {})
    # modes may be a scalar or [modes1, modes2]
    modes = model_cfg.get("modes", 24)
    if isinstance(modes, (list, tuple)) and len(modes) >= 2:
        m1_default, m2_default = modes[0], modes[1]
    else:
        m1_default = m2_default = int(modes)

    width_default = int(model_cfg.get("width", 16))
    in_default    = int(model_cfg.get("in_channels", 4))
    pad_default   = int(model_cfg.get("pad", 8))

    # CLI overrides take precedence
    m1 = args.modes1 if args.modes1 is not None else m1_default
    m2 = args.modes2 if args.modes2 is not None else m2_default
    width = args.width if args.width is not None else width_default
    in_channels = args.in_channels if args.in_channels is not None else in_default
    pad = args.pad if args.pad is not None else pad_default

    core = FNO2d(modes1=m1, modes2=m2, width=width, in_channels=in_channels)
    net  = PadCropFNO(core, pad=pad)
    return net

def main():
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint (weights + config + norms)
    state = load_ckpt(args.ckpt, map_location=device)
    cfg   = state.get("config", None)
    norm  = state.get("norm", None)
    run_id = state.get("run_id", "unknown")

    # Build model and load weights
    fno = build_model_from_config(cfg, args).to(device)
    fno.load_state_dict(state["model_state"], strict=True)
    fno.eval()

    # Load YAML (only to get dataset filenames if needed)
    if cfg is None:
        with open(args.yaml, "r") as f:
            cfg = yaml.safe_load(f)

    # Validation dataset
    norm = PDEDatasetLoader_Multi(which="train").get_norm()
    val_ds = PDEDatasetLoader_Multi(which="test")
    val_ds.load_norm(norm)

    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False)

    # Metrics
    total_mse = 0.0
    total_rel = 0.0
    n_batches = 0

    # Where to put images
    os.makedirs("results_val", exist_ok=True)
    saved_pairs = 0

    with torch.no_grad():
        for (x, y) in val_loader:
            x = x.to(device, non_blocking=False)
            y = y.to(device, non_blocking=False)
            if y.ndim == 3:  # (B,H,W) -> (B,1,H,W)
                y = y.unsqueeze(1)

            y_pred = fno(x)

            # Metrics
            batch_mse = mse_loss(y_pred, y).item()
            # Rel-L2%: ||y_pred - y|| / ||y||
            rel = (torch.norm(y_pred - y) / torch.norm(y)).item() * 100.0

            total_mse += batch_mse
            total_rel += rel
            n_batches += 1
            path = "results_val/"

            # Optionally save a few images
            if args.save_images > 0 and saved_pairs < args.save_images or args.save_images == -1 and saved_pairs < len(val_ds):
                # Save first item in the batch
                save_temperature_plot(y_pred[0, 0], path, name_prefix=f"val_pred_run{run_id}")
                save_temperature_plot(y[0, 0], path, name_prefix=f"val_gt_run{run_id}")
                saved_pairs += 1

    mean_mse = total_mse / max(1, n_batches)
    mean_rel = total_rel / max(1, n_batches)

    print("==============================================")
    print(f"Checkpoint:      {args.ckpt}")
    print(f"Run ID:          {run_id}")
    print(f"Device:          {device}")
    print(f"Batches:         {n_batches}")
    print(f"Mean MSE:        {mean_mse:.6e}")
    print(f"Mean Rel-L2 (%): {mean_rel:.3f}")
    if "val_metric" in state and state["val_metric"] is not None:
        print(f"(Stored val at ckpt time: {state['val_metric']:.3f} %)")
    print("Images saved to: ./results_val (if requested)")
    print("==============================================")

if __name__ == "__main__":
    main()
