import torch
import argparse
import yaml
import os

from datetime import datetime
from torch.utils.data import DataLoader

from dataloader import PDEDatasetLoader_Multi
from model_cno import CNO2d
from utils import save_temperature_plot

def parse_args_fno():
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
    # p.add_argument("--save-images", type=int, default=2, help="How many (pred, gt) pairs to save (0 = none).")
    p.add_argument("--yaml", default="configs/default.yaml", help="(Optional) Fallback YAML to read dataset paths.")
    return p.parse_args()

# -----------------------------
# Args (CNO)
# -----------------------------
def parse_args_cno():
    p = argparse.ArgumentParser(
        description="Roll out a trained CNO checkpoint on the test set."
    )
    p.add_argument("--ckpt", required=True, help="Path to checkpoint .pt")
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--steps", type=int, default=10, help="Number of rollout steps")

    # Model overrides (fallback to ckpt config if omitted)
    p.add_argument("--in_dim", type=int, default=None)
    p.add_argument("--out_dim", type=int, default=None)
    p.add_argument("--size", type=int, default=None)
    p.add_argument("--N_layers", type=int, default=None)
    p.add_argument("--N_res", type=int, default=None)
    p.add_argument("--N_res_neck", type=int, default=None)
    p.add_argument("--channel_multiplier", type=int, default=None)

    # Fallback YAML to locate datasets if ckpt lacks config
    p.add_argument("--yaml", default="configs/default.yaml")
    return p.parse_args()

# -----------------------------
# Checkpoint helpers
# -----------------------------
def load_ckpt(ckpt_path, map_location="cpu"):
    # Expect: epoch, val_metric, model_state, optimizer_state, scheduler_state, config, norm, run_id
    state = torch.load(ckpt_path, map_location=map_location, weights_only=False)
    return state

def build_model_from_config_cno(cfg, args):
    model_cfg = (cfg or {}).get("model", {})

    def choose(key, default_from_cfg, override):
        return override if override is not None else int(model_cfg.get(key, default_from_cfg))

    in_dim             = choose("in_dim", 4,  args.in_dim)
    out_dim            = choose("out_dim", 1, args.out_dim)
    size               = choose("size", 44, args.size)
    N_layers           = choose("N_layers", 5, args.N_layers)
    N_res              = choose("N_res", 4, args.N_res)
    N_res_neck         = choose("N_res_neck", 4, args.N_res_neck)
    channel_multiplier = choose("channel_multiplier", 16, args.channel_multiplier)

    model = CNO2d(
        in_dim=in_dim,
        out_dim=out_dim,
        size=size,
        N_layers=N_layers,
        N_res=N_res,
        N_res_neck=N_res_neck,
        channel_multiplier=channel_multiplier,
    )
    return model

def get_initial_pred(model, val_loader, device):
    with torch.no_grad():
        for (x, y) in val_loader:
            x = x.to(device, non_blocking=False)
            y = y.to(device, non_blocking=False)
            if y.ndim == 3:  # (B,H,W) -> (B,1,H,W)
                y = y.unsqueeze(1)

            y_pred = model(x)
    return y_pred

# -----------------------------
# Rollout
# -----------------------------
@torch.no_grad()
def rollout():
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    args = parse_args_cno()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # Load checkpoint (weights + config + norms)
    state = load_ckpt(args.ckpt, map_location=device)
    cfg   = state.get("config", None)
    
    # Build model and load weights
    model = build_model_from_config_cno(cfg, args).to(device)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()

    # Load YAML (only to get dataset filenames if needed)
    if cfg is None:
        with open(args.yaml, "r") as f:
            cfg = yaml.safe_load(f)

    # Validation dataset
    norm = PDEDatasetLoader_Multi(which="train").get_norm()
    val_ds = PDEDatasetLoader_Multi(which="test", seq_len=args.steps, N=args.steps)
    val_ds.load_norm(norm)

    val_loader = DataLoader(
        val_ds,
        batch_size=1,           # one sequence per batch
        shuffle=False,          # keep file/index order
        num_workers=0,          # keep strict order + HDF5 safety
        pin_memory=(device.type=="cuda"),
    )

    # --- One sample ---
    inp, tgt = next(iter(val_loader))    # inp: (1, 1+3N, H, W)
    inp = inp.to(device, dtype=torch.float32)
    B, C_all, H, W = inp.shape
    assert B == 1, f"Use batch_size=1, got {B}"

    # --- Decode N from channels and cross-check with args.steps ---
    assert (C_all - 1) % 3 == 0, f"Channels don't fit 1+3N pattern: C_all={C_all}"
    N = (C_all - 1) // 3
    if N < args.steps:
        print(f"[WARN] dataset N={N} < steps={args.steps}; reducing steps to {N}")
    steps = min(args.steps, N)

    print(f"[INFO] Channels => temp:1, power:{N}, shift:{2*N}  -> total {1+3*N}")
    print(f"[INFO] Rollout steps: {steps}")

    # --- Initial state (t=0) is temp channel 0 ---
    temp_t = inp[:, 0:1, ...]            # (1,1,H,W)

    preds = []
    model.eval()
    with torch.no_grad():
        for t in range(steps):
            # channel layout:
            # power_t index:      1 + t
            # shift_x_t index:    1 + N + 2*t
            # shift_y_t index:    1 + N + 2*t + 1
            power_t   = inp[:, 1 + t : 1 + t + 1, ...]              # (1,1,H,W)
            shift_x_t = inp[:, 1 + N + 2*t : 1 + N + 2*t + 1, ...]  # (1,1,H,W)
            shift_y_t = inp[:, 1 + N + 2*t + 1 : 1 + N + 2*t + 2, ...]  # (1,1,H,W)

            exog_t = torch.cat([power_t, shift_x_t, shift_y_t], dim=1)   # (1,3,H,W)
            model_in = torch.cat([temp_t, exog_t], dim=1)                # (1,4,H,W)

            # sanity check vs model expectation
            # (replace 'conv_in' if your first layer is named differently)
            # expected_in = state.get("config", {}).get("model", {}).get("in_dim", 4)
            # assert model_in.size(1) == expected_in

            y_pred = model(model_in)         # (1,1,H,W)
            preds.append(y_pred.detach().cpu())

            # feedback
            temp_t = y_pred

            # save
            save_temperature_plot(
                y_pred[0, 0],
                path=f"results_val/{timestamp}",
                name_prefix=f"seq_prediction_step{t}"
            )
    
    print("\nRollout complete!")
    print(f"Total steps predicted: {len(preds)}")


if __name__ == "__main__":
    rollout()