#
# WRITE HERE HOW TO RUN ROLLOUT WITH --ckpt which file etc.
#

import torch, os, glob, argparse, yaml

from datetime import datetime
from torch.utils.data import DataLoader

from dataloader import PDEDatasetLoader_Multi
from model_cno import CNO2d
from model_fno import FNO2d
from utils_images import save_temperature_plot
from utils_train import mse_evalute_avg, relative_l2_percent

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Rollout of trained model checkpoint on the test set."
    )
    p.add_argument("--ckpt", required=True, help="Path to checkpoint directory")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    p.add_argument("--steps", type=int, default=1, help="Number of rollout steps")
    p.add_argument("--yaml", default="configs/default.yaml")
    p.add_argument("--idx", type=int, default=0, help="Which sequence index to use from the test set (default: 0)")
    return p.parse_args()

# -----------------------------
# Checkpoint helpers
# -----------------------------
def load_weights_into_model(model: torch.nn.Module, ckpt_file: str, device: torch.device):
    ckpt = torch.load(ckpt_file, map_location=device)
    sd = None
    if isinstance(ckpt, dict):
        if "state_dict" in ckpt:       # PyTorch Lightning
            sd = ckpt["state_dict"]
        elif "model_state" in ckpt:    # your custom format
            sd = ckpt["model_state"]
    if sd is None:
        raise RuntimeError(f"Could not find weights in {ckpt_file} (looked for 'state_dict' or 'model_state').")

    # Strip common prefixes
    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[6:]
        elif k.startswith("net."):
            k = k[4:]
        elif k.startswith("module."):
            k = k[7:]
        new_sd[k] = v

    missing, unexpected = model.load_state_dict(new_sd, strict=False)
    if missing:
        print("[WARN] Missing keys:", missing)
    if unexpected:
        print("[WARN] Unexpected keys:", unexpected)

    model.to(device).eval()
    print("[INFO] Weights loaded. Params:",
          sum(p.numel() for p in model.parameters()),
          "Trainable:", sum(p.numel() for p in model.parameters() if p.requires_grad))
    return model

def read_hparams(ckpt_dir: str) -> dict:
    hp_path = os.path.join(ckpt_dir, "logs", "version_0", "hparams.yaml")
    if not os.path.exists(hp_path):
        raise FileNotFoundError(f"hparams.yaml not found at {hp_path}")
    with open(hp_path, "r") as f:
        return yaml.safe_load(f)
    
def load_ckpt(ckpt_path, map_location="cpu"):
    # Expect: epoch, val_metric, model_state, optimizer_state, scheduler_state, config, norm, run_id
    state = torch.load(ckpt_path, map_location=map_location) # removed weight_only=False
    return state

def find_ckpt_file(ckpt_dir: str) -> str:
    """Prefer epoch*.ckpt, else last.ckpt."""
    candidates = glob.glob(os.path.join(ckpt_dir, "checkpoints", "epoch*.ckpt"))
    if not candidates:
        candidates = glob.glob(os.path.join(ckpt_dir, "checkpoints", "last.ckpt"))
    if not candidates:
        raise FileNotFoundError(f"No .ckpt found under {ckpt_dir}/checkpoints/")
    # newest by mtime
    candidates.sort(key=os.path.getmtime, reverse=True)
    print(f"[INFO] Using checkpoint file: {candidates[0]}")
    return candidates[0]

def build_model(hparams: dict):
    model_name = hparams.get("model_name", "CNO")
    model_cfg  = hparams["model_cfg"]
    if model_name == "CNO":
        return CNO2d(**model_cfg)
    elif model_name == "FNO":
        return FNO2d(**model_cfg)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

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
    args = parse_args()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    
    # Load checkpoint and model
    ckpt_dir  = f"checkpoints/{args.ckpt}"
    hparams = read_hparams(ckpt_dir)
    model   = build_model(hparams)
    ckpt_file = find_ckpt_file(ckpt_dir)
    model     = load_weights_into_model(model, ckpt_file, device)

    # Validation dataset
    norm = PDEDatasetLoader_Multi(which="train").get_norm()
    val_ds = PDEDatasetLoader_Multi(which="test", seq_len=args.steps, N=args.steps, return_sequence=True)
    val_ds.load_norm(norm)

    val_loader = DataLoader(
        val_ds,
        batch_size=1,           # one sequence per batch
        shuffle=False,          # keep file/index order
        num_workers=0           # keep strict order + HDF5 safety
    )

    # --- One sample ---
    inp, tgt_seq = next(iter(val_loader))    # inp: (1, 1+3N, H, W)
    inp = inp.to(device, dtype=torch.float32)
    tgt_seq = tgt_seq.to(device, dtype=torch.float32)
    B, C_all, H, W = inp.shape
    T_gt = tgt_seq.size(1)
    assert B == 1, f"Use batch_size=1, got {B}"

    # --- Decode N from channels and cross-check with args.steps ---
    assert (C_all - 1) % 3 == 0, f"Channels don't fit 1+3N pattern: C_all={C_all}"
    N = (C_all - 1) // 3
    if N < args.steps:
        print(f"[WARN] dataset N={N} < steps={args.steps}; reducing steps to {N}")
    steps = min(args.steps, N, T_gt)

    print(f"[INFO] Channels => temp:1, power:{N}, shift:{2*N}  -> total {1+3*N}")
    print(f"[INFO] Rollout steps: {steps}")

    # --- Initial state (t=0) is temp channel 0 ---
    temp_t = inp[:, 0:1, ...]            # (1,1,H,W)

    preds, ground_truth, mse_mask = [], [], []
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
            expected_in = hparams.get("model_cfg", {}).get("in_dim", 4)
            assert model_in.size(1) == expected_in
            
            # if tgt_seq.dim() == 3:                 # (B,H,W)
            #     tgt_seq = tgt_seq.unsqueeze(1)         # -> (B,1,H,W)
            # y_gt = tgt.cpu()  # (1,1,H,W)
            # ground_truth.append(y_gt)

            # print("tgt_seq shape:", tgt_seq.shape)
            gt_t   = tgt_seq[:, t, :, :]                 # (1,1,H,W)
            # print("gt_t shape:", gt_t.shape)
            if gt_t.dim() == 3:                 # (B,H,W)
                gt_t = gt_t.unsqueeze(1)         # -> (B,1,H,W)
            y_pred = model(model_in)                        # (1,1,H,W)
            # print("y_pred shape:", y_pred.shape)
            
            ground_truth.append(gt_t.detach().cpu())
            preds.append(y_pred.detach().cpu())
            temp_t = y_pred                        # feedback

            mse_mask.append((y_pred - gt_t).pow(2).detach().cpu())
            # max_mse = mse_mask[-1].max().item()

            # save
            save_temperature_plot(
                preds[-1][0, 0],
                path=f"results_val/{timestamp}/results_pred",
                name_prefix=f"seq_prediction_step{t}"
            )
            save_temperature_plot(
                ground_truth[-1][0, 0],
                path=f"results_val/{timestamp}/results_gt",
                name_prefix=f"seq_ground_truth_step{t}"
            )
            save_temperature_plot(
                mse_mask[-1][0, 0],
                path=f"results_val/{timestamp}/results_mse",
                name_prefix=f"seq_mse_step{t}",
                label="MSE",
                scale_fix=True,
                max_val=1.0
            )

    print(f"Average MSE: {mse_evalute_avg(preds, ground_truth):.4e}")
    print(f"Average Rel-L2: {relative_l2_percent(preds, ground_truth):.4f}%")
    print("\nRollout complete!")
    print(f"Total steps predicted: {len(preds)}")


if __name__ == "__main__":
    rollout()