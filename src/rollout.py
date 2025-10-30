#
# WRITE HERE HOW TO RUN ROLLOUT WITH --ckpt which file etc.
#

#
# Was soll das Programm können:
# - Verschiedene Sequenz IDX auswählen können
# - Sequenlänge bestimmen, mit --steps -1 für volle Länge
#

import torch, os, glob, argparse, yaml

from datetime import datetime
from torch.utils.data import DataLoader

from dataloader import PDEDatasetLoader_Multi
from model_cno import CNO2d
from model_fno import FNO2d
from utils_images import save_temperature_plot, plot_error
from utils_train import mse_evaluate_avg, relative_l2_percent, relative_l1_percent

# -----------------------------
# Argument parser
# -----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Rollout of trained model checkpoint on the test set."
    )
    p.add_argument("--ckpt", required=True, help="Path to checkpoint directory")
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"], help="Device selection")
    p.add_argument("--steps", type=int, default=-1, help="Number of rollout steps")
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
    N = hparams.get("training", {}).get("N", 1)
    print(args.steps, N)
    assert args.steps > 0, "Rollout steps must be positive"
    norm = PDEDatasetLoader_Multi(which="train").get_norm()
    val_ds = PDEDatasetLoader_Multi(which="test", 
                                    N=N,
                                    K=args.steps
    )
    val_ds.load_norm(norm)

    norm = {
        "min_p": norm[0],
        "max_p": norm[1],
        "min_shift": norm[2],
        "max_shift": norm[3],
        "min_model": norm[4],
        "max_model": norm[5]
    }

    # val_loader = DataLoader(
    #     val_ds,
    #     batch_size=1,           # one sequence per batch     # SOLLTE DIESER WERT NICHT N SEIN???? DAMIT INP = OUT = N IST????????
    #     shuffle=False,          # keep file/index order
    #     num_workers=0           # keep strict order + HDF5 safety
    # )

    # --- One specific sequence (use --idx) ---
    seq_inp, seq_tgt = val_ds[args.idx]                         # seq_inp: (K, 4N+3, H, W), seq_tgt: (K, N, H, W)
    inp = seq_inp.unsqueeze(0).to(device, dtype=torch.float32)  # inp: (1, K, 4N+3, H, W)  per time-step: [N temp | N power | 2N shift]
    tgt = seq_tgt.unsqueeze(0).to(device, dtype=torch.float32)  # tgt: (1, K, N, H, W)  target for each step

    # Sanity check
    print(f"Input shape: {inp.shape}, Target sequence shape: {tgt.shape}")
    B, T, C, H, W = tgt.shape
    assert C == N, f"Expected {N} channels in target, got {C}"
    B, T, C, H, W = inp.shape
    assert C == 4*N+3, f"Expected {4*N+3} channels per step, got {C}"
    assert B == 1, f"Use batch_size=1, got {B}"
    
    # decide rollout length
    steps = T if args.steps < 0 else min(args.steps, T)

    print(f"[INFO] Channels => temp:{N}, power:{N}, shift:{2*N}  -> total {4*N}")
    print(f"[INFO] Rollout steps: {steps}")

    # --- Initial state (t=0 .. t=N): temp-channel aus inp nehmen ---
    temp_stack = inp[:, 0, 0:N, ...].contiguous()               # (B, N, H, W)

    preds, ground_truth, mse_mask, rel_l2, rel_l1, mse_error = [], [], [], [], [], []
    model.eval()
    with torch.no_grad():
        for t in range(steps):
            # Prepare model input for this step
            exog_t = inp[:, t, N:4*N+3, ...].contiguous()       # (B, 3N+3, H, W)

            # Combine temp_stack + exog_t
            model_in = torch.cat([temp_stack, exog_t], dim=1)   # (B, 4N+3, H, W)

            # sanity check
            expected_in = hparams.get("model_cfg", {}).get("in_dim", 4*N+3)
            assert model_in.size(1) == expected_in, f"in_dim mismatch: {model_in.size(1)} vs {expected_in}"

            # Ground truth at this step
            gt_t = tgt[:, t:t+N, ...]                           # (B, N, H, W)

            # ---- Shape-Normalisation -> (B,1,H,W) ----
            if gt_t.dim() == 3:                  # (B,H,W)
                gt_t = gt_t.unsqueeze(1)         # -> (B,1,H,W) 
            elif gt_t.dim() == 5:                # (B,?,1,H,W)
                gt_t = gt_t[:, 0, ...]           # 
                if gt_t.dim() == 3:              # (B,H,W)
                    gt_t = gt_t.unsqueeze(1)     # -> (B,1,H,W)
            elif gt_t.dim() == 2:                # (H,W)
                gt_t = gt_t.unsqueeze(0).unsqueeze(0)  # -> (1,1,H,W)

            # Prediction
            y_pred = model(model_in)    # (B,N,H,W)
            # y_pred = (y_pred - norm["min_model"]) / (norm["max_model"] - norm["min_model"]) # normalize
            # y_pred_vis = y_pred * (norm["max_model"] - norm["min_model"]) + norm["min_model"] # denormalize

            if y_pred.dim() == 3:
                y_pred = y_pred.unsqueeze(1)

            # Log & autoregressive feedback
            ground_truth.append(gt_t.detach().cpu())
            preds.append(y_pred.detach().cpu())
            # print(f"y_preds[{t}] shape: {y_pred.shape}, gt_t shape: {gt_t.shape}")

            # Update temp_stack for next step
            temp_stack = y_pred  # (B, N, H, W)

            mse_mask.append((y_pred - gt_t).pow(2).detach().cpu())
            rel_l2.append((relative_l2_percent(y_pred, gt_t), t))
            rel_l1.append((relative_l1_percent(y_pred, gt_t), t))
            mse_error.append((torch.mean((y_pred - gt_t) ** 2).item(), t))

    
    # Save
    for t in range(len(preds)):
        save_temperature_plot(
            preds[t][0, 0],
            path=f"results_val/{timestamp}/results_pred",
            name_prefix=f"seq_prediction_step{t} - {args.ckpt}"
        )
    for t in range(len(ground_truth)):
        save_temperature_plot(
            ground_truth[t][0, 0],
            path=f"results_val/{timestamp}/results_gt",
            name_prefix=f"seq_ground_truth_step{t} - {args.ckpt}"
        )
    for t in range(len(mse_mask)):
        save_temperature_plot(
            mse_mask[t][0, 0],
            path=f"results_val/{timestamp}/results_mse",
            name_prefix=f"seq_mse_step{t} - {args.ckpt}",
            label="MSE",
            scale_fix=True
            #max_val=1.0
        )

    plot_error(rel_l2, f"results_val/{timestamp}", filename="rel_l2.png", title=f"Rel. L2 - {args.ckpt}", y_axis="Rel. L2 [%]", x_axis="Step t")
    plot_error(rel_l1, f"results_val/{timestamp}", filename="rel_l1.png", title=f"Rel. L1 - {args.ckpt}", y_axis="Rel. L1 [%]", x_axis="Step t")
    plot_error(mse_error, f"results_val/{timestamp}", filename="mse.png", title=f"MSE - {args.ckpt}", y_axis="MSE", x_axis="Step t")

    print(f"Average MSE: {mse_evaluate_avg(preds, ground_truth):.4e}")
    print(f"Average Rel-L2: {relative_l2_percent(preds, ground_truth):.4f}%")
    print(f"Average Rel-L1: {relative_l1_percent(preds, ground_truth):.4f}%")
    print("\nRollout complete!")
    print(f"Total steps predicted: {len(preds)*N}")


if __name__ == "__main__":
    rollout()