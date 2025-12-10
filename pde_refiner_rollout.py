# ---------------------------------------------------------
# HOW TO RUN THIS ROLLOUT
#
# Example:
# python pde_refiner_rollout.py \
#     --ckpt RUN_2025_01_27_12_00 \
#     --strategy 3 \
#     --idx 0 \
#     --steps -1 \
#     --mode phys \
#     --device cuda
#
# Checkpoint folder must be:
# checkpoints/<RUN_NAME>/checkpoints/epoch=XX-step=YY.ckpt
#
# --idx selects sequence from test set
# --steps -1 means use full sequence length
# ---------------------------------------------------------

import torch, os, glob, argparse, yaml, time, csv
from datetime import datetime

from model.model_cno_timeModule_refiner import CNO2d_Temporal
from pde_refiner import PDERefiner
from utils.utils_images import save_temperature_plot, plot_error
from utils.utils_train import mse_evaluate_avg, relative_l2_percent, relative_l1_percent

# ---------------------------------------------------------
# ARGS
# ---------------------------------------------------------
def parse_args():
    p = argparse.ArgumentParser("Rollout of PDERefiner on test set")
    p.add_argument("--ckpt", required=True, help="Name of checkpoint directory")
    p.add_argument("--mode", default="norm", choices=["norm", "phys"])
    p.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    p.add_argument("--strategy", type=int, choices=[1,2,3,4])
    p.add_argument("--steps", type=int, default=-1)
    p.add_argument("--idx", type=int, default=0)
    return p.parse_args()

# ---------------------------------------------------------
# CKPT HELPERS
# ---------------------------------------------------------
def load_weights_into_model(model, ckpt_file, device):
    ckpt = torch.load(ckpt_file, map_location=device)
    sd = ckpt.get("state_dict") or ckpt.get("model_state")
    if sd is None:
        raise RuntimeError("No state_dict or model_state in checkpoint!")

    new_sd = {}
    for k, v in sd.items():
        if k.startswith("model."):
            k = k[6:]
        if k.startswith("net."):
            k = k[4:]
        if k.startswith("module."):
            k = k[7:]
        new_sd[k] = v

    model.load_state_dict(new_sd, strict=False)
    model.to(device).eval()
    print("[INFO] Weights loaded.")
    return model

def read_hparams(ckpt_dir):
    hp = os.path.join(ckpt_dir, "logs", "version_0", "hparams.yaml")
    if not os.path.exists(hp):
        raise FileNotFoundError(f"hparams.yaml not found at {hp}")
    return yaml.safe_load(open(hp))

def find_ckpt_file(ckpt_dir):
    pattern = os.path.join(ckpt_dir, "checkpoints", "*.ckpt")
    ckpts = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    if not ckpts:
        raise FileNotFoundError(f"No ckpt found under {ckpt_dir}/checkpoints/")
    print(f"[INFO] Using checkpoint: {ckpts[0]}")
    return ckpts[0]

# ---------------------------------------------------------
# MODEL BUILDER
# ---------------------------------------------------------
def build_model(hparams):
    config = hparams
    use_model = config["training"]['model'].upper()

    if use_model not in ["CNO", "FNO", "CNO_TEMP"]:
        raise ValueError(f"Invalid model_name: {config['model_name']}")

    model_cfg = config["model_cfg"]

    if use_model == "CNO_TEMP":
        core_model = CNO2d_Temporal(**model_cfg)
    
    return PDERefiner(
        neural_operator=core_model,
        config=config
    )

# ---------------------------------------------------------
def feature_to_physical_space_T(T, norm):
    min_v = norm["min_model"]
    max_v = norm["max_model"]
    return torch.exp(T * (max_v - min_v) + min_v)

def get_steps(steps, N):
    r = steps % N
    return steps if r == 0 else steps + (N - r)

def save_rollout_csv(out_dir, rollout_times):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, "rollout_times.csv")

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["block_start", "block_end", "duration_sec"])
        for row in rollout_times:
            writer.writerow(row)

    print(f"[INFO] Saved rollout timing CSV to {filename}")

def save_metrics_csv(out_dir, metrics):
    os.makedirs(out_dir, exist_ok=True)
    filename = os.path.join(out_dir, "metrics.csv")

    # Summary first: global averages
    avg_l1 = sum(m[1] for m in metrics) / len(metrics)
    avg_l2 = sum(m[2] for m in metrics) / len(metrics)
    avg_mse = sum(m[3] for m in metrics) / len(metrics)

    summary = [("summary", avg_l1, avg_l2, avg_mse)]

    with open(filename, mode="w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["timestep", "L1", "L2", "MSE"])

        # write summary at top
        for row in summary:
            writer.writerow(row)

        # write all tracked metrics
        for row in metrics:
            writer.writerow(row)

    print(f"[INFO] Saved metrics CSV to {filename}")



# ---------------------------------------------------------
# MAIN ROLLOUT
# ---------------------------------------------------------
@torch.no_grad()
def rollout():
    args = parse_args()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # DEVICE
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)

    # LOAD MODEL
    ckpt_dir = f"checkpoints/{args.ckpt}"
    hparams = read_hparams(ckpt_dir)
    model = build_model(hparams)
    ckpt_file = find_ckpt_file(ckpt_dir)
    model = load_weights_into_model(model, ckpt_file, device)

    # SELECT STRATEGY
    if args.strategy is None:
        raise ValueError("You MUST pass --strategy 1|2|3|4")

    if args.strategy == 1:
        from dataloader.dataloader_1 import PDEDatasetLoader_Multi
    elif args.strategy == 2:
        from dataloader.dataloader_2 import PDEDatasetLoader_Multi
    elif args.strategy == 3:
        from dataloader.dataloader_3 import PDEDatasetLoader_Multi
    else:
        from dataloader.dataloader_4 import PDEDatasetLoader_Multi

    # LOAD NORMS + TEST DATASET
    N = hparams["training"]["N"]
    steps_calc = get_steps(args.steps, N)
    steps_return = args.steps

    train_norm = PDEDatasetLoader_Multi("train", refiner_output=False).get_norm()
    val_ds = PDEDatasetLoader_Multi(
        which="test",
        N=N,
        K=steps_calc,
        refiner_output=False
    )
    val_ds.load_norm(train_norm)

    norm = {
        "min_p": train_norm[0],
        "max_p": train_norm[1],
        "min_shift": train_norm[2],
        "max_shift": train_norm[3],
        "min_model": train_norm[4],
        "max_model": train_norm[5]
    }

    # LOAD SEQUENCE
    x, y = val_ds[args.idx]          # shapes (T,C,H,W)
    x = x.unsqueeze(0).to(device)
    y = y.unsqueeze(0).to(device)

    print(f"Input:  {x.shape}")
    print(f"Target: {y.shape}")

    B, T, _, H, W = y.shape
    assert B == 1

    # initial temp window (history)
    temp_stack = x[:, 0, 0:N, ...].contiguous()     # (1,N,H,W)

    preds = []
    gt = []                      # (B, N, H, W)
    # print(f"GT shape: {gt.shape}")
    mse_mask, rel_l2, rel_l1, mse_error = [], [], [], []
    rollout_times, metrics = [], []

    # ROLLOUT LOOP
    for t in range(0, steps_calc, N):

        cond_t = x[:, t, N:, ...].contiguous()          # (B, C_exog, H, W)
        temp_prev = temp_stack                          # (B, N, H, W)
        t0 = time.time()

        # predict N frames at once: (B, N, H, W)
        y_pred = model(temp_prev, cond_t)

        t1 = time.time()
        rollout_times.append((t, t+N, t1 - t0))

        if y_pred.ndim == 3:
            y_pred = y_pred.unsqueeze(1)                # → (B,1,H,W)

        print(f"y_pred shape {y_pred.shape}")

        # iterate through N predicted frames
        for i in range(N):
            yp = y_pred[:, i:i+1]                       # (B,1,H,W)
            t_global = t + i                            # correct global time index

            if t_global >= steps_calc:                       # avoid overflow
                break

            # LOGGING
            if args.mode == "phys":
                preds.append(feature_to_physical_space_T(yp[:,0].cpu(), norm))      
                gt.append(feature_to_physical_space_T(y[:, t_global, 0].cpu(), norm))
            else:
                preds.append(yp[:,0].cpu())             # (B,H,W)
                gt.append(y[:, t_global, 0].cpu())     # (B,H,W)

            gt_frame = gt[-1].to(yp.device)
            # METRICS
            mse_mask.append(((yp - gt_frame).pow(2).cpu()))
            rel_l2.append((relative_l2_percent(yp, gt_frame), t_global))
            rel_l1.append((relative_l1_percent(yp, gt_frame), t_global))
            mse_error.append((torch.mean((yp - gt_frame)**2).item(), t_global))

            metrics.append((
                t_global,
                float(relative_l1_percent(yp, gt_frame)),
                float(relative_l2_percent(yp, gt_frame)),
                float(torch.mean((yp - gt_frame)**2))
            ))


        # autoregressive: last predicted frame becomes newest history frame
        temp_stack = torch.cat([temp_stack[:, 1:], y_pred[:, -1:]], dim=1)

        t2 = time.time()
        print(f"Block {t}-{t+N} done. Time left: {(steps_calc - t - N) / N * (t1 - t0):.2f} sec")
    
    # print(f"{gt[0, 0].shape}, {gt[:, 0].shape}, {gt[0, 0:1].shape}")
    # print(f"{preds[0].shape} predicted frames.")
    # print(f"{gt[0, :, 0].shape} ground truth frames.")

    # SAVE RESULTS
    out_dir = f"results_val/{timestamp}"
    for t in range(steps_return):
        save_temperature_plot(preds[t][0], f"{out_dir}/results_pred",
                              f"pred_t{t}-{args.ckpt}")
        save_temperature_plot(gt[t], f"{out_dir}/results_gt",
                              f"gt_t{t}-{args.ckpt}")
        save_temperature_plot(mse_mask[t][0], f"{out_dir}/results_mse",
                              f"mse_t{t}-{args.ckpt}")
    
    # Runtime informations
    total_time = sum(rt[2] for rt in rollout_times)
    summary_entry = (0, steps_calc-1, total_time)
    rollout_times = [summary_entry] + rollout_times
    save_rollout_csv(out_dir, rollout_times)

    save_metrics_csv(out_dir, metrics)

    plot_error(rel_l2, out_dir, "rel_l2.png",
               f"Rel-L2 {args.ckpt}", "Rel-L2 [%]", "t")
    plot_error(rel_l1, out_dir, "rel_l1.png",
               f"Rel-L1 {args.ckpt}", "Rel-L1 [%]", "t")
    plot_error(mse_error, out_dir, "mse.png",
               f"MSE {args.ckpt}", "MSE", "t")

    print(f"Avg MSE:    {mse_evaluate_avg(preds, gt):.4e}")
    print(f"Avg Rel-L2: {relative_l2_percent(preds, gt):.4f}%")
    print(f"Avg Rel-L1: {relative_l1_percent(preds, gt):.4f}%")
    print("Done.")

if __name__ == "__main__":
    rollout()
