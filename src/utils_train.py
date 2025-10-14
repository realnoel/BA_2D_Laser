import os
import torch
from datetime import datetime

def now_run_id():
    """DDMMYYYY_HHMMSS string for folder/run id."""
    return datetime.now().strftime("%d%m%Y_%H%M%S")

def _ckpt_dir(run_id):
    d = os.path.join("checkpoints", run_id)
    os.makedirs(d, exist_ok=True)
    return d

def save_checkpoint(model, optimizer, scheduler, epoch, val_metric, config, norm_tuple, run_id, is_best=False):
    """
    Save a training checkpoint.
    - model:     nn.Module (works with PadCropFNO as-is)
    - optimizer: torch optimizer
    - scheduler: LR scheduler or None
    - epoch:     int
    - val_metric: float (e.g., Rel-L2 % on validation)
    - config:    dict (your YAML config)
    - norm_tuple: (min_p, max_p, min_shift, max_shift, min_model, max_model)
    - run_id:    folder name (e.g., timestamp)
    - is_best:   if True, also write best.pt
    """
    state = {
        "epoch": epoch,
        "val_metric": val_metric,
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "config": config,
        "norm": norm_tuple,
        "run_id": run_id,
    }
    ckpt_dir = _ckpt_dir(run_id)
    epoch_path = os.path.join(ckpt_dir, f"epoch_{epoch:04d}.pt")
    last_path  = os.path.join(ckpt_dir, "last.pt")
    tmp_path   = epoch_path + ".tmp"

    torch.save(state, tmp_path)      # atomic-ish write
    os.replace(tmp_path, epoch_path)
    torch.save(state, last_path)
    if is_best:
        best_path = os.path.join(ckpt_dir, "best.pt")
        torch.save(state, best_path)
    return epoch_path

def load_checkpoint(model, ckpt_path, optimizer=None, scheduler=None, map_location="cpu"):
    """
    Load a checkpoint; returns (epoch, val_metric, config, norm, run_id).
    If optimizer/scheduler provided, restore their states too.
    """
    state = torch.load(ckpt_path, map_location=map_location)
    model.load_state_dict(state["model_state"])
    if optimizer is not None and state.get("optimizer_state") is not None:
        optimizer.load_state_dict(state["optimizer_state"])
    if scheduler is not None and state.get("scheduler_state") is not None:
        scheduler.load_state_dict(state["scheduler_state"])

    return (
        state.get("epoch", 0),
        state.get("val_metric", None),
        state.get("config", None),
        state.get("norm", None),
        state.get("run_id", None),
    )

def mse_evalute_avg(y_pred: torch.Tensor, y_true: torch.Tensor):
    """
    Compute Mean Squared Error (MSE) between predicted and true tensors.
    """
    y_pred = torch.stack(y_pred, dim=0)
    y_true = torch.stack(y_true, dim=0)
    assert y_pred.shape == y_true.shape, "Shape mismatch between y_pred and y_true"
    mse = 0
    for y_pred, y_true in zip(y_pred, y_true):
        mse += torch.mean((y_pred - y_true) ** 2).item()
    return mse / len(y_pred)

def relative_l2_percent(y_pred: torch.Tensor, y_true: torch.Tensor, eps: float = 1e-12) -> torch.Tensor:
    # handles your unsqueeze convention (B,H,W) -> (B,1,H,W)
    y_pred = torch.stack(y_pred, dim=0)
    y_true = torch.stack(y_true, dim=0)
    # Relative L2 as you did: sqrt(mean((y_hat - y)^2)/mean(y^2)) * 100
    return (torch.mean((y_pred - y_true) ** 2) / torch.mean(y_true ** 2)).sqrt() * 100.0