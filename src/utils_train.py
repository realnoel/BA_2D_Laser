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

def mse_evaluate_avg(y_preds: torch.Tensor, y_trues: torch.Tensor):
    """
    Compute Mean Squared Error (MSE) between predicted and true tensors.
    """
    Yp = torch.stack(y_preds, dim=0)   # (T,B,1,H,W)
    Yt = torch.stack(y_trues, dim=0)   # (T,B,1,H,W)
    if Yp.dim() == 5 and Yt.dim() == 4:
        Yt = Yt.unsqueeze(1)
    assert Yp.shape == Yt.shape 

    return torch.mean((Yp - Yt) ** 2).item()  # true average over all dims

def relative_l2_percent(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    # handles your unsqueeze convention (B,H,W) -> (B,1,H,W)
    if torch.is_tensor(y_pred) and torch.is_tensor(y_true):
        Yp = y_pred
        Yt = y_true
    elif isinstance(y_pred, (list, tuple)) and isinstance(y_true, (list, tuple)) and len(y_pred) > 1 and len(y_true) > 1:
        Yp = torch.stack(y_pred, dim=0)
        Yt = torch.stack(y_true, dim=0)
    elif isinstance(y_pred, (list, tuple)) and isinstance(y_true, (list, tuple)) and len(y_pred) == 1 and len(y_true) == 1:
        Yp = y_pred[0]
        Yt = y_true[0]
    else: 
        raise ValueError(f"Unexpected input types: {type(Yp)}, {type(Yt)}")
    
    # Relative L2 as you did: sqrt(mean((y_hat - y)^2)/mean(y^2)) * 100
    return (torch.mean((Yp - Yt) ** 2) / torch.mean(Yt ** 2)).sqrt() * 100.0

def relative_l1_percent(y_pred: torch.Tensor, y_true: torch.Tensor) -> torch.Tensor:
    """Compute Relative L1 (%) between predicted and true tensors."""
    if torch.is_tensor(y_pred) and torch.is_tensor(y_true):
        Yp = y_pred
        Yt = y_true
    elif isinstance(y_pred, (list, tuple)) and isinstance(y_true, (list, tuple)) and len(y_pred) > 1 and len(y_true) > 1:
        Yp = torch.stack(y_pred, dim=0)
        Yt = torch.stack(y_true, dim=0)
    elif isinstance(y_pred, (list, tuple)) and isinstance(y_true, (list, tuple)) and len(y_pred) == 1 and len(y_true) == 1:
        Yp = y_pred[0]
        Yt = y_true[0]
    else: 
        raise ValueError(f"Unexpected input types: {type(Yp)}, {type(Yt)}")
    
    return (torch.mean(torch.abs(Yp - Yt)) / torch.mean(torch.abs(Yt))) * 100.0