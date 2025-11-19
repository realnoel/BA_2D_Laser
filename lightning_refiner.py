# train_lightning.py
import pytorch_lightning as L
import torch
import yaml
import numpy as np

from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from model.model_fno_baseline import FNO2dBaseline
from model.pde_refiner import PDERefiner
from model.model_fno_refiner import FNO2dRefiner

from refiner.model_cno import CNO2d # Delete

# ---------- Helpers ----------
def relative_l2_percent(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # handles your unsqueeze convention (B,H,W) -> (B,1,H,W)
    # print(f"relative_l2_percent: y_hat dim: {y_hat.ndim}, y_hat shape: {y_hat.shape}")
    # print(f"relative_l2_percent: y dim: {y.ndim}, y shape: {y.shape}")
    if y.ndim == 3:  # (B,H,W)
        y = y.unsqueeze(1)
    if y_hat.ndim == 3:
        y_hat = y_hat.unsqueeze(1)
    # Relative L2 as you did: sqrt(mean((y_hat - y)^2)/mean(y^2)) * 100
    return (torch.mean((y_hat - y) ** 2) / torch.mean(y ** 2)).sqrt() * 100.0

def mse_evaluate_avg(Yp: torch.Tensor, Yt: torch.Tensor):
    """
    Compute Mean Squared Error (MSE) between predicted and true tensors.
    """
    if Yt.ndim == 3:  # (B,H,W)
        Yt = Yt.unsqueeze(1)
    if Yp.ndim == 3:
        Yp = Yp.unsqueeze(1)

    # Mean Squared Error
    return torch.mean((Yp - Yt) ** 2).item()  # true average over all dims

def relative_l1_percent(Yp: torch.Tensor, Yt: torch.Tensor) -> torch.Tensor:
    """Compute Relative L1 (%) between predicted and true tensors."""
    if Yt.ndim == 3:  # (B,H,W)
        Yt = Yt.unsqueeze(1)
    if Yp.ndim == 3:
        Yp = Yp.unsqueeze(1)

    # Relative L1 as you did: mean(|y_hat - y|)/mean(|y|) * 100
    return (torch.mean(torch.abs(Yp - Yt)) / torch.mean(torch.abs(Yt))) * 100.0

# ---------- LightningModule ----------
class LitModel(L.LightningModule):
    def __init__(self, baseline: nn.Module, refiner: nn.Module, config: dict, K: int, name_base: str, name_ref: str):
        super().__init__()
        self.baseline = baseline
        self.refiner = refiner
        self.config = config
        self.K = K
        self.crit = nn.MSELoss()

        # pick the right block to store
        if name_base.upper() == "CNO":
            base_cfg = dict(config.get("model_cno", {}))
        elif name_base.upper() == "FNO":
            base_cfg = dict(config.get("model_fno", {}))
        else:
            raise ValueError(f"Unknown name_base={name_base!r}")
        
        if name_ref.upper() == "REFINER_MODEL_FNO":
            ref_cfg = dict(config.get("refiner_model_fno", {}))
        else:
            raise ValueError(f"Unknown name_ref={name_ref!r}")

        # keep ckpt small but self-contained
        opt_key = self.config["training"]["optimizer"]      # optimizer name from default.yaml
        sch_key = self.config["training"]["scheduler"]      # scheduler name from default.yaml
        self.save_hyperparameters({
            "name_base":  name_base,     
            "base_cfg":   base_cfg,      
            "name_ref":   name_ref,
            "ref_cfg":    ref_cfg,
            "optimizer":   self.config.get(opt_key, {}), 
            "scheduler":   self.config.get(sch_key, {}),
            "training":    self.config.get("training", {})
        })

    def forward(self, x):
        return self.refiner(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        u_t = self.baseline(x)[:, 0:1, :, :]
        u_prev = x[:, 0:1, :, :]
        pred = self.refiner(u_t, u_prev)
        loss = self.crit(pred, y)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        u_t = self.baseline(x)[:, 0:1, :, :]
        u_prev = x[:, 0:1, :, :]
        y_hat = self.refiner(u_t, u_prev)
        rell2 = relative_l2_percent(y_hat, y)
        mse = mse_evaluate_avg(y_hat, y)
        rell1 = relative_l1_percent(y_hat, y)
        self.log("val_mse", mse, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rel_l1_percent", rell1, on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rel_l2_percent", rell2, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_rel_l2_percent": rell2}

    def configure_optimizers(self):
        opt_key = self.config["training"]["optimizer"]      # optimizer name from default.yaml
        sch_key = self.config["training"]["scheduler"]      # scheduler name from default.yaml
        opt_conf = self.config[opt_key]                     # hyperparameter of optimizer from default.yaml
        sch_conf = self.config[sch_key]                     # hyperparameter of scheduler from default.yaml
        
        if self.config['training']['optimizer'] == "optimizer_adam":
            optimizer = Adam(self.parameters(), lr=opt_conf["lr"])
        elif self.config['training']['optimizer'] == "optimizer_adamw":
            optimizer = AdamW(self.parameters(), lr=opt_conf["lr"], weight_decay=opt_conf['weight_decay'])
        else:
            ValueError(f"Optimizer not installed {self.config['training']['optimizer']}")

        if self.config['training']['scheduler'] == "scheduler_cosineannealinglr":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max = self.config['training']['epochs'],
            )
        elif self.config['training']['scheduler'] == "scheduler_steplr":
            scheduler = StepLR(
                optimizer,
                step_size=sch_conf["step_size"],
                gamma=sch_conf["gamma"]
            )
        else:
            ValueError(f"Scheduler not installed {self.config['training']['scheduler']}")

        return { # For ReduceLROnPlateau, you must include "monitor": "<metric_name>" and set "reduce_on_plateau": True if you use the older tuple form.
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch",
                "frequency": 1,
            }
        }

# ---------- DataModule ----------
class PDEDataModule(L.LightningDataModule):
    def __init__(self, config: dict):
        super().__init__()
        self.config = config
        self.batch_size = config["training"]["batch_size"]
        self.seed = config["training"]["seed"]

        # will be set in setup()
        self.train_ds = None
        self.val_ds = None

    def setup(self, stage=None):
        full_ds = PDEDatasetLoader_Multi(which="train", N=self.config["training"]["N"], K=1)
        n_total = len(full_ds)
        n_val = int(self.config["training"]["test_split"] * n_total)
        n_train = n_total - n_val

        gen = torch.Generator().manual_seed(self.seed)
        self.train_ds, self.val_ds = random_split(full_ds, [n_train, n_val], generator=gen)

    def train_dataloader(self):
        return DataLoader(self.train_ds, 
                          batch_size=self.batch_size, 
                          shuffle=True, 
                          num_workers=0
                          )

    def val_dataloader(self):
        return DataLoader(self.val_ds, 
                         batch_size=self.batch_size, 
                         shuffle=False, 
                         num_workers=0
                         )
    
def load_baseline_weights(baseline_model: nn.Module, ckpt_path: str) -> nn.Module:
    print(f"[INFO] Loading baseline checkpoint: {ckpt_path}")

    ckpt = torch.load(ckpt_path, map_location="cpu")

    # Lightning saves parameters under "state_dict"
    state_dict = ckpt["state_dict"]

    # Filter out keys that belong only to the model
    model_state_dict = {
        k.replace("model.", ""): v
        for k, v in state_dict.items()
        if k.startswith("model.")
    }

    baseline_model.load_state_dict(model_state_dict, strict=False)

    # Freeze baseline
    for p in baseline_model.parameters():
        p.requires_grad = False

    print("[INFO] Baseline weights loaded successfully.")
    return baseline_model.eval()
    
# ---- BASELINE MODELS -----
#
# STRATEGY 1: Exogenous variables (Q, dx, dy) N future steps and N past steps.     Endogenous variable (T) only N past steps, target is N future steps.
# STRATEGY 2: Exogenous variables (Q, dx, dy) N future steps                       Endogenous variable (T) only N past steps, target is N future steps.
# STRATEGY 3: Exogenous variables (Q, dx, dy) N past steps plus next future step.  Endogenous variable (T) only N past steps, target is next future steps.
# STRATEGY 4: Exogenous variables (Q, dx, dy) next future step.                    Endogenous variable (T) only N past steps, target is next future steps.
#
# ---------- Main ----------
if __name__ == "__main__":
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)

    with open(config["refiner"]["baseline_hparams"], "r") as f:
        baseline_hparams = yaml.safe_load(f)

    # STRATEGY = config["training"]["strategy"]
    STRATEGY = 1
    assert STRATEGY in [1,2,3,4], "STRATEGY must be 1, 2, 3, or 4"

    N = config["training"]["N"]
    assert N >= 1, "N must be at least 1"

    if STRATEGY == 1:
        print("[INFO] Using STRATEGY 1: Exogenous variables N future steps and N past steps. Endogenous variable (T) only N past steps, target is N future steps.")
        from dataloader.dataloader_1 import PDEDatasetLoader_Multi
        IN_DIM = 7*N
        OUT_DIM = N
    elif STRATEGY == 2:
        print("[INFO] Using STRATEGY 2: Exogenous variables N future steps. Endogenous variable (T) only N past steps, target is N future steps.")
        from dataloader.dataloader_2 import PDEDatasetLoader_Multi
        IN_DIM = 4*N
        OUT_DIM = N
    elif STRATEGY == 3:
        print("[INFO] Using STRATEGY 3: Exogenous variables N past steps plus next future step. Endogenous variable (T) only N past steps, target is next future steps.")
        from dataloader.dataloader_3 import PDEDatasetLoader_Multi
        IN_DIM = 4*N+3
        OUT_DIM = 1
    elif STRATEGY == 4:
        print("[INFO] Using STRATEGY 4: Exogenous variables next future step. Endogenous variable (T) only N past steps, target is next future steps.")
        from dataloader.dataloader_4 import PDEDatasetLoader_Multi
        IN_DIM = N+3
        OUT_DIM = 1

    L.seed_everything(config["training"]["seed"], workers=True)

    name_base = baseline_hparams["training"]["model"]
    if name_base == "FNO":
        model_core = FNO2dBaseline(
            modes1=baseline_hparams["model_cfg"]["modes1"],
            modes2=baseline_hparams["model_cfg"]["modes2"],
            width=baseline_hparams["model_cfg"]["width"],
            in_dim=baseline_hparams["model_cfg"]["in_dim"],
            out_dim=baseline_hparams["model_cfg"]["out_dim"],
            pad=baseline_hparams["model_cfg"]["pad"]
        )
    elif name_base == "CNO":
        model_core = CNO2d(
            in_dim=baseline_hparams["model_cfg"]["in_dim"],
            out_dim=baseline_hparams["model_cfg"]["out_dim"],
            size=baseline_hparams["model_cfg"]["size"],
            N_layers=baseline_hparams["model_cfg"]["N_layers"],
            N_res=baseline_hparams["model_cfg"]["N_res"],
            N_res_neck=baseline_hparams["model_cfg"]["N_res_neck"],
            channel_multiplier=baseline_hparams["model_cfg"]["channel_multiplier"],
        )
    else:
        raise ValueError(f"Unknown model {name_base!r}, valid options are 'FNO' and 'CNO'")
    
    # -- load weights from baseline & freeze --
    config["training"]["using_refiner"] = True
    baseline = load_baseline_weights(model_core, config["refiner"]["baseline_path"])

    # -- build refiner_core model --
    name_ref = config["refiner"]["model"]
    if name_ref == "refiner_model_fno":
        refiner_core = FNO2dRefiner(
            modes1=config["refiner_model_fno"]["modes1"],
            modes2=config["refiner_model_fno"]["modes2"],
            width=config["refiner_model_fno"]["width"],
            in_dim=config["refiner_model_fno"]["in_dim"],
            out_dim=config["refiner_model_fno"]["out_dim"],
            pad=config["refiner_model_fno"]["pad"]
        )
    else:
        raise ValueError(f"Unknown refiner model {name_ref!r}, valid option is 'refiner_model_fno'")
    
    # -- build refiner --
    refiner = PDERefiner(
            model=refiner_core,
            K=config["refiner"]["k"],
            min_noise_std=config["refiner"]["min_noise_std"],
            )

    datamodule = PDEDataModule(baseline_hparams)
    K = config["refiner"]["k"]
    lit_model = LitModel(baseline=baseline, refiner=refiner, config=config, K=K, name_base=name_base, name_ref=name_ref)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("checkpoints") / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=outdir / "checkpoints",
        filename="best_epoch{epoch:03d}-{val_rel_l2_percent:.2f}",
        monitor="val_rel_l2_percent",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")
    logger = CSVLogger(save_dir=str(outdir), name="logs")

    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name()}")
        trainer = L.Trainer(
            max_epochs=config["training"]["epochs"],
            accelerator="gpu",
            devices="auto",
            # precision="16-mixed" if torch.cuda.is_available() else "32-true",
            logger=logger,
            callbacks=[ckpt_cb, lr_cb],
            log_every_n_steps=10
        )
    else:
        print("[INFO] Using CPU")
        trainer = L.Trainer(
            max_epochs=config["training"]["epochs"],
            accelerator="cpu",
            logger=logger,
            callbacks=[ckpt_cb, lr_cb],
            log_every_n_steps=10,
            
        )

    trainer.fit(lit_model, datamodule=datamodule)

    # After training: print best ckpt and final metrics path
    print(f"[INFO] Best checkpoint: {ckpt_cb.best_model_path}")
    print(f"[INFO] Logs stored in: {logger.log_dir}")