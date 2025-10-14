# train_lightning.py
import pytorch_lightning as L
import torch
import yaml

from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam
from torch.optim.lr_scheduler import StepLR
from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor
from pytorch_lightning.loggers import CSVLogger

from dataloader import PDEDatasetLoader_Multi
from model_fno import FNO2d
from model_cno import CNO2d
# from utils_images import save_temperature_plot  # optional

# ---------- Helpers ----------
def relative_l2_percent(y_hat: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    # handles your unsqueeze convention (B,H,W) -> (B,1,H,W)
    if y.ndim == 3:  # (B,H,W)
        y = y.unsqueeze(1)
    if y_hat.ndim == 3:
        y_hat = y_hat.unsqueeze(1)
    # Relative L2 as you did: sqrt(mean((y_hat - y)^2)/mean(y^2)) * 100
    return (torch.mean((y_hat - y) ** 2) / torch.mean(y ** 2)).sqrt() * 100.0

# ---------- LightningModule ----------
class LitModel(L.LightningModule):
    def __init__(self, model: nn.Module, config: dict, use_model: str):
        super().__init__()
        self.model = model
        self.config = config
        self.criterion = nn.MSELoss()
        self.use_model = use_model

        # pick the right block to store
        if use_model.upper() == "CNO":
            model_cfg = dict(config.get("model_cno", {}))
            model_name = "CNO"
        elif use_model.upper() == "FNO":
            model_cfg = dict(config.get("model_fno", {}))
            model_name = "FNO"
        else:
            raise ValueError(f"Unknown use_model={use_model!r}")

        # keep ckpt small but self-contained
        self.save_hyperparameters({
            "model_name":  model_name,     # "CNO" or "FNO"
            "model_cfg":   model_cfg,      # ONLY the chosen sub-config
            "optimizer":   config.get("optimizer", {}),
            "training":    config.get("training", {})
        })

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        # print(next(self.model.parameters()).device)
        # print(x.device)
        # unify dims like in your loop
        if y.ndim == 3:
            y = y.unsqueeze(1)
        loss = self.criterion(y_hat, y)
        self.log("train_mse", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        rel = relative_l2_percent(y_hat, y)
        self.log("val_rel_l2_percent", rel, on_step=False, on_epoch=True, prog_bar=True)
        return {"val_rel_l2_percent": rel}

    def configure_optimizers(self):
        opt_conf = self.config["optimizer"]
        optimizer = Adam(self.parameters(), lr=opt_conf["lr"])
        scheduler = StepLR(
            optimizer,
            step_size=opt_conf["step_size"],
            gamma=opt_conf["gamma"]
        )
        return {
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
        full_ds = PDEDatasetLoader_Multi(which="train")
        n_total = len(full_ds)
        n_val = int(self.config["training"]["test_split"] * n_total)
        n_train = n_total - n_val

        # (optional) expose your normalization tuple if needed later
        self.norm_tuple = (
            full_ds.min_p, full_ds.max_p,
            full_ds.min_shift, full_ds.max_shift,
            full_ds.min_model, full_ds.max_model
        )

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

# ---------- Main ----------
if __name__ == "__main__":
    with open("configs/default.yaml", "r") as f:
        config = yaml.safe_load(f)
    
    # torch.set_float32_matmul_precision('high') #('medium' | 'high')

    print(torch.cuda.get_device_name())

    L.seed_everything(config["training"]["seed"], workers=True)

    # pick model (CNO default, switch to FNO by uncommenting)
    use_model = "CNO"
    if use_model == "FNO":
        model_core = FNO2d(
            modes1=config["model_fno"]["modes1"],
            modes2=config["model_fno"]["modes2"],
            width=config["model_fno"]["width"],
            in_channels=config["model_fno"]["in_channels"],
            out_channels=config["model_fno"]["out_channels"],
            pad=config["model_fno"]["pad"]
        )
    elif use_model == "CNO":
        model_core = CNO2d(
            in_dim=config["model_cno"]["in_dim"],
            out_dim=config["model_cno"]["out_dim"],
            size=config["model_cno"]["size"],
            N_layers=config["model_cno"]["N_layers"],
            N_res=config["model_cno"]["N_res"],
            N_res_neck=config["model_cno"]["N_res_neck"],
            channel_multiplier=config["model_cno"]["channel_multiplier"],
        )

    lit_model = LitModel(model_core, config, use_model)
    datamodule = PDEDataModule(config)

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

    trainer = L.Trainer(
        max_epochs=config["training"]["epochs"],
        accelerator="gpu",
        devices="auto",
        # precision="16-mixed" if torch.cuda.is_available() else "32-true",
        logger=logger,
        callbacks=[ckpt_cb, lr_cb],
        log_every_n_steps=10
    )

    trainer.fit(lit_model, datamodule=datamodule)

    # After training: print best ckpt and final metrics path
    print(f"[INFO] Best checkpoint: {ckpt_cb.best_model_path}")
    print(f"[INFO] Logs stored in: {logger.log_dir}")
