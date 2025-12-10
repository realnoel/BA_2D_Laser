# train_lightning.py
import pytorch_lightning as L
import torch.nn.functional as F
import torch
import yaml
import os

from torch import nn
from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import StepLR, CosineAnnealingLR
from datetime import datetime
from pathlib import Path
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping
from pytorch_lightning.loggers import CSVLogger
from diffusers.schedulers import DDPMScheduler

from model.model_fno_refiner import FNO2dRefiner
from model.model_cno_timeModule_refiner import CNO2d_Temporal
from utils.utils_refiner import CustomMSELoss, ExponentialMovingAverage, MSE, REL_L1, REL_L2


class PDERefiner(L.LightningModule):
    def __init__(self, neural_operator: nn.Module, config: dict):
        super().__init__()
        self.neural_operator = neural_operator
        self.config = config
        
        self.N = config["training"]["N"]
        self.use_residual = config["refiner"]["use_residual"]
        self.cfg_train = config["training"]
        self.core_model_name = config["training"]["model"]
        self.cfg_model = config.get(f"model_{self.core_model_name.lower()}", {})
        self.cfg_refiner = config.get("refiner", {})
        cfg_opt = config["training"]["optimizer"]
        cfg_sch = config["training"]["scheduler"]      
        
        self.K_max = config["refiner"]["k_max"]
        self.min_noise_std = float(config["refiner"]["min_noise_std"])
        self.difference_weights = config["refiner"]["difference_weights"]

        self.ema = ExponentialMovingAverage(self.neural_operator, decay=config["refiner"]["ema_decay"])
        betas = [self.min_noise_std ** (k / self.K_max) for k in reversed(range(self.K_max + 1))]
        self.scheduler = DDPMScheduler(
            num_train_timesteps=self.K_max + 1,
            trained_betas=betas,
            prediction_type="v_prediction",
            clip_sample=False,
        )
        self.train_criterion = CustomMSELoss()
        self.time_multiplier = config["refiner"]["time_multiplier"] / self.K_max
        self.time_future = config["refiner"]["time_future"]  # Assuming single time step prediction

        self.save_hyperparameters({
            "model_cfg":   self.cfg_model,
            "optimizer":   config.get(cfg_opt, {}), 
            "scheduler":   config.get(cfg_sch, {}),
            "training":    config.get("training", {}),
            "refiner":    config.get("refiner", {}),
        })

    def forward(self, x, cond):
        return self.predict_next_solution(x, cond)
    
    @torch.no_grad()
    def predict_next_solution(self, x, cond):
        y_noised = torch.randn(size=(x.shape[0], self.time_future, *x.shape[2:]), dtype=x.dtype, device=x.device)
        self.scheduler.set_timesteps(self.K_max, device=x.device)
        for k in self.scheduler.timesteps:
            time = torch.zeros(size=(x.shape[0],), dtype=x.dtype, device=x.device) + k
            x_in = torch.cat([x, y_noised, cond], dim=1)
            pred = self.neural_operator(x_in, time=time*self.time_multiplier)
            y_noised = self.scheduler.step(pred, k, y_noised).prev_sample
        if self.use_residual:
            return y_noised * self.difference_weights + x[:, 0:self.N]
        return y_noised * self.difference_weights
    
    def train_step(self, x, y, cond):
        if self.use_residual:
            residual = (y - x[:, 0:self.N]) / self.difference_weights
        else: residual = y
        k = torch.randint(0, self.scheduler.config.num_train_timesteps, (x.shape[0],), device=x.device)
        noise = torch.randn_like(residual)
        y_noised = self.scheduler.add_noise(residual, noise, k)
        x_in = torch.cat([x, y_noised, cond], dim=1)
        pred = self.neural_operator(x_in, time=k*self.time_multiplier)
        target = self.scheduler.get_velocity(residual, noise, k)
        loss = self.train_criterion(pred, target)
        return loss, pred, target

    def training_step(self, batch):
        x, y, cond = batch
        loss, pred, target = self.train_step(x, y, cond)
        self.log("train_mse", MSE(pred, target), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_rel_l2", REL_L2(pred, target), on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_loss", loss, on_step=False, on_epoch=True, prog_bar=True)
        return loss

    @torch.no_grad()
    def validation_step(self, batch):
        x, y, cond = batch
        y_hat = self(x, cond)
        self.log("val_mse", MSE(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rel_l1", REL_L1(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        self.log("val_rel_l2", REL_L2(y_hat, y), on_step=False, on_epoch=True, prog_bar=True)
        return {"val_rel_l2": REL_L2(y_hat, y)}

    def configure_optimizers(self):
        opt_key = self.config["training"]["optimizer"]      # optimizer name from default.yaml
        sch_key = self.config["training"]["scheduler"]      # scheduler name from default.yaml
        opt_conf = self.config[opt_key]                     # hyperparameter of optimizer from default.yaml
        sch_conf = self.config[sch_key]                     # hyperparameter of scheduler from default.yaml
        
        if self.config['training']['optimizer'] == "optimizer_adam":
            optimizer = Adam(self.parameters(), lr=float(opt_conf["lr"]))
        elif self.config['training']['optimizer'] == "optimizer_adamw":
            optimizer = AdamW(self.parameters(), lr=float(opt_conf["lr"]), weight_decay=float(opt_conf['weight_decay']))
        else:
            ValueError(f"Optimizer not installed {self.config['training']['optimizer']}")

        if self.config['training']['scheduler'] == "scheduler_cosineannealinglr":
            scheduler = CosineAnnealingLR(
                optimizer,
                T_max = self.config['training']['epochs'],
                eta_min = float(self.config['scheduler_cosineannealinglr']["eta_min"])
            )
        elif self.config['training']['scheduler'] == "scheduler_steplr":
            scheduler = StepLR(
                optimizer,
                step_size=sch_conf["step_size"],
                gamma=float(sch_conf["gamma"])
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
    
    def on_fit_start(self):
        self.ema.register()

    def on_train_batch_end(self, *args, **kwargs):
        self.ema.update()

    def on_validation_start(self):
        self.apply_ema()

    def on_validation_end(self):
        self.remove_ema()

    def on_test_start(self):
        self.apply_ema()

    def on_test_end(self):
        self.remove_ema()

    def apply_ema(self):
        self.ema.apply_shadow()

    def remove_ema(self):
        self.ema.restore()

    def on_save_checkpoint(self, checkpoint):
        checkpoint["ema"] = self.ema.shadow

    def on_load_checkpoint(self, checkpoint):
        if "ema" in checkpoint:
            self.ema.shadow = checkpoint["ema"]

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
        from dataloader.dataloader_3 import PDEDatasetLoader_Multi
        full_ds = PDEDatasetLoader_Multi(which="train", 
                                         N=self.config["training"]["N"], 
                                         K=1,
                                         refiner_output=True)
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

    STRATEGY = config["training"]["strategy"]
    assert STRATEGY in [1,2,3,4], "STRATEGY must be 1, 2, 3, or 4"

    N = config["training"]["N"]
    assert N >= 1, "N must be at least 1"

    if STRATEGY == 1:
        print("[INFO] Using STRATEGY 1: Exogenous variables N future steps and N past steps. Endogenous variable (T) only N past steps, target is N future steps.")
        from dataloader.dataloader_1 import PDEDatasetLoader_Multi
        IN_DIM = 4*2*N
        OUT_DIM = N #N
        config["refiner"]["time_future"] = N
    elif STRATEGY == 2:
        print("[INFO] Using STRATEGY 2: Exogenous variables N future steps. Endogenous variable (T) only N past steps, target is N future steps.")
        from dataloader.dataloader_2 import PDEDatasetLoader_Multi
        IN_DIM = 5*N
        OUT_DIM = N # N
        config["refiner"]["time_future"] = N
    elif STRATEGY == 3:
        print("[INFO] Using STRATEGY 3: Exogenous variables N past steps plus next future step. Endogenous variable (T) only N past steps, target is next future steps.")
        from dataloader.dataloader_3 import PDEDatasetLoader_Multi
        IN_DIM = (N+1)*4
        OUT_DIM = 1
        config["refiner"]["time_future"] = 1
    elif STRATEGY == 4:
        print("[INFO] Using STRATEGY 4: Exogenous variables next future step. Endogenous variable (T) only N past steps, target is next future steps.")
        from dataloader.dataloader_4 import PDEDatasetLoader_Multi
        IN_DIM = 4+N
        OUT_DIM = 1
        config["refiner"]["time_future"] = 1

    L.seed_everything(config["training"]["seed"], workers=True)

    use_model = config["training"]["model"]
    if use_model.upper() == "FNO":
        if config["model_fno"]["in_dim"] != IN_DIM:
            config["model_fno"]["in_dim"] = IN_DIM
        if config["model_fno"]["out_dim"] != OUT_DIM:
            config["model_fno"]["out_dim"] = OUT_DIM
        model_core = FNO2dRefiner(
            modes1=config["model_fno"]["modes1"],
            modes2=config["model_fno"]["modes2"],
            width=config["model_fno"]["width"],
            in_dim=config["model_fno"]["in_dim"],
            out_dim=config["model_fno"]["out_dim"],
            pad=config["model_fno"]["pad"]
        )
    elif use_model.upper() == "CNO_TEMP":
        config["model_cno_temp"]["time_steps"] = config["refiner"]["k_max"]
        if config["model_cno_temp"]["in_dim"] != IN_DIM:
            config["model_cno_temp"]["in_dim"] = IN_DIM
        if config["model_cno_temp"]["out_dim"] != OUT_DIM:
            config["model_cno_temp"]["out_dim"] = OUT_DIM
        cfg = config["model_cno_temp"]
        model_core = CNO2d_Temporal(  
                in_dim               = cfg["in_dim"],                    
                in_size              = cfg["in_size"],                   
                N_layers             = cfg["N_layers"],                  
                N_res                = cfg["N_res"],                 
                N_res_neck           = cfg["N_res_neck"],            
                channel_multiplier   = cfg["channel_multiplier"],   
                conv_kernel          = cfg["conv_kernel"],             
                cutoff_den           = cfg["cutoff_den"],       
                filter_size          = cfg["filter_size"],             
                lrelu_upsampling     = cfg["lrelu_upsampling"],      
                half_width_mult      = cfg["half_width_mult"],    
                radial               = cfg["radial"],            
                batch_norm           = cfg["batch_norm"],         
                out_dim              = cfg["out_dim"],               
                out_size             = cfg["out_size"],              
                expand_input         = cfg["expand_input"],      
                latent_lift_proj_dim = cfg["latent_lift_proj_dim"], 
                add_inv              = cfg["add_inv"],            
                activation           = cfg["activation"],  
                is_att               = cfg["is_att"],
                patch_size           = cfg["patch_size"],
                dim_multiplier       = cfg["dim_multiplier"],
                depth                = cfg["depth"],
                heads                = cfg["heads"],
                dim_head_multiplier  = cfg["dim_head_multiplier"],
                mlp_dim_multiplier   = cfg["mlp_dim_multiplier"],
                emb_dropout          = cfg["emb_dropout"],
                time_steps           = cfg["time_steps"],
                is_time              = cfg["is_time"],
                nl_dim               = cfg["nl_dim"],
                )

    lit_model = PDERefiner(model_core, config)
    datamodule = PDEDataModule(config)

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    outdir = Path("checkpoints") / run_id
    outdir.mkdir(parents=True, exist_ok=True)

    ckpt_cb = ModelCheckpoint(
        dirpath=outdir / "checkpoints",
        filename="best_epoch{epoch:03d}-{val_rel_l2:.2f}",
        monitor="val_rel_l2",
        mode="min",
        save_top_k=1,
        save_last=True
    )
    lr_cb = LearningRateMonitor(logging_interval="epoch")

    patience = 350
    min_delta = 0.001
    early_cb = EarlyStopping(
        monitor="val_rel_l2",
        patience=patience,
        mode="min",
        min_delta=min_delta,
        verbose=True,
        check_finite=True,
    )

    logger = CSVLogger(save_dir=str(outdir), name="logs")

    if torch.cuda.is_available():
        print(f"[INFO] Using GPU: {torch.cuda.get_device_name()}")
        trainer = L.Trainer(
            max_epochs=config["training"]["epochs"],
            accelerator="gpu",
            devices="auto",
            precision="16-mixed" if torch.cuda.is_available() else "32-true",
            logger=logger,
            callbacks=[ckpt_cb, lr_cb, early_cb],
            log_every_n_steps=100
        )
    else:
        print("[INFO] Using CPU")
        trainer = L.Trainer(
            max_epochs=config["training"]["epochs"],
            accelerator="cpu",
            logger=logger,
            callbacks=[ckpt_cb, lr_cb, early_cb],
            log_every_n_steps=100,
            
        )

    trainer.fit(lit_model, datamodule=datamodule)

    # After training: print best ckpt and final metrics path
    print(f"[INFO] Best checkpoint: {ckpt_cb.best_model_path}")
    print(f"[INFO] Logs stored in: {logger.log_dir}")