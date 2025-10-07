import torch
import numpy as np
import time
import yaml
from datetime import datetime

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss

from dataloader import PDEDatasetLoader_Multi
from model_fno import FNO2d
from model_cno import CNO2d
from utils import save_temperature_plot
from utils_train import save_checkpoint, load_checkpoint, now_run_id

def train(model, should_evaluate=False):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    start_time = time.time()
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])

    run_id = now_run_id()
    print(f"[INFO] Checkpoints will be saved under checkpoints/{run_id}")

    # Load datasets
    full_dataset = PDEDatasetLoader_Multi(which="train")
    # print("Training dataset shape:", full_dataset.U.shape)

    n_total = len(full_dataset)
    n_test = int(config["training"]["test_split"] * n_total)
    n_train = n_total - n_test

    norm = (full_dataset.min_p, full_dataset.max_p,
            full_dataset.min_shift, full_dataset.max_shift,
            full_dataset.min_model, full_dataset.max_model)

    train_dataset, test_dataset = random_split(full_dataset, 
                                               [n_train, n_test],
                                               generator=torch.Generator().manual_seed(config["training"]["seed"])
                                               )

    # Loads validation dataset for final evaluation
    # validation_dataset = PDEDatasetLoader_Multi(which="test")

    batch_size = config["training"]["batch_size"]

    # Create data loaders for batching
    training_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testing_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    # fno = FNO1d(modes=modes, width=width) # Old width is 64
    optimizer = Adam(model.parameters(), lr=config["optimizer"]["lr"]) # Adam is first-order gradient 
    # optimizer = torch.optim.LBFGS([x],lr=0.05) # LBFGS is quasi Newton, might be good for PDE solvers
    scheduler = StepLR(optimizer, step_size=config["optimizer"]["step_size"], gamma=config["optimizer"]["gamma"])
    l = MSELoss()
    freq_print = 1

    best_val = float("inf")
    epochs = config["training"]["epochs"]


    for epoch in range(epochs):
        train_mse = 0.0
        model.train()
        if epoch == 1:
            print(f"Training for {epochs} epochs will take approx. {(time.time() - start_time) * epochs / 60:.2f} minutes")
        for (x, y) in training_set:
            x, y = x.to(device), y.to(device)
            # Check dim inp, out batch
            if y.ndim == 3:            # (B, H, W)
                y = y.unsqueeze(1)     # -> (B, 1, H, W)
            optimizer.zero_grad()
            y_pred = model(x) # Removed .squeeze(2)
            loss_f = l(y_pred, y)
            loss_f.backward()
            optimizer.step()
            train_mse += loss_f.item()
        train_mse /= len(training_set)

        scheduler.step()

        with torch.no_grad():
            model.eval()
            test_relative_l2 = 0.0
            for (x, y) in testing_set:
                x, y = x.to(device), y.to(device)
                y_pred = model(x)

                if y.ndim == 3:            # (B, H, W)
                    y = y.unsqueeze(1)     # -> (B, 1, H, W)

                # Save prediction and ground truth of first batch element of last epoch
                # save_temperature_plot(y_pred[0, 0], path=f"results/{timestamp}",name_prefix="prediction", epoch=epoch)
                # save_temperature_plot(y[0, 0], path=f"results/{timestamp}", name_prefix="groundtruth", epoch=epoch)

                loss_f = (torch.mean((y_pred - y) ** 2) / torch.mean(y ** 2)) ** 0.5 * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(testing_set)

        if epoch % freq_print == 0: 
            print("######### Epoch:", f"{epoch}", " ######### Train Loss:", f"{train_mse:.4e}", \
                   "######### Relative L2 Test Norm:", f"{test_relative_l2:.2f}%", \
                   f"###### {time.time() - start_time:.2f}s ######")

        # ---- Checkpoint ----
        is_best = test_relative_l2 < best_val
        if is_best:
            best_val = test_relative_l2

        save_checkpoint(
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            epoch=epoch,
            val_metric=test_relative_l2,
            config=config,
            norm_tuple=norm,
            run_id=run_id,
            is_best=is_best
        )

    # Final evaluation on validation dataset
    if should_evaluate:
        end_time = time.time()
        print(f"### Training time: {end_time - start_time:.2f} ###")
        print(f"### With batch size: {batch_size} ###")

with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    # model = FNO2d(modes1=config["model_fno"]["modes1"], 
    #             modes2=config["model_fno"]["modes2"], 
    #             width=config["model_fno"]["width"], 
    #             in_channels=config["model_fno"]["in_channels"], 
    #             out_channels=config["model_fno"]["out_channels"], 
    #             pad=config["model_fno"]["pad"]
    #             )
    model = CNO2d(in_dim=config["model_cno"]["in_dim"],
                  out_dim=config["model_cno"]["out_dim"],
                  size=config["model_cno"]["size"],
                  N_layers=config["model_cno"]["N_layers"],
                  N_res=config["model_cno"]["N_res"],
                  N_res_neck=config["model_cno"]["N_res_neck"],
                  channel_multiplier=config["model_cno"]["channel_multiplier"],
                )
    train(model, should_evaluate=True)



