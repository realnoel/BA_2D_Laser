import torch
import numpy as np
import time
import yaml

from torch.utils.data import DataLoader, random_split
from torch.optim import Adam, LBFGS
from torch.optim.lr_scheduler import StepLR
from torch.nn import MSELoss

from dataloader import PDEDatasetLoader_Multi
from model_fno import FNO2d, PadCropFNO
from utils import save_temperature_plot

def train(fno, should_evaluate=False):
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fno.to(device)

    start_time = time.time()
    torch.manual_seed(config["training"]["seed"])
    np.random.seed(config["training"]["seed"])
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
    validation_dataset = PDEDatasetLoader_Multi(which="test")

    # Loads training normalization values into validation dataset
    validation_dataset.min_p, validation_dataset.max_p, \
    validation_dataset.min_shift, validation_dataset.max_shift, \
    validation_dataset.min_model, validation_dataset.max_model = norm

    batch_size = config["training"]["batch_size"]
    # modes = config["model"]["modes"]
    # width = config["model"]["width"]
    # Create data loaders for batching
    training_set = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    testing_set = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Initialize model, optimizer, and loss function
    # fno = FNO1d(modes=modes, width=width) # Old width is 64
    optimizer = Adam(fno.parameters(), lr=config["optimizer"]["lr"]) # Adam is first-order gradient 
    # optimizer = torch.optim.LBFGS([x],lr=0.05) # LBFGS is quasi Newton, might be good for PDE solvers
    scheduler = StepLR(optimizer, step_size=config["optimizer"]["step_size"], gamma=config["optimizer"]["gamma"])
    l = MSELoss()
    freq_print = 1

    epochs = config["training"]["epochs"]
    for epoch in range(epochs):
        train_mse = 0.0
        fno.train()
        for (x, y) in training_set:
            x, y = x.to(device), y.to(device)
            # Check dim inp, out batch
            if y.ndim == 3:            # (B, H, W)
                y = y.unsqueeze(1)     # -> (B, 1, H, W)
            optimizer.zero_grad()
            y_pred = fno(x) # Removed .squeeze(2)
            loss_f = l(y_pred, y)
            loss_f.backward()
            optimizer.step()
            train_mse += loss_f.item()
        train_mse /= len(training_set)

        scheduler.step()

        with torch.no_grad():
            fno.eval()
            test_relative_l2 = 0.0
            for (x, y) in testing_set:
                x, y = x.to(device), y.to(device)
                y_pred = fno(x)

                if y.ndim == 3:            # (B, H, W)
                    y = y.unsqueeze(1)     # -> (B, 1, H, W)

                # Save prediction and ground truth of first batch element of last epoch
                save_temperature_plot(y_pred[0, 0], name_prefix="prediction", epoch=epoch)
                save_temperature_plot(y[0, 0], name_prefix="groundtruth", epoch=epoch)

                loss_f = (torch.mean((y_pred - y) ** 2) / torch.mean(y ** 2)) ** 0.5 * 100
                test_relative_l2 += loss_f.item()
            test_relative_l2 /= len(testing_set)

        if epoch % freq_print == 0: print("######### Epoch:", epoch, " ######### Train Loss:", train_mse, " ######### Relative L2 Test Norm:", test_relative_l2)

    # Final evaluation on validation dataset
    if should_evaluate:
        end_time = time.time()
        print("### Training time:", end_time - start_time, " ###")
        print("### Using modes:", config["model"]["modes"], "and width:", config["model"]["width"], "with batch size:", batch_size, " ###")

with open("configs/default.yaml", "r") as f:
    config = yaml.safe_load(f)

if __name__ == "__main__":
    modes = config["model"]["modes"]
    width = config["model"]["width"]
    fno = PadCropFNO(FNO2d(modes1=24, modes2=24, width=16, in_channels=4), pad=8) # Width is 4
    train(fno, should_evaluate=True)

