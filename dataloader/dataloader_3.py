import torch
import h5py
import yaml

from torch.utils.data import Dataset

class PDEDatasetLoader_Single(Dataset):
    """
    Single-step PDE dataset loader.

    Loads spatio-temporal samples of a thermal process with exogenous controls:
        • Endogenous variable: temperature field T(x,y,t)
        • Exogenous controls: laser power Q(x,y,t) and spatial shifts (dx, dy)

    Each sample is centered around a base index and includes:
        - Past N temperature fields ............................ (N, H, W)
          (endogenous input)
        - (N + 1) exogenous power fields Q ...................... (N+1, 1, H, W)
          (covering past N, current, and next-target step)
        - (N + 1) exogenous shift fields (dx, dy) ............... (N+1, 2, H, W)
          (same temporal span as Q)
        - Target: single future temperature field ............... (1, H, W)

    → Model input channels  = 4 N + 3  
       (N past T + (N+1) Q + 2 × (N+1) shifts → 4N + 3)

    → Model target channels = 1  
       (next-step temperature field)

    Normalization constants are read from the HDF5 file.
    """
    def __init__(self, which="train", dtype=torch.float32, s=44, N=1):
        super().__init__()

        self.N = N
        self.s = s
        self.dtype = dtype

        with open("configs/default.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        if which == "train":
            self.reader = h5py.File(f"./data/{self.config['dataset']['train_file']}", 'r')
        elif which == "test":
            self.reader = h5py.File(f"./data/{self.config['dataset']['test_file']}", 'r')

        # Load normalization constants
        self.min_p = self.reader['min_q'][()]
        self.max_p = self.reader['max_q'][()]
        self.min_shift = self.reader['min_shift'][()]
        self.max_shift = self.reader['max_shift'][()]
        self.min_model = self.reader['min_t'][()]
        self.max_model = self.reader['max_t'][()]

        # Load trajectory information
        self.trajectories = [k for k in self.reader.keys() if k.startswith("trajectory_")]
        
        # Build index map for usable samples
        self.index_map = []
        for traj in self.trajectories:
            samples = [k for k in self.reader[traj].keys() if k.startswith("sample_")]
            num_samples = len(samples)
            min_base = self.N
            max_base = num_samples - self.N - 1 
            print(f"[{which}] {traj}: num_samples={num_samples}, N={self.N}, usable_bases={max(0, max_base - min_base + 1)}")
            for i in range(min_base, max_base + 1):
                self.index_map.append((traj, i))

        print(f"Total usable samples: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx): 
        traj_name, base_idx = self.index_map[idx]

        temp_bundle = []
        target_bundle = []
        power_bundle = []
        shift_bundle = []

        # --- Past N controls: t = base_idx - N ... base_idx ---
        for i in range(self.N + 1):
            t = base_idx + i - self.N
            sample_idx = f"sample_{t}"

            # --- Power ---
            input_p = torch.from_numpy(self.reader[traj_name][sample_idx]["input_p"][:]) \
                        .float().reshape(self.s, self.s, 1)
            input_p = (input_p - self.min_p) / (self.max_p - self.min_p)
            power_bundle.append(input_p.permute(2, 0, 1))  # (1, H, W)

            # --- Shift/Direction (2 channels) ---
            dx = torch.from_numpy(self.reader[traj_name][sample_idx]["dx"][:]).float().squeeze(0)
            dx = (dx - self.min_shift) / (self.max_shift - self.min_shift)
            shift_bundle.append(dx.permute(2, 0, 1))       # (2, H, W)
        
        for i in range(self.N):
            sample_idx_past = f"sample_{base_idx - self.N + i}"

            # --- Temperature ---
            temp = torch.from_numpy(self.reader[traj_name][sample_idx_past]["output"][:]).float().reshape(self.s, self.s, 1)
            temp = (temp - self.min_model) / (self.max_model - self.min_model)
            temp_bundle.append(temp.permute(2, 0, 1))  # (1, H, W)

        # --- Target temperature at future time ---
        sample_idx_future = f"sample_{base_idx}"
        temp = torch.from_numpy(self.reader[traj_name][sample_idx_future]["output"][:]).float().reshape(self.s, self.s, 1)
        temp = (temp - self.min_model) / (self.max_model - self.min_model)
        target_bundle.append(temp.permute(2, 0, 1))  # (1, H, W)

        temp_tensor   = torch.cat(temp_bundle, dim=0)     # (N,H,W), list gets returned that u_prev is at index 0
        target_tensor = torch.cat(target_bundle, dim=0)   # (1,H,W)
        power_tensor  = torch.stack(power_bundle, dim=0)  # (N+1,1,H,W)
        shift_tensor  = torch.stack(shift_bundle, dim=0)  # (N+1,2,H,W)

        return temp_tensor, power_tensor, shift_tensor, target_tensor

class PDEDatasetLoader_Multi(PDEDatasetLoader_Single):
    def __init__(self, which, dtype=torch.float32, s=44, N=1, K=1, refiner_output=False):
        super().__init__(which, dtype, s, N)
        self.K = K
        self.N = N
        self.refiner_output = refiner_output

    def __getitem__(self, idx):
        xs, conds, ys = [], [], []

        for i in range(self.K):
            temp, power, shift, target = super().__getitem__(idx + i)
            # Shapes:
            # temp:  (N,   H, W)      → T_{-N} ... T_{-1}
            # power: (N+1,1, H, W)    → Q_{-N} ... Q_0
            # shift: (N+1,2, H, W)    → (dx,dy)_{-N} ... (dx,dy)_0
            # target: (1, H, W)       → T_{0} oder T_{+1}, je nach Definition

            if not self.refiner_output:
                # -----------------------------
                # Full concatenation mode:
                # Zeitlich sortiert:
                #   T_-N, Q_-N, dx_-N, dy_-N,
                #   ...
                #   T_-1, Q_-1, dx_-1, dy_-1,
                #   Q_0, dx_0, dy_0
                # -----------------------------
                x_channels = []

                # Vergangene N Schritte: t=-N ... -1
                for t in range(self.N):
                    # T_{t-N}
                    x_channels.append(temp[t:t+1, ...])            # (1,H,W)

                    # Q_{t-N}
                    x_channels.append(power[t, ...])               # (1,H,W)

                    # dx_{t-N}, dy_{t-N}
                    x_channels.append(shift[t, 0:1, ...])          # (1,H,W)
                    x_channels.append(shift[t, 1:2, ...])          # (1,H,W)

                # Aktueller Schritt t = 0: nur Q_0, dx_0, dy_0
                x_channels.append(power[self.N, ...])              # Q_0 (1,H,W)
                x_channels.append(shift[self.N, 0:1, ...])         # dx_0 (1,H,W)
                x_channels.append(shift[self.N, 1:2, ...])         # dy_0 (1,H,W)

                # (4N + 3, H, W)
                x_t = torch.cat(x_channels, dim=0)

                xs.append(x_t)
                ys.append(target)  # (1,H,W)

            else:
                # Refiner mode: wie gehabt, aber hier kannst du bei Bedarf
                # auch eine zeitliche Sortierung in cond einbauen
                temp_c = temp                     # (N,H,W)
                power_c = power.view(-1, self.s, self.s)   # (N+1, H, W) → flach
                shift_c = shift.view(-1, self.s, self.s)   # (2*(N+1),H,W)
                y_t = target

                x_t = temp_c
                cond_t = torch.cat([power_c, shift_c], dim=0)

                xs.append(x_t)
                ys.append(y_t)
                conds.append(cond_t)

        if self.K == 1:
            if not self.refiner_output:
                return xs[0], ys[0]
            else:
                return xs[0], ys[0], conds[0]

        # K > 1 → stack along time dimension
        x = torch.stack(xs, dim=0)
        y = torch.stack(ys, dim=0)

        if not self.refiner_output:
            return x, y
        else:
            cond = torch.stack(conds, dim=0)
            return x, y, cond
