import torch
import h5py
import yaml

from torch.utils.data import Dataset

class PDEDatasetLoader_Single(Dataset):
    def __init__(self, which, dtype=torch.float32, s=44, N=1, seq_len=1):
        super().__init__()

        self.N = N
        self.seq_len = seq_len
        self.s = s
        self.dtype = dtype

        with open("configs/default.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        if which == "train":
            self.reader = h5py.File(f"./{self.config['dataset']['train_file']}", 'r')
        elif which == "test":
            self.reader = h5py.File(f"./{self.config['dataset']['test_file']}", 'r')

        self.min_p = self.reader['min_q'][()]
        self.max_p = self.reader['max_q'][()]
        self.min_shift = self.reader['min_shift'][()]
        self.max_shift = self.reader['max_shift'][()]
        self.min_model = self.reader['min_t'][()]
        self.max_model = self.reader['max_t'][()]

        self.trajectories = [k for k in self.reader.keys() if k.startswith("trajectory_")]

        self.index_map = []

        for traj in self.trajectories:
            samples = [k for k in self.reader[traj].keys() if k.startswith("sample_")]
            num_samples = len(samples)
            max_start = num_samples - (self.N + 1) # Changed this # Prior: - self.seq_len
            for i in range(max_start):
                self.index_map.append((traj, i))

        print(f"Total usable samples: {len(self.index_map)}")

    def __len__(self):
        return len(self.index_map)
    
    def __getitem__(self, idx):
        traj_name, base_idx = self.index_map[idx]

        temp_bundle = []
        power_bundle = []
        shift_bundle = []
        future_bundle = []

        # --- Temperature ---
        temp = torch.from_numpy(self.reader[traj_name][f"sample_{base_idx}"]["output"][:]).float().reshape(self.s, self.s, 1)
        temp = (temp - self.min_model) / (self.max_model - self.min_model)
        temp_bundle.append(temp.permute(2, 0, 1))  # (1, H, W)
        

        for i in range(self.N):
            sample_idx = f"sample_{base_idx + i}"

            # --- Power ---
            input_p = torch.from_numpy(self.reader[traj_name][sample_idx]["input_p"][:]).float().reshape(self.s, self.s, 1)
            input_p = (input_p - self.min_p) / (self.max_p - self.min_p)
            power_bundle.append(input_p.permute(2, 0, 1))  # (1, H, W)

            # --- Shift/Direction ---
            dx = torch.from_numpy(self.reader[traj_name][sample_idx]["dx"][:]).float().squeeze(0)
            dx = (dx - self.min_shift) / (self.max_shift - self.min_shift)
            shift_bundle.append(dx.permute(2, 0, 1))  # (2, H, W)


        for i in range((self.N + 1)):
            sample_idx = f"sample_{base_idx + i}"
            # --- Ground truth after N*K steps ---
            output = torch.from_numpy(self.reader[traj_name][sample_idx]["output"][:]).float().reshape(self.s, self.s, 1)
            output = (output - self.min_model) / (self.max_model - self.min_model)
            future_bundle.append(output.permute(2, 0, 1))  # (1, s, s)

        temp_tensor = torch.cat(temp_bundle, dim=0)
        power_tensor = torch.cat(power_bundle, dim=0)
        shift_tensor = torch.stack(shift_bundle, dim=0)
        target_tensor = torch.cat(future_bundle, dim=0)

        return temp_tensor, power_tensor, shift_tensor, target_tensor

class PDEDatasetLoader_Multi(PDEDatasetLoader_Single):
    def __init__(self, which, dtype=torch.float32, s=44, N=1, seq_len=1):
        super().__init__(which, dtype, s, N, seq_len)

    def __getitem__(self, idx):
        # Get the 4 tensors produced by the parent class
        temp, power, shift, target = super().__getitem__(idx)
        # Shapes from your parent:
        # temp:   (1, s, s)
        # power:  (N, s, s)
        # shift:  (N, 2, s, s)
        # target: (N+1, 1, s, s)   (you built future_bundle with (1,s,s) per step)

        # Flatten sequence dims into channels
        power_c = power.reshape(-1, self.s, self.s)                       # (N, s, s)
        shift_c = shift.reshape(-1, self.s, self.s)                       # (2N, s, s)

        inp = torch.cat([temp, power_c, shift_c], dim=0)                    # (1+N+2N, s, s) = (1+3N, s, s)
        tgt = target[-1]                                                    # last future frame -> (1, s, s)

        return inp, tgt
    
    def get_norm(self):
        norm = (self.min_p, self.max_p,
                self.min_shift, self.max_shift,
                self.min_model, self.max_model)
        return norm
    
    def load_norm(self, norm):
        self.min_p = norm[0]
        self.max_p = norm[1]
        self.min_shift = norm[2]
        self.max_shift = norm[3]
        self.min_model = norm[4]
        self.max_model = norm[5]