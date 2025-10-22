import torch
import h5py
import yaml

from torch.utils.data import Dataset

class PDEDatasetLoader_Single(Dataset):
    def __init__(self, which="train", dtype=torch.float32, s=44, N=1, seq_len=1):
        super().__init__()

        self.N = N
        self.seq_len = seq_len
        self.s = s
        self.dtype = dtype

        with open("configs/default.yaml", "r") as f:
            self.config = yaml.safe_load(f)

        if which == "train":
            self.reader = h5py.File(f"./data/{self.config['dataset']['train_file']}", 'r')
        elif which == "test":
            self.reader = h5py.File(f"./data/{self.config['dataset']['test_file']}", 'r')

        self.min_p = self.reader['min_q'][()]
        self.max_p = self.reader['max_q'][()]
        self.min_shift = self.reader['min_shift'][()]
        self.max_shift = self.reader['max_shift'][()]
        self.min_model = self.reader['min_t'][()]
        self.max_model = self.reader['max_t'][()]

        self.trajectories = [k for k in self.reader.keys() if k.startswith("trajectory_")]

        # Old version
        # self.index_map = []
        # for traj in self.trajectories:
        #     samples = [k for k in self.reader[traj].keys() if k.startswith("sample_")]
        #     num_samples = len(samples)
        #     max_start = num_samples - (self.N + 1) # Changed this # Prior: - self.seq_len
        #     for i in range(max_start):
        #         self.index_map.append((traj, i))
        
        # New version
        self.index_map = []
        for traj in self.trajectories:
            samples = [k for k in self.reader[traj].keys() if k.startswith("sample_")]
            num_samples = len(samples)
            # base_idx must have N past (>= N) and N future (<= num_samples - N - 1)
            min_base = self.N
            max_base = num_samples - self.N - 1           # inclusive upper bound for base_idx
            for i in range(min_base, max_base + 1):
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
        
        # # Old code version
        # for i in range(self.N):
        #     sample_idx = f"sample_{base_idx + i}"

        #     # --- Power ---
        #     input_p = torch.from_numpy(self.reader[traj_name][sample_idx]["input_p"][:]).float().reshape(self.s, self.s, 1)
        #     input_p = (input_p - self.min_p) / (self.max_p - self.min_p)
        #     power_bundle.append(input_p.permute(2, 0, 1))  # (1, H, W)

        #     # --- Shift/Direction ---
        #     dx = torch.from_numpy(self.reader[traj_name][sample_idx]["dx"][:]).float().squeeze(0)
        #     dx = (dx - self.min_shift) / (self.max_shift - self.min_shift)
        #     shift_bundle.append(dx.permute(2, 0, 1))  # (2, H, W)


        # for i in range((self.N + 1)):
        #     sample_idx = f"sample_{base_idx + i}"
        #     # --- Ground truth after N*K steps ---
        #     output = torch.from_numpy(self.reader[traj_name][sample_idx]["output"][:]).float().reshape(self.s, self.s, 1)
        #     output = (output - self.min_model) / (self.max_model - self.min_model)
        #     future_bundle.append(output.permute(2, 0, 1))  # (1, s, s)

        # --- Past N controls: t = base_idx-N ... base_idx-1 ---
        for i in range(self.N):
            t = base_idx - self.N + i
            sample_idx = f"sample_{t}"

            # Power
            input_p = torch.from_numpy(self.reader[traj_name][sample_idx]["input_p"][:]) \
                        .float().reshape(self.s, self.s, 1)
            input_p = (input_p - self.min_p) / (self.max_p - self.min_p)
            power_bundle.append(input_p.permute(2, 0, 1))  # (1, H, W)

            # Shift/Direction (2 channels)
            dx = torch.from_numpy(self.reader[traj_name][sample_idx]["dx"][:]).float().squeeze(0)
            dx = (dx - self.min_shift) / (self.max_shift - self.min_shift)
            shift_bundle.append(dx.permute(2, 0, 1))       # (2, H, W)

        # --- Future N temperatures: t = base_idx+1 ... base_idx+N ---
        for j in range(1, self.N + 1):
            t = base_idx + j
            out = torch.from_numpy(self.reader[traj_name][f"sample_{t}"]["output"][:]) \
                    .float().reshape(self.s, self.s, 1)
            out = (out - self.min_model) / (self.max_model - self.min_model)
            future_bundle.append(out.permute(2, 0, 1))     # (1, H, W)

        temp_tensor = torch.cat(temp_bundle, dim=0)
        power_tensor = torch.cat(power_bundle, dim=0)
        shift_tensor = torch.stack(shift_bundle, dim=0)
        target_tensor = torch.cat(future_bundle, dim=0)

        return temp_tensor, power_tensor, shift_tensor, target_tensor

class PDEDatasetLoader_Multi(PDEDatasetLoader_Single):
    def __init__(self, which, dtype=torch.float32, s=44, N=1, seq_len=1, return_sequence=False):
        # if N > 1:
        #     super().__init__(which, dtype, s, N, seq_len=N,return_sequence=True)
        super().__init__(which, dtype, s, N, seq_len)
        self.return_sequence = return_sequence

    def __getitem__(self, idx):
        # Get the 4 tensors produced by the parent class
        temp, power, shift, target = super().__getitem__(idx)
        # Shapes from your parent:
        # temp:   (1, H, W)   at t = base
        # power:  (N, H, W)   past controls t = base-N ... base-1
        # shift:  (N, 2, H, W) past shifts
        # target: (N, 1, H, W) future temps t = base+1 ... base+N

        # Flatten sequence dims into channels
        power_c = power.reshape(-1, self.s, self.s)                       # (N, s, s)
        shift_c = shift.reshape(-1, self.s, self.s)                       # (2N, s, s)

        inp = torch.cat([temp, power_c, shift_c], dim=0)                  # (1+N+2N, s, s) = (1+3N, s, s)
        # Ich sehe ein Problem hier vllt Funktioniert der ganze spass nicht weil ich nur T ausgebe als inp aber vllt will ich t, t+1 und t+2 ausgeben
    
        # if self.return_sequence: # This is the version which works for my rollout function
        #     T = min(self.seq_len, self.N)
        #     tgt = target[1:1+T].contiguous()      # (T,1,H,W)
        if self.return_sequence: # This is the version which works for training with N
            T = min(self.seq_len, self.N)
            tgt = target[:T].squeeze(1).contiguous()      # (T,H,W)
        else:
            tgt = target[-1]                      # (1,H,W)
        return inp, tgt
    
    def get_norm(self):
        norm = (self.min_p, self.max_p,
                self.min_shift, self.max_shift,
                self.min_model, self.max_model)
        return norm
    
    def load_norm(self, norm):
        """Accepts (tuple) or (dict) and loads into this dataset."""
        if isinstance(norm, dict):
            self.min_p     = float(norm["min_p"])
            self.max_p     = float(norm["max_p"])
            self.min_shift = float(norm["min_shift"])
            self.max_shift = float(norm["max_shift"])
            self.min_model = float(norm["min_model"])
            self.max_model = float(norm["max_model"])
        else:
            # tuple/list: (min_p, max_p, min_shift, max_shift, min_model, max_model)
            self.min_p     = float(norm[0])
            self.max_p     = float(norm[1])
            self.min_shift = float(norm[2])
            self.max_shift = float(norm[3])
            self.min_model = float(norm[4])
            self.max_model = float(norm[5])