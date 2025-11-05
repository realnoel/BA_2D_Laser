import torch
import h5py
import yaml

from torch.utils.data import Dataset

class PDEDatasetLoader_Single(Dataset):
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

        self.min_p = self.reader['min_q'][()]
        self.max_p = self.reader['max_q'][()]
        self.min_shift = self.reader['min_shift'][()]
        self.max_shift = self.reader['max_shift'][()]
        self.min_model = self.reader['min_t'][()]
        self.max_model = self.reader['max_t'][()]

        self.trajectories = [k for k in self.reader.keys() if k.startswith("trajectory_")]
        
        # New version
        self.index_map = []
        for traj in self.trajectories:
            samples = [k for k in self.reader[traj].keys() if k.startswith("sample_")]
            num_samples = len(samples)
            # base_idx must have N past (>= N) and N future (<= num_samples - N - 1)
            min_base = self.N
            max_base = num_samples - self.N - 1           # inclusive upper bound for base_idx
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

        # --- Past N controls: t = base_idx ... base_idx-N-1 ---
        # For E1/E2 use self.N input_p and dx
        # For E3/E4 use 2*self.N input_p and dx and 
        # 
        for i in range(self.N):
            t = base_idx + i
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
            sample_idx_past   = f"sample_{base_idx - self.N + i}"
            sample_idx_future = f"sample_{base_idx + i}"

            # --- Temperature ---
            temp = torch.from_numpy(self.reader[traj_name][sample_idx_past]["output"][:]).float().reshape(self.s, self.s, 1)
            temp = (temp - self.min_model) / (self.max_model - self.min_model)
            temp_bundle.append(temp.permute(2, 0, 1))  # (1, H, W)

            temp = torch.from_numpy(self.reader[traj_name][sample_idx_future]["output"][:]).float().reshape(self.s, self.s, 1)
            temp = (temp - self.min_model) / (self.max_model - self.min_model)
            target_bundle.append(temp.permute(2, 0, 1))  # (1, H, W)

        temp_tensor   = torch.cat(temp_bundle, dim=0)     # (N,H,W)
        target_tensor = torch.cat(target_bundle, dim=0)   # (N,H,W)
        power_tensor  = torch.stack(power_bundle, dim=0)  # (N+1,1,H,W)
        shift_tensor  = torch.stack(shift_bundle, dim=0)  # (N+1,2,H,W)

        # print(f"Dataset __getitem__ idx={idx}: temp shape: {temp_tensor.shape}, power shape: {power_tensor.shape}, shift shape: {shift_tensor.shape}, target shape: {target_tensor.shape}")

        return temp_tensor, power_tensor, shift_tensor, target_tensor

class PDEDatasetLoader_Multi(PDEDatasetLoader_Single):
    def __init__(self, which, dtype=torch.float32, s=44, N=1, K=1):
        super().__init__(which, dtype, s, N)
        self.K = K
        self.N = N 
    
    def __getitem__(self, idx):
        """
        Returns a sequence of length self.K.
        Each time step packs N exogenous frames -> inp_t: (4N+3,H,W), tgt_t: (N,H,W)
        Output shapes:
            # Parent:
            #   temp:   (N,H,W)           @ base-N .. base-1
            #   power:  (N+1,1,H,W)       @ base-N .. base
            #   shift:  (N+1,2,H,W)       @ base-N .. base
            #   target: (N,H,W)           @ base   .. base+N-1
            #
            # Flattened per time step:
            #   temp_c  : 0        → N-1
            #   power_c : N        → 2N
            #   shift_c : 2N+1     → 4N+2
            #   total channels = 4N+3
            #
            # Final stacked sequence:
            #   seq_inp : (K, 4N+3, H, W)
            #   seq_tgt : (K, N, H, W)
        """
        inp_list, tgt_list = [], []

        # if idx + self.N * (self.K - 1) >= len(self.index_map):
        #     raise IndexError(f"Requested K={self.K} exceeds available sequence length for N={self.N}. "
        #                     f"Max allowed K is {(len(self.index_map) // self.N)}.")

        if self.K > 1:
            for i in range(self.K):
                temp, power, shift, target = super().__getitem__(idx + i)

                temp_c = temp                                # (N,H,W)
                power_c = power.reshape(-1, self.s, self.s)  # (N+1,H,W)
                shift_c = shift.reshape(-1, self.s, self.s)  # (2*(N+1),H,W)

                inp_t = torch.cat([temp_c, power_c, shift_c], dim=0)  # (4N+3,H,W)
                # inp_t - > temp_c is [0:N], power_c is [N:2N+1], shift_c is [2N+1:4N+3]
                tgt_t = target                                        # (N,H,W)
                inp_list.append(inp_t)
                tgt_list.append(tgt_t)

            seq_inp = torch.stack(inp_list, dim=0)  # (K, 4N+3, H, W)
            seq_tgt = torch.stack(tgt_list, dim=0)  # (K, N, H, W)
            # print(f"Dataset __getitem__ idx={idx}: seq_inp shape: {seq_inp.shape}, seq_tgt shape: {seq_tgt.shape}")
            return seq_inp, seq_tgt
        
        elif self.K == 1:
            temp, power, shift, target = super().__getitem__(idx)
            
            temp_c = temp                                # (N,H,W)
            power_c = power.reshape(-1, self.s, self.s)  # (N+1,H,W)
            shift_c = shift.reshape(-1, self.s, self.s)  # (2*(N+1),H,W)

            inp_t = torch.cat([temp_c, power_c, shift_c], dim=0)  # (4N+3,H,W)
            # inp_t - > temp_c is [0:N], power_c is [N:2N+1], shift_c is [2N+1:4N+3]
            tgt_t = target                                        # (N,H,W)
            # print(f"Dataset __getitem__ idx={idx}: inp shape: {inp.shape}, tgt shape: {tgt.shape}")
            return inp_t, tgt_t
        else:
            raise ValueError(f"Invalid K: {self.K}")

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