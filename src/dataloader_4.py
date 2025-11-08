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
          (endogenous input window)
        - Next-step exogenous control fields ................... (3, H, W)
          consisting of:
              Q(x,y,t+1)       → 1 channel
              dx(x,y,t+1), dy(x,y,t+1) → 2 channels
        - Target: temperature at the next time step ............ (1, H, W)

    → Model input channels  =  N + 3
       (N past T  +  Q, dx, dy  for next step)

    → Model target channels =  1
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

        # --- Past N controls: t = base_idx ---
        for i in range(1):
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

        temp_tensor   = torch.cat(temp_bundle, dim=0)     # (N,H,W)
        target_tensor = torch.cat(target_bundle, dim=0)   # (1,H,W)
        power_tensor  = torch.stack(power_bundle, dim=0)  # (1,H,W)
        shift_tensor  = torch.stack(shift_bundle, dim=0)  # (2,H,W)

        return temp_tensor, power_tensor, shift_tensor, target_tensor

class PDEDatasetLoader_Multi(PDEDatasetLoader_Single):
    def __init__(self, which, dtype=torch.float32, s=44, N=1, K=1):
        super().__init__(which, dtype, s, N)
        self.K = K
        self.N = N 
    
    def __getitem__(self, idx):
        """
        Multi-step PDE dataset loader.

        Returns sequences of length K (K consecutive prediction steps).

        For each step, the model input concatenates:
            - N past temperature fields ............................ (N, H, W)
            - 3 exogenous control fields for next step ............. (3, H, W)
            (Q, dx, dy)

        → total input channels  =  N + 3
        (N past T  +  3 exogenous next-step controls)

        → target channels       =  1
        (future temperature)

        Shapes:
            seq_inp : (K, N+3, H, W)
            seq_tgt : (K, 1,   H, W)
        """
        inp_list, tgt_list = [], []

        if self.K > 1:
            for i in range(self.K):
                temp, power, shift, target = super().__getitem__(idx + i)

                temp_c = temp                                
                power_c = power.reshape(-1, self.s, self.s) 
                shift_c = shift.reshape(-1, self.s, self.s)  

                inp_t = torch.cat([temp_c, power_c, shift_c], dim=0) 
                tgt_t = target                                       
                inp_list.append(inp_t)
                tgt_list.append(tgt_t)

            seq_inp = torch.stack(inp_list, dim=0)  
            seq_tgt = torch.stack(tgt_list, dim=0)  
            return seq_inp, seq_tgt # (K, N+3, H, W), (K, 1, H, W)
        
        elif self.K == 1:
            temp, power, shift, target = super().__getitem__(idx)
            
            temp_c = temp                                
            power_c = power.reshape(-1, self.s, self.s)  
            shift_c = shift.reshape(-1, self.s, self.s)  

            inp_t = torch.cat([temp_c, power_c, shift_c], dim=0)  
            tgt_t = target                                      

            return inp_t, tgt_t # (N+3, H, W), (1, H, W)
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
            self.min_p     = float(norm[0])
            self.max_p     = float(norm[1])
            self.min_shift = float(norm[2])
            self.max_shift = float(norm[3])
            self.min_model = float(norm[4])
            self.max_model = float(norm[5])