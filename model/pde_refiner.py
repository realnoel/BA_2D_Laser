import torch
import torch.nn as nn

from random import randint

class PDERefiner(nn.Module):
    def __init__(self, model: nn.Module, K: int, min_noise_std: float):
        super().__init__()
        self.K = K
        self.min_noise_std = float(min_noise_std)
        self.neural_operator = model

    def forward(self, u_t, u_prev):
        for k in range(1, self.K + 1):
            noise_std = self.min_noise_std ** (k / self.K)
            noise = torch.randn_like(u_t)
            u_t_noised = u_t + noise * noise_std
            pred_noise = self.neural_operator(u_t_noised, u_prev, k)
            u_t = u_t_noised - pred_noise * noise_std
        return u_t
    
    def training_loss(self, u_t, u_prev, k):
        noise_std = self.min_noise_std ** (k / self.K)
        noise = torch.randn_like(u_t)
        u_t_noised = u_t + noise * noise_std
        pred = self.neural_operator(u_t_noised, u_prev, k)
        target = noise
        return torch.mean((pred - target) ** 2)

