#
# Fourier Neural Operator in 2D, modified from: camlab-ethz Tutorial 5 Operator Learing - Fourier Neural Operator
#
import torch
import torch.nn as nn
import torch.nn.functional as F

class SpectralConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, modes1, modes2):
        super().__init__()
        self.in_channels  = in_channels
        self.out_channels = out_channels
        self.modes1 = modes1
        self.modes2 = modes2
        scale = 1.0 / (in_channels * out_channels)
        # Two sets of complex weights: top-left and bottom-left bands
        self.weights1 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))
        self.weights2 = nn.Parameter(scale * torch.randn(in_channels, out_channels, modes1, modes2, dtype=torch.cfloat))

    def forward(self, x):
        # x: (B, C, H, W)
        B, C, H, W = x.shape
        x_ft = torch.fft.rfft2(x)  # (B, C, H, W//2 + 1), complex
        Hm = min(self.modes1, H)
        Wm = min(self.modes2, W//2 + 1)

        out_ft = torch.zeros(B, self.out_channels, H, W//2 + 1, device=x.device, dtype=torch.cfloat)

        # Top-left low frequencies
        out_ft[:, :, :Hm, :Wm] = torch.einsum(
            'bchw,cohw->bohw',
            x_ft[:, :, :Hm, :Wm],
            self.weights1[:, :, :Hm, :Wm],
        )
        # Bottom-left low frequencies (negative ky)
        out_ft[:, :, -Hm:, :Wm] = torch.einsum(
            'bchw,cohw->bohw',
            x_ft[:, :, -Hm:, :Wm],
            self.weights2[:, :, :Hm, :Wm],
        )

        x = torch.fft.irfft2(out_ft, s=(H, W))  # back to real space
        return x


class FNO2dRefiner(nn.Module):
    def __init__(self, modes1, modes2, width, in_dim, out_dim=1, pad=8):
        super().__init__()
        self.width = width
        self.pad = pad
        self.lift  = nn.Conv2d(in_dim, width, 1)

        self.spect1 = SpectralConv2d(width, width, modes1, modes2)
        self.spect2 = SpectralConv2d(width, width, modes1, modes2)
        self.spect3 = SpectralConv2d(width, width, modes1, modes2)

        self.skip1  = nn.Conv2d(width, width, 1)
        self.skip2  = nn.Conv2d(width, width, 1)
        self.skip3  = nn.Conv2d(width, width, 1)

        self.proj   = nn.Conv2d(width, out_dim, 1)
        self.act    = nn.SiLU()

        # --- NEW: embedding for refinement step k ---
        self.k_embed = nn.Sequential(
            nn.Linear(1, width),
            nn.SiLU(),
            nn.Linear(width, width)
        )

    def forward(self, u_in, u_prev, k: int):
        """
        u_in  : (B, 1, H, W)    -> current or noised field
        u_prev: (B, 1, H, W)    -> previous state field
        k     : scalar or tensor (int/float) refinement index
        """
        # Combine spatial inputs along channel dimension
        x = torch.cat([u_in, u_prev], dim=1)  # (B, 2, H, W)

        # --- Embed k ---
        if not torch.is_tensor(k):
            # make sure k is tensor on the right device and dtype
            k = torch.tensor([float(k)], dtype=x.dtype, device=x.device)
        elif k.ndim == 0:
            # scalar tensor -> add batch dimension
            k = k.unsqueeze(0)

        # repeat for all samples in the batch
        k = k.repeat(x.shape[0])  # shape: (B,)

        # embedding layer: converts scalar step → feature vector of size `width`
        k_embed = self.k_embed(k[:, None]).view(x.shape[0], self.width, 1, 1)

        # Lift to feature space
        x = self.lift(x)                      # (B, width, H, W)
        x = x + k_embed  # k_embed in den Fourier Space bringen                     # inject k-conditioning

        # Pad input
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        # Spectral layers
        x = self.act(self.spect1(x) + self.skip1(x))
        x = self.act(self.spect2(x) + self.skip2(x))
        x = self.act(self.spect3(x) + self.skip3(x))
        # Remove padding
        x = x[..., self.pad:-self.pad, self.pad:-self.pad]
        y = self.proj(x)                        # (B, 1, H, W)
        return y