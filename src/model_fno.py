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


class FNO2d(nn.Module):
    def __init__(self, modes1, modes2, width, in_channels, out_channels=1):
        super().__init__()
        self.width = width
        self.lift  = nn.Conv2d(in_channels, width, 1)

        self.spect1 = SpectralConv2d(width, width, modes1, modes2)
        self.spect2 = SpectralConv2d(width, width, modes1, modes2)
        self.spect3 = SpectralConv2d(width, width, modes1, modes2)

        self.skip1  = nn.Conv2d(width, width, 1)
        self.skip2  = nn.Conv2d(width, width, 1)
        self.skip3  = nn.Conv2d(width, width, 1)

        self.proj   = nn.Conv2d(width, out_channels, 1)
        self.act    = nn.Tanh()

    def forward(self, x):
        # x: (B, C_in, H, W)  e.g. (B, 4, 44, 44)
        x = self.lift(x)                        # (B, width, H, W)
        x = self.act(self.spect1(x) + self.skip1(x))
        x = self.act(self.spect2(x) + self.skip2(x))
        x = self.act(self.spect3(x) + self.skip3(x))
        y = self.proj(x)                        # (B, 1, H, W)
        return y
    
class PadCropFNO(nn.Module):
    def __init__(self, core_fno: nn.Module, pad: int = 8):
        super().__init__()
        self.core = core_fno
        self.pad  = pad

    def forward(self, x):
        # x: (B, C, H, W)
        x = F.pad(x, (self.pad, self.pad, self.pad, self.pad), mode='reflect')
        y = self.core(x)
        return y[..., self.pad:-self.pad, self.pad:-self.pad]
