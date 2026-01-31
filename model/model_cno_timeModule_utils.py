import numpy as np
import scipy.signal
import torch

from utils.torch_utils import misc
from utils.torch_utils import persistence
from utils.torch_utils.ops import filtered_lrelu

import torch.nn as nn

@persistence.persistent_class
class LReLu(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
        in_cutoff,                      # Input cutoff frequency (f_c).
        out_cutoff,                     # Output cutoff frequency (f_c).
        in_half_width,                  # Input  transition band half-width (f_h).
        out_half_width,                 # Output Transition band half-width (f_h).

        # Hyperparameters.
        filter_size         = 6,        # Low-pass filter size relative to the lower resolution when up/downsampling.
        lrelu_upsampling    = 2,        # Relative sampling rate for leaky ReLU. Ignored for final the ToRGB layer.
        
        is_critically_sampled = False,  # Does this layer use critical sampling?  #NOT IMPORTANT FOR CNO.
        use_radial_filters    = False,  # Use radially symmetric downsampling filter?
    ):
        super().__init__()
        
        
        self.is_critically_sampled = is_critically_sampled

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = np.broadcast_to(np.asarray(in_size), [2])
        self.out_size = np.broadcast_to(np.asarray(out_size), [2])
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate
        self.tmp_sampling_rate = max(in_sampling_rate, out_sampling_rate) *lrelu_upsampling
        self.in_cutoff = in_cutoff
        self.out_cutoff = out_cutoff
        self.in_half_width = in_half_width
        self.out_half_width = out_half_width
        
        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))

        # Design upsampling filter.
        self.up_factor = int(np.rint(self.tmp_sampling_rate / self.in_sampling_rate))
        self.up_taps = filter_size * self.up_factor if self.up_factor > 1  else 1
        
        self.register_buffer('up_filter', self.design_lowpass_filter(
            numtaps=self.up_taps, cutoff=self.in_cutoff, width=self.in_half_width*2, fs=self.tmp_sampling_rate))

        # Design downsampling filter.
        self.down_factor = int(np.rint(self.tmp_sampling_rate / self.out_sampling_rate))
        self.down_taps = filter_size * self.down_factor if self.down_factor > 1 else 1
        self.down_radial = use_radial_filters and not self.is_critically_sampled
        self.register_buffer('down_filter', self.design_lowpass_filter(
            numtaps=self.down_taps, cutoff=self.out_cutoff, width=self.out_half_width*2, fs=self.tmp_sampling_rate, radial=self.down_radial))

            
        # Compute padding. ------------------------------------------------------------------------------
        
        pad_total = (self.out_size - 1) * self.down_factor + 1 # Desired output size before downsampling.
        pad_total -= (self.in_size * self.up_factor) # Input size after upsampling.
        pad_total += self.up_taps + self.down_taps - 2 # Size reduction caused by the filters.
                
        pad_lo = (pad_total + self.up_factor) // 2 # Shift sample locations according to the symmetric interpretation (Appendix C.3).
        pad_hi = pad_total - pad_lo
        self.padding = [int(pad_lo[0]), int(pad_hi[0]), int(pad_lo[1]), int(pad_hi[1])]
            
        #------------------------------------------------------------------------------------------------

    def forward(self, x, noise_mode='random', force_fp32=False, update_emas=False):
       
        dtype = torch.float32

        # Execute bias, filtered leaky ReLU, and clamping.
        gain = np.sqrt(2)
        slope = 0.2
        x = filtered_lrelu.filtered_lrelu(x=x, fu=self.up_filter, fd=self.down_filter, b=self.bias.to(x.dtype),
            up=self.up_factor, down=self.down_factor, padding=self.padding, gain=gain, slope=slope, clamp=None)


        # Ensure correct shape and dtype.
        misc.assert_shape(x, [None, self.out_channels, int(self.out_size[1]), int(self.out_size[0])])
        assert x.dtype == dtype
        return x

    @staticmethod
    def design_lowpass_filter(numtaps, cutoff, width, fs, radial=False):
        assert numtaps >= 1

        # Identity filter.
        if numtaps == 1:
            return None

        # Separable Kaiser low-pass filter.
        if not radial:
            f = scipy.signal.firwin(numtaps=numtaps, cutoff=cutoff, width=width, fs=fs)
            return torch.as_tensor(f, dtype=torch.float32)

        # Radially symmetric jinc-based filter.
        x = (np.arange(numtaps) - (numtaps - 1) / 2) / fs
        r = np.hypot(*np.meshgrid(x, x))
        f = scipy.special.j1(2 * cutoff * (np.pi * r)) / (np.pi * r)
        beta = scipy.signal.kaiser_beta(scipy.signal.kaiser_atten(numtaps, width / (fs / 2)))
        w = np.kaiser(numtaps, beta)
        f *= np.outer(w, w)
        f /= np.sum(f)
        return torch.as_tensor(f, dtype=torch.float32)

    def extra_repr(self):
        return '\n'.join([
            f'w_dim={self.w_dim:d}, is_torgb={self.is_torgb},',
            f'is_critically_sampled={self.is_critically_sampled}, use_fp16={self.use_fp16},',
            f'in_sampling_rate={self.in_sampling_rate:g}, out_sampling_rate={self.out_sampling_rate:g},',
            f'in_cutoff={self.in_cutoff:g}, out_cutoff={self.out_cutoff:g},',
            f'in_half_width={self.in_half_width:g}, out_half_width={self.out_half_width:g},',
            f'in_size={list(self.in_size)}, out_size={list(self.out_size)},',
            f'in_channels={self.in_channels:d}, out_channels={self.out_channels:d}'])

#----------------------------------------------------------------------------
#----------------------------------------------------------------------------

class LReLu_standard(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
    ):
        super().__init__()
        
        
        self.activation = nn.LeakyReLU() 
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.in_size = in_size
        self.out_size = out_size
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate

                    
        #------------------------------------------------------------------------------------------------

    def forward(self, x):
        
        if self.in_sampling_rate == 2*self.out_sampling_rate:
            return nn.AvgPool2d(2, stride=2, padding=0)(self.activation(x))
        elif self.in_sampling_rate == 4*self.out_sampling_rate:
            return nn.AvgPool2d(4, stride=4, padding=1)(self.activation(x))
        else:
            return nn.functional.interpolate(self.activation(x), size=self.out_size)

class LReLu_torch(torch.nn.Module):
    def __init__(self,
        in_channels,                    # Number of input channels.
        out_channels,                   # Number of output channels.
        in_size,                        # Input spatial size: int or [width, height].
        out_size,                       # Output spatial size: int or [width, height].
        in_sampling_rate,               # Input sampling rate (s).
        out_sampling_rate,              # Output sampling rate (s).
    ):
        super().__init__()
        
        
        self.activation = nn.LeakyReLU() 
        
        self.in_channels = in_channels
        self.out_channels = out_channels
        
        assert in_channels == out_channels
        
        self.in_size = in_size
        self.out_size = out_size
        self.in_sampling_rate = in_sampling_rate
        self.out_sampling_rate = out_sampling_rate

        self.bias = torch.nn.Parameter(torch.zeros([self.out_channels]))
        

        #------------------------------------------------------------------------------------------------

    def forward(self, x):
        
        x = nn.functional.interpolate(x, size = 2*self.in_size,mode='bicubic', antialias = True)
        x = self.activation(x)
        x = nn.functional.interpolate(x, size = self.in_size,mode='bicubic', antialias = True)
        x = nn.functional.interpolate(x, size = self.out_size,mode='bicubic', antialias = True)

        x = x.permute(0,2,3,1)
        x = torch.add(x, torch.broadcast_to(self.bias, x.shape))
        x = x.permute(0,3,1,2)
        return x