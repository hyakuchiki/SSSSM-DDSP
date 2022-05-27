import numpy as np
import torch
from diffsynth.processor import Gen

def soft_clamp_min(x, min_v, T=100):
    x = torch.sigmoid((min_v-x)*T)*(min_v-x)+x
    return x

class LFO(Gen):
    def __init__(self, name='lfo', channels=1, rate_range=(1, 100), level_range=(0, 1), frame_rate=60, sample_rate=16000):
        super().__init__(name=name)
        self.channels = channels
        self.frame_rate = frame_rate
        self.sample_rate = sample_rate
        self.param_desc = {
            'rate':     {'size': self.channels, 'range': rate_range,  'type': 'sigmoid'},
            'level':    {'size': self.channels, 'range': level_range, 'type': 'sigmoid'}, 
            }
    
    def forward(self, rate, level, n_samples):
        """
        Args:
            rate (torch.Tensor): in Hz (batch, 1, self.channels)
            level (torch.Tensor): LFO level (batch, 1, self.channels)
            n_frames (int, optional): number of frames to generate. Defaults to None.

        Returns:
            torch.Tensor: lfo signal (batch_size, n_frames, self.channels)
        """
        n_secs = n_samples / self.sample_rate
        n_frames = int(self.frame_rate * n_secs)
        batch_size = rate.shape[0]
        final_phase = rate * n_secs * np.pi * 2
        x = torch.linspace(0, 1, n_frames, device=rate.device)[None, :, None].repeat(batch_size, 1, self.channels) # batch, n_frames, channels
        phase = x * final_phase
        wave = level * torch.sin(phase)
        return wave