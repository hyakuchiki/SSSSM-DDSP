import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
import librosa
from torchaudio.transforms import MelScale
from torchaudio.functional import create_dct
from diffsynth.util import log_eps, pad_or_trim_to_expected_length

amp = lambda x: x[...,0]**2 + x[...,1]**2

class MelSpec(nn.Module):
    def __init__(self, n_fft=2048, hop_length=1024, n_mels=128, sample_rate=16000, power=1, f_min=40, f_max=7600, pad_end=True, center=False):
        """
        
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.f_min = f_min
        self.f_max = f_max
        self.sample_rate = sample_rate
        self.n_mels = n_mels
        self.pad_end = pad_end
        self.center = center
        self.mel_scale = MelScale(self.n_mels, self.sample_rate, self.f_min, self.f_max, self.n_fft // 2 + 1)
    
    def forward(self, audio):
        if self.pad_end:
            _batch_dim, l_x = audio.shape
            remainder = (l_x - self.n_fft) % self.hop_length
            pad = 0 if (remainder == 0) else self.hop_length - remainder
            audio = F.pad(audio, (0, pad), 'constant')
        spec = spectrogram(audio, self.n_fft, self.hop_length, self.power, self.center)
        mel_spec = self.mel_scale(spec)
        return mel_spec

class Spec(nn.Module):
    def __init__(self, n_fft=2048, hop_length=1024, power=2, pad_end=True, center=False):
        """
        
        """
        super().__init__()
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.power = power
        self.pad_end = pad_end
        self.center = center
    
    def forward(self, audio):
        if self.pad_end:
            _batch_dim, l_x = audio.shape
            remainder = (l_x - self.n_fft) % self.hop_length
            pad = 0 if (remainder == 0) else self.hop_length - remainder
            audio = F.pad(audio, (0, pad), 'constant')
        spec = spectrogram(audio, self.n_fft, self.hop_length, self.power, self.center)
        return spec

class Mfcc(nn.Module):
    def __init__(self, n_fft=2048, hop_length=1024, n_mels=128, n_mfcc=40, norm='ortho', sample_rate=16000, f_min=40, f_max=7600, pad_end=True, center=False):
        """
        uses log mels
        """
        super().__init__()
        self.norm = norm
        self.n_mfcc = n_mfcc
        self.melspec = MelSpec(n_fft, hop_length, n_mels, sample_rate, power=2, f_min=f_min, f_max=f_max, pad_end=pad_end, center=center)
        dct_mat = create_dct(self.n_mfcc, self.melspec.n_mels, self.norm)
        self.register_buffer('dct_mat', dct_mat)

    def forward(self, audio):
        mel_spec = self.melspec(audio)
        mel_spec = torch.log(mel_spec+1e-6)
        # (batch, n_mels, time).tranpose(...) dot (n_mels, n_mfcc)
        # -> (batch, time, n_mfcc).tranpose(...)
        mfcc = torch.matmul(mel_spec.transpose(1, 2), self.dct_mat).transpose(1, 2)
        return mfcc

def spectrogram(audio, size=2048, hop_length=1024, power=2, center=False, window=None):
    power_spec = amp(torch.view_as_real(torch.stft(audio, size, window=window, hop_length=hop_length, center=center, return_complex=True)))
    if power == 2:
        spec = power_spec
    elif power == 1:
        spec = power_spec.sqrt()
    return spec

def compute_lsd(resyn_audio, orig_audio):
    window = torch.hann_window(1024).to(orig_audio.device)
    orig_power_s = spectrogram(orig_audio, 1024, 256, window=window).detach()
    resyn_power_s = spectrogram(resyn_audio, 1024, 256, window=window).detach()
    lsd = torch.sqrt(((10 * (torch.log10(resyn_power_s+1e-5)-torch.log10(orig_power_s+1e-5)))**2).sum(dim=(1,2))) / orig_power_s.shape[-1]
    lsd = lsd.mean()
    return lsd

def spectral_convergence(resyn_audio, target_audio):
    window = torch.hann_window(1024).to(target_audio.device)
    target_power_s = spectrogram(target_audio, 1024, 256, window=window).detach()
    resyn_power_s = spectrogram(resyn_audio, 1024, 256, window=window).detach()
    sc_loss = torch.linalg.norm(resyn_power_s - target_power_s, 'fro', dim=(1,2)) / torch.linalg.norm(target_power_s, 'fro', dim=(1,2))
    return sc_loss.mean()

def multiscale_fft(audio, sizes=[64, 128, 256, 512, 1024, 2048], hop_lengths=None, windows=None) -> torch.Tensor:
    """multiscale fft power spectrogram
    uses torch.stft so it should be differentiable

    Args:
        audio : (batch) input audio tensor Shape: [(batch), n_samples]
        sizes : fft sizes. Defaults to [64, 128, 256, 512, 1024, 2048].
        overlap : overlap between windows. Defaults to 0.75.
    """
    specs = []
    if hop_lengths is None:
        overlap = 0.75
        hop_lengths = [int((1-overlap)*s) for s in sizes]
    if windows is None:
        windows = [torch.hann_window(s, device=audio.device) for s in sizes]
    if isinstance(audio, np.ndarray):
        audio = torch.from_numpy(audio)
    stft_params = zip(sizes, hop_lengths, windows)
    for n_fft, hl, win in stft_params:
        wl = win.shape[-1]
        stft = torch.stft(audio, n_fft, window=win, hop_length=hl, win_length=wl, center=False, return_complex=True)
        stft = torch.view_as_real(stft)
        specs.append(amp(stft))
    return specs

def compute_loudness(audio, sample_rate=16000, frame_rate=50, n_fft=2048, range_db=120.0, ref_db=20.7, a_weighting=None):
    """Perceptual loudness in dB, relative to white noise, amplitude=1.

    Args:
        audio: tensor. Shape [batch_size, audio_length] or [audio_length].
        sample_rate: Audio sample rate in Hz.
        frame_rate: Rate of loudness frames in Hz.
        n_fft: Fft window size.
        range_db: Sets the dynamic range of loudness in decibels. The minimum loudness (per a frequency bin) corresponds to -range_db.
        ref_db: Sets the reference maximum perceptual loudness as given by (A_weighting + 10 * log10(abs(stft(audio))**2.0). The default value corresponds to white noise with amplitude=1.0 and n_fft=2048. There is a slight dependence on fft_size due to different granularity of perceptual weighting.

    Returns:
        Loudness in decibels. Shape [batch_size, n_frames] or [n_frames,].
    """
    # Temporarily a batch dimension for single examples.
    is_1d = (len(audio.shape) == 1)
    if is_1d:
        audio = audio[None, :]

    # Take STFT.
    hop_length = sample_rate // frame_rate
    s = torch.stft(audio, n_fft=n_fft, hop_length=hop_length, return_complex=True)
    s = torch.view_as_real(s)
    # batch, frequency_bins, n_frames

    # Compute power of each bin
    power = amp(s)
    power_db = torch.log10(power + 1e-5)
    power_db *= 10.0

    # Perceptual weighting.
    if a_weighting is None:
        frequencies = librosa.fft_frequencies(sr=sample_rate, n_fft=n_fft)
        a_weighting = librosa.A_weighting(frequencies)[None, :, None]
        a_weighting = torch.from_numpy(a_weighting.astype(np.float32)).to(audio.device)
    loudness = power_db + a_weighting

    # Set dynamic range.
    loudness -= ref_db
    loudness = torch.clamp(loudness, min=-range_db)

    # Average over frequency bins.
    loudness = torch.mean(loudness, dim=1)

    # Remove temporary batch dimension.
    loudness = loudness[0] if is_1d else loudness

    # Compute expected length of loudness vector
    n_secs = audio.shape[-1] / float(sample_rate)  # `n_secs` can have milliseconds
    expected_len = int(n_secs * frame_rate)

    # Pad with `-range_db` noise floor or trim vector
    loudness = pad_or_trim_to_expected_length(loudness, expected_len, -range_db)
    return loudness

def loudness_loss(input_audio, target_audio, sr=16000):
    input_l = compute_loudness(input_audio, sr)
    target_l = compute_loudness(target_audio, sr)
    return F.l1_loss(torch.pow(10, input_l/10), torch.pow(10, target_l/10))

def fix_f0(f0, diff_width=4, thres=0.4):
    """
    f0: [batch, n_frames]
    f0 computed by crepe tends to fall off near note end
    fix the f0 value to the last sane value when f0 falls off fast
    """
    orig_shape = f0.shape
    if len(orig_shape) == 3: #[batch, n_frames, feature_dim=1]
        f0 = f0.squeeze(-1)
    norm_diff = (f0[:, diff_width:] - f0[:, :-diff_width]) / f0[:, diff_width:]
    norm_diff = F.pad(norm_diff, (0, diff_width))
    spike = norm_diff.abs()<thres
    mask = torch.cumprod(spike, dim=0)==1
    last_v = f0[mask][-1]
    fixed_f0 = torch.where(mask, f0, last_v)
    if len(orig_shape) == 3: #[batch, n_frames, feature_dim=1]
        fixed_f0 = fixed_f0.unsqueeze(1)
    return fixed_f0