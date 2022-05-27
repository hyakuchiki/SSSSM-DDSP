from librosa.core import fft
import torch
import torch.nn as nn
from torchaudio.functional import compute_deltas
from diffsynth.spectral import multiscale_fft, compute_loudness, spectrogram
import librosa
from diffsynth.util import log_eps, resample_frames
import torch.nn.functional as F
import functools

def multispectrogram_loss(x_audio, target_audio, fft_sizes=[64, 128, 256, 512, 1024, 2048], hop_ls=None, windows=None, log_mag_w=1.0, mag_w=1.0, fro_w=0.0, reduce='mean'):
    x_specs = multiscale_fft(x_audio, fft_sizes, hop_ls, windows)
    target_specs = multiscale_fft(target_audio, fft_sizes, hop_ls, windows)
    spec_loss = {}
    log_spec_loss = {}
    fro_spec_loss = {}
    if reduce == 'mean':
        reduce_dims = (0,1,2)
    else:
        # do not reduce batch
        reduce_dims = (1,2)
    for n_fft, x_spec, target_spec in zip(fft_sizes, x_specs, target_specs):
        if mag_w > 0:
            spec_loss[n_fft] = mag_w * torch.mean(torch.abs(x_spec - target_spec), dim=reduce_dims)
        if log_mag_w > 0:
            log_spec_loss[n_fft] = log_mag_w * torch.mean(torch.abs(log_eps(x_spec) - log_eps(target_spec)), dim=reduce_dims)
        if fro_w > 0: # spectral convergence
            fro_loss = torch.linalg.norm(x_spec - target_spec, 'fro', dim=(1,2)) / torch.linalg.norm(target_spec, 'fro', dim=(1,2))
            fro_spec_loss[n_fft] = torch.mean(fro_loss) if reduce=='mean' else fro_loss
    return {'spec':spec_loss, 'logspec':log_spec_loss, 'fro': fro_spec_loss}

def waveform_loss(x_audio, target_audio, l1_w=0, l2_w=1.0, linf_w=0, linf_k=1024, norm=None):
    norm = {'l1':1.0, 'l2':1.0} if norm is None else norm
    l1_loss = l1_w * torch.mean(torch.abs(x_audio - target_audio)) / norm['l1'] if l1_w > 0 else 0.0
    # mse loss
    l2_loss = l2_w * torch.mean((x_audio - target_audio)**2) / norm['l2'] if l2_w > 0 else 0.0
    if linf_w > 0:
        # actually gets k elements
        residual = (x_audio - target_audio)**2
        values, _ = torch.topk(residual, linf_k, dim=-1)
        linf_loss = torch.mean(values) / norm['l2']
    else:
        linf_loss = 0.0
    return {'l1':l1_loss, 'l2':l2_loss, 'linf':linf_loss}

class SpectralLoss(nn.Module):
    """
    loss for reconstruction with multiscale spectrogram loss and waveform loss
    """
    def __init__(self, fft_sizes=[64, 128, 256, 512, 1024, 2048], hop_lengths=None, win_lengths=None, mag_w=1.0, log_mag_w=1.0, fro_w=0.0):
        super().__init__()
        self.fft_sizes = fft_sizes
        self.hop_lengths = hop_lengths
        self.win_lengths = win_lengths
        self.mag_w = mag_w
        self.log_mag_w = log_mag_w
        self.fro_w = fro_w
        if win_lengths is not None:
            for wl in win_lengths:
                self.register_buffer(f'window_{wl}', torch.hann_window(wl))
        self.spec_loss = functools.partial(multispectrogram_loss, fft_sizes=fft_sizes, hop_ls=hop_lengths, log_mag_w=log_mag_w, mag_w=mag_w, fro_w=fro_w)
        
    def __call__(self, output_dict, target_dict, sum_losses=True):
        x_audio = output_dict['output']
        target_audio = target_dict['audio']
        if self.win_lengths is not None:
            windows = [getattr(self, f'window_{wl}') for wl in self.win_lengths]
        else:
            windows = None
        spec_losses = self.spec_loss(x_audio, target_audio, windows=windows)
        if sum_losses:
            multi_spec_loss = sum(spec_losses['spec'].values()) + sum(spec_losses['logspec'].values()) + sum(spec_losses['fro'].values())
            multi_spec_loss /= (len(self.fft_sizes)*(self.mag_w + self.log_mag_w + self.fro_w))
            return multi_spec_loss
        else:
            multi_spec_losses = {k: sum(v.values()) for k, v in spec_losses.items()}
            return multi_spec_losses

class WaveformLoss(nn.Module):
    def __init__(self, l1_w=0, l2_w=0.0, linf_w=0.0, linf_k=1024) -> None:
        super().__init__()
        self.l1_w=l1_w
        self.l2_w=l2_w
        self.linf_w=linf_w
        self.wave_loss = functools.partial(waveform_loss, l1_w=l1_w, l2_w=l2_w, linf_w=linf_w, linf_k=linf_k)

    def __call__(self, output_dict, target_dict):
        x_audio = output_dict['output']
        target_audio = target_dict['audio']
        wave_losses = self.wave_loss(x_audio, target_audio)
        wave_loss = wave_losses['l1'] + wave_losses['l2'] + wave_losses['linf']
        wave_loss /= (self.l1_w + self.l2_w + self.linf_w)
        return wave_loss

class SpectralDeltaLoss(nn.Module):
    def __init__(self, fft_size=2048, hop_length=512, win_length=2048) -> None:
        super().__init__()
        self.fft_size=fft_size
        self.hop_length=hop_length 
        self.register_buffer('window', torch.hann_window(win_length))
        self.spec = functools.partial(spectrogram, size=fft_size, hop_length=hop_length)
    
    def __call__(self, output_dict, target_dict):
        x_audio = output_dict['output']
        target_audio = target_dict['audio']
        x_spec = log_eps(self.spec(x_audio, window=self.window))
        target_spec = log_eps(self.spec(target_audio, window=self.window))
        delta_x = compute_deltas(x_spec)
        delta_target = compute_deltas(target_spec)
        return F.l1_loss(delta_x, delta_target)

class ParamLoss(nn.Module):
    def __init__(self, dag_summary, fixed_param_names, loss='l1') -> None:
        super().__init__()
        self.dag_summary = dag_summary
        self.fixed_param_names = fixed_param_names
        if loss=='l1':
            self.loss_func = F.l1_loss 
        elif loss=='smooth':
            self.loss_func = F.smooth_l1_loss
        elif loss=='mse':
            self.loss_func = F.mse_loss

    def __call__(self, output_dict, target_dict):
        loss = 0
        if 'params' in target_dict:
            target_param = target_dict['params']
            for k, target in target_param.items():
                output_name = self.dag_summary[k]
                if output_name in self.fixed_param_names:
                    continue
                if target.numel() == 0:
                    continue
                x = output_dict[output_name]
                if target.shape[1] > 1:
                    x = resample_frames(x, target.shape[1])
                loss += self.loss_func(x, target)
            loss = loss / len(target_param.keys())
        return loss

class LoudnessLoss(nn.Module):
    def __init__(self, fft_size=1024, sr=16000, frame_rate=50, db=False) -> None:
        super().__init__()
        self.fft_size = fft_size
        self.sr = sr
        self.frame_rate = frame_rate
        self.db = db
        frequencies = librosa.fft_frequencies(sr=sr, n_fft=fft_size)
        a_weighting = librosa.A_weighting(frequencies)[None, :, None]
        self.register_buffer('a_weighting', torch.from_numpy(a_weighting).float())

    def __call__(self, output_dict, target_dict):
        x_audio = output_dict['output']
        target_audio = target_dict['audio']
        x_loud = compute_loudness(x_audio, self.sr, self.frame_rate, a_weighting=self.a_weighting)
        target_loud = compute_loudness(target_audio, self.sr, self.frame_rate, a_weighting=self.a_weighting)
        l1_loss = F.l1_loss(torch.pow(10, x_loud/10), torch.pow(10, target_loud/10))
        if self.db:
            l1_loss = 10 * torch.log10(l1_loss)
        return l1_loss

def log_sinkhorn_loss(source_dist, target_dist, cost_matrix, epsilon=0.01, niter = 10):
    """
    logarithmic updates from 
    https://github.com/gpeyre/SinkhornAutoDiff/blob/master/sinkhorn_pointcloud.py
    https://github.com/jiupinjia/stylized-neural-painting/blob/e7a66b20259141656df04d41629b7d4f27fd9262/pytorch_batch_sinkhorn.py
    Parameters
    ----------
    source: (batch, dim_a)
    target: (batch, dim_b)
    cost_matrix: (batch, dim_a, dim_b)
    n_iter: number of iterations
    """

    def M(u, v):
        """
        Modified cost for logarithmic updates
        $M_{ij} = (-c_{ij} + u_i + v_j) / \epsilon$
        # source, target
        """
        return (-cost_matrix + u.unsqueeze(2) + v.unsqueeze(1)) / epsilon

    def normalize(X, eps=1e-9):
        # normalize to probability vector
        X.data = torch.clamp(X.data, min=0, max=1e9) # keep gradients?
        X = X + eps
        return X / X.sum(-1, True)

    # def lse(A):
    #     "log-sum-exp"
    #     return torch.log(torch.exp(A).sum(dim=2) + 1e-6)  # add 10^-6 to prevent NaN

    mu = torch.log(normalize(source_dist))
    nu = torch.log(normalize(target_dist))
    u, v = torch.zeros_like(mu), torch.zeros_like(nu)

    for i in range(niter):
        # batch, source, ~~target~~
        u = epsilon * (mu - torch.logsumexp(M(u, v), dim=2)) + u
        # batch, target, ~~source~~
        v = epsilon * (nu - torch.logsumexp(M(u, v).transpose(1, 2), dim=2)) + v
        # u = epsilon * (mu - lse(M(u, v))) + u
        # v = epsilon * (nu - lse(M(u, v).transpose(1, 2))) + v
    
    pi = torch.exp(M(u, v))  # Transport plan pi = diag(a)*K*diag(b)
    cost = torch.sum(pi * cost_matrix, dim=[1, 2])  # Sinkhorn cost
    return pi, cost

class OscSpecSinkhornLoss(nn.Module):
    def __init__(self, fft_size=1024, win_length=None, hop_length=None, sr=16000, niter=5, epsilon=0.01, reduce='mean', amp_key='SIN_AMPS', frq_key='SIN_FRQS') -> None:
        super().__init__()
        self.niter = niter
        self.fft_size = fft_size
        self.win_length = win_length
        if win_length is None:
            win_length = fft_size
        if hop_length is None:
            hop_length = fft_size//2
        self.hop_length = hop_length
        self.register_buffer('window', torch.hann_window(win_length))
        self.epsilon = epsilon
        self.reduce = reduce
        self.amp_key = amp_key
        self.frq_key = frq_key
        self.register_buffer('bin_freqs', torch.from_numpy(librosa.fft_frequencies(sr=sr, n_fft=fft_size)).float())

    def cost_matrix(self, freqs):
        # batch*n_frames, harmonics
        N = freqs.shape[0]
        cost_M = torch.cdist(freqs[:, :, None], self.bin_freqs[None, :, None].expand(N, -1, 1), p=1)
        return cost_M

    def forward(self, output_dict, target_dict):
        target_audio = target_dict['audio']
        target_spec = spectrogram(target_audio, self.fft_size, window=self.window, hop_length=self.hop_length, center=False, power=1)
        batch_size, n_features, n_frames = target_spec.shape
        target_spec = target_spec.transpose(1, 2).flatten(0, 1) #(batch*n_frames, fft_freqs)
        #(batch*n_frames, harmonics)
        params = output_dict['params']
        output_freqs = resample_frames(params[self.amp_key], n_frames).flatten(0,1)
        n_freqs = output_freqs.shape[-1]
        output_amps = resample_frames(params[self.frq_key], n_frames).flatten(0,1)
        cost_matrix = self.cost_matrix(output_freqs)
        transport_plan, cost = log_sinkhorn_loss(output_amps, target_spec, cost_matrix, self.epsilon, self.niter)
        if self.reduce == 'mean':
            return cost.mean()
        else: # for debugging
            transport_plan = transport_plan.reshape(batch_size, n_frames, n_freqs, n_features)
            cost = cost.reshape(batch_size, n_frames)
            return transport_plan, cost

# doesn't really help
class SinkhornSpectralLoss(nn.Module):
    """
    compare two spectra with sinkhorn distance
    """
    def __init__(self, fft_size=1024, win_length=None, hop_length=None, metric='euclidean', sr=None, niter=5, epsilon=0.01, reduce='mean') -> None:
        super().__init__()
        self.niter = niter
        self.epsilon = epsilon
        self.fft_size = fft_size
        self.nbins = fft_size//2 + 1
        if win_length is None:
            win_length = fft_size
        if hop_length is None:
            hop_length = fft_size//2
        self.sr = sr
        self.register_buffer('cost_matrix', self.get_cost_matrix(metric))
        self.register_buffer('window', torch.hann_window(win_length))
        self.hop_length = hop_length
        self.reduce = reduce

    def get_cost_matrix(self, metric):
        # n_bins / 
        if metric == 'euclidean':
            # nbins = self.fft_size//2+1
            x = torch.linspace(0, 1, self.nbins)[None, :, None]
            cost_matrix = torch.cdist(x, x).squeeze(0)
            return cost_matrix
        elif metric == 'mel':
            mels = torch.from_numpy(librosa.hz_to_mel(librosa.fft_frequencies(sr=self.sr, n_fft=self.fft_size)))[None, :, None]
            x = mels/mels.max()
            return torch.cdist(x, x).squeeze(0)
        elif metric == 'harmonic':
            freqs = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=1024))
            harm_mat, indices = torch.min(torch.sqrt((freqs[None, :, None] - freqs[None, None, :] * torch.arange(1, 32)[:, None, None])**2), dim=0)
            return harm_mat / harm_mat.max()
        elif metric == 'harmonic2':
            freqs = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=1024))
            harmonics = torch.cat([1 / torch.arange(1, 5), torch.arange(1, 16)], dim=0)
            harm_mat, indices = torch.min(torch.sqrt((freqs[None, :, None] - freqs[None, None, :] * harmonics[:, None, None])**2), dim=0)
            return harm_mat / harm_mat.max()
        elif metric == 'mel_harm':
            freqs = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=1024))
            harm_mat, indices = torch.min(torch.sqrt((freqs[None, :, None] - freqs[None, None, :] * torch.arange(1, 32)[:, None, None])**2), dim=0)
            mels = torch.from_numpy(librosa.hz_to_mel(freqs))[None, :, None]
            mels = mels/mels.max()
            mel_mat = torch.cdist(mels, mels).squeeze(0)
            return (harm_mat + mel_mat) / 2
        elif metric == 'mel_harm2':
            freqs = torch.from_numpy(librosa.fft_frequencies(sr=16000, n_fft=1024))
            harmonics = torch.cat([1 / torch.arange(1, 5), torch.arange(1, 16)], dim=0)
            harm_mat, indices = torch.min(torch.sqrt((freqs[None, :, None] - freqs[None, None, :] * harmonics[:, None, None])**2), dim=0)
            mels = torch.from_numpy(librosa.hz_to_mel(freqs))[None, :, None]
            mels = mels/mels.max()
            mel_mat = torch.cdist(mels, mels).squeeze(0)
            return (harm_mat + mel_mat) / 2
        else:
            raise NotImplementedError(metric)

    def __call__(self, output_dict, target_dict):
        x_audio = output_dict['output']
        target_audio = target_dict['audio']
        x_spec = spectrogram(x_audio, self.fft_size, window=self.window, hop_length=self.hop_length, center=False, power=2)
        target_spec = spectrogram(target_audio, self.fft_size, window=self.window, hop_length=self.hop_length, center=False, power=2)
        x_spec = log_eps(x_spec).transpose(1, 2).flatten(0, 1) # batch*time, n_bins
        target_spec = log_eps(target_spec).transpose(1, 2).flatten(0, 1) # batch*time, n_bins
        transport_plan, cost = log_sinkhorn_loss(x_spec, target_spec, self.cost_matrix, self.epsilon, self.niter)
        if self.reduce == 'mean':
            return cost.mean()
        else: # for debugging
            return transport_plan, cost

def batched_unbalanced_sinkhorn(source, target, cost_matrix, reg=1.0, reg_m=1.0, niter=10):
    """
    Ported from python-ot and modified to support batching
    Parameters
    ----------
    source: (batch, dim_a)
    target: (batch, dim_b)
    cost_matrix: (batch, dim_a, dim_b)
    reg: should be > 0. entropic regularization.
    reg_m: should be > 0. marginal regulatization. reg_m=inf equates to unrelaxed OT.
    n_iter: number of iterations
    """
    batch_dim, dim_a, dim_b = cost_matrix.shape
    u = torch.ones_like(source) / dim_a
    v = torch.ones_like(target) / dim_b
    K = torch.exp(cost_matrix / (-reg))
    fi = reg_m / (reg_m + reg)

    for i in range(niter):
        # Kv = nx.dot(K, v)
        Kv = torch.sum(K * v[:, None, :], dim=2)
        u = (source / Kv) ** fi
        Ktu = torch.sum(K * u[:, :, None], dim=1)
        v = (target / Ktu) ** fi

    ot_plan = u[:, :, None] * K * v[:, None, :]
    loss = torch.sum(ot_plan * cost_matrix) / batch_dim
    return loss, ot_plan

def batched_unbalanced_logsinkhorn(source, target, cost_matrix, reg=1.0, reg_m=1.0, niter=10):
    """
    log-domain stabilised version of unbalanced sinkhorn loss
    Doesn't work well
    Ported from python-ot and modified to support batching
    Parameters
    ----------
    source: (batch, dim_a)
    target: (batch, dim_b)
    cost_matrix: (batch, dim_a, dim_b)
    reg: should be > 0. entropic regularization.
    reg_m: should be > 0. marginal regulatization. reg_m=inf equates to unrelaxed OT.
    n_iter: number of iterations
    """
    _batch_dim, dim_a, dim_b = cost_matrix.shape
    u = torch.ones_like(source) / dim_a
    v = torch.ones_like(target) / dim_b

    K = torch.exp(-cost_matrix / reg)
    fi = reg_m / (reg_m + reg)

    tau=1e5

    alpha = torch.zeros_like(source)
    beta  = torch.zeros_like(target)
    for i in range(niter):
        Kv = torch.sum(K * v[:, None, :], dim=2)
        f_alpha = torch.exp(- alpha / (reg + reg_m))
        f_beta = torch.exp(- beta / (reg + reg_m))
        u = ((source / (Kv + 1e-8)) ** fi) * f_alpha
        Ktu = torch.sum(K * u[:, :, None], dim=1)
        v = ((target / (Ktu + 1e-8)) ** fi) * f_beta
        if torch.any(u > tau) or torch.any(v > tau):
            alpha = alpha + reg * torch.log(torch.max(u, dim=1, keepdim=True)[0])
            beta = beta + reg * torch.log(torch.max(v, dim=1, keepdim=True)[0])
            K = torch.exp((alpha[:, :, None] + beta[:, None, :] - cost_matrix) / reg)
            v = torch.ones_like(v)
    logu = alpha / reg + torch.log(u)
    logv = beta / reg + torch.log(v)
    loss = torch.logsumexp(torch.log(cost_matrix+1e-100) + logu[:, :, None] + logv[:, None, :] / reg, dim=(1,2))
    loss = torch.mean(torch.exp(loss))
    return loss 