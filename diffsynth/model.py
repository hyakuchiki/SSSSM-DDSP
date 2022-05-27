import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from diffsynth.spectral import compute_lsd, loudness_loss, Mfcc, spectral_convergence
import pytorch_lightning as pl
from diffsynth.modelutils import construct_synth_from_conf
from diffsynth.schedules import ParamSchedule
import hydra
from diffsynth.estimator import F0MFCCEstimator, F0MelEstimator
from itertools import chain

class EstimatorSynth(pl.LightningModule):
    """
    audio -> Estimator -> Synth -> audio
    """
    def __init__(self, model_cfg, synth_cfg, losses_cfg, ext_f0=False):
        super().__init__()
        self.synth = construct_synth_from_conf(synth_cfg, ext_f0)
        self.estimator = hydra.utils.instantiate(model_cfg.estimator, output_dim=self.synth.ext_param_size)
        # Initialize losses
        self.loss_w_sched = ParamSchedule(losses_cfg.sched) # loss weighting
        self.losses = nn.ModuleDict()
        for loss_name, loss_cfg in losses_cfg.losses.items():
            if loss_name == 'param':
                self.losses['param'] = hydra.utils.instantiate(loss_cfg, dag_summary=self.synth.dag_summary, fixed_param_names=self.synth.fixed_param_names)
            else:
                self.losses[loss_name] = hydra.utils.instantiate(loss_cfg)
        # loss must correspond to a loss weight
        assert all([(loss_name in self.loss_w_sched.sched) for loss_name in self.losses])
        self.log_grad = model_cfg.log_grad
        self.lr = model_cfg.lr
        self.decay_rate = model_cfg.decay_rate
        self.mfcc = Mfcc(n_fft=1024, hop_length=256, n_mels=40, n_mfcc=20, sample_rate=16000)

    def estimate_param(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: estimated parameters in Tensor ranged 0~1
        """
        if isinstance(self.estimator, (F0MelEstimator, F0MFCCEstimator)):
            return self.estimator(conditioning['audio'], conditioning['BFRQ'])
        return self.estimator(conditioning['audio'])

    def log_param_grad(self, params_dict):
        def save_grad(name):
            def hook(grad):
                # batch, n_frames, feat_size
                grad_v = grad.abs().mean(dim=(0, 1))
                for i, gv in enumerate(grad_v):
                    self.log('train/param_grad/'+name+f'_{i}', gv, on_step=False, on_epoch=True)
            return hook

        if self.log_grad:
            for k, v in params_dict.items():
                if v.requires_grad == True:
                    v.register_hook(save_grad(k))

    def forward(self, conditioning):
        """
        Args:
            conditioning (dict): {'PARAM NAME': Conditioning Tensor, ...}

        Returns:
            torch.Tensor: audio
        """
        audio_length = conditioning['audio'].shape[1]
        est_param = self.estimate_param(conditioning)
        params_dict = self.synth.fill_params(est_param, conditioning)
        if self.log_grad is not None:
            self.log_param_grad(params_dict)

        resyn_audio, outputs = self.synth(params_dict, audio_length)
        return resyn_audio, outputs

    def get_params(self, conditioning):
        """
        Don't render audio
        """
        est_param = self.estimate_param(conditioning)
        params_dict = self.synth.fill_params(est_param, conditioning)
        if self.log_grad is not None:
            self.log_param_grad(params_dict)
        
        synth_params = self.synth.calculate_params(params_dict)
        return synth_params

    def train_losses(self, output, target, loss_w=None):
        loss_dict = {}
        for k, loss in self.losses.items():
            weight = 1.0 if loss_w is None else loss_w[k]
            if weight > 0.0:
                loss_dict[k] = weight * loss(output, target)
            else:
                loss_dict[k] = 0.0
        return loss_dict

    def monitor_losses(self, output, target):
        mon_losses = {}
        # Audio losses
        target_audio = target['audio']
        resyn_audio = output['output']
        # losses not used for training
        ## audio losses
        mon_losses['lsd'] = compute_lsd(resyn_audio, target_audio)
        mon_losses['sc'] = spectral_convergence(resyn_audio, target_audio)
        mon_losses['loud'] = loudness_loss(resyn_audio, target_audio)
        mon_losses['mfcc_l1'] = F.l1_loss(self.mfcc(resyn_audio)[:, 1:], self.mfcc(target_audio)[:, 1:])
        mon_losses['wave_l1'] = F.l1_loss(resyn_audio, target_audio)
        return mon_losses

    def training_step(self, batch_dict, batch_idx):
        # get loss weights
        loss_weights = self.loss_w_sched.get_parameters(self.global_step)
        self.log_dict({'lw/'+k: v for k, v in loss_weights.items()}, on_epoch=True, on_step=False)
        if sum(loss_weights.values()) == loss_weights['param']:
            # only parameter loss is used so don't render audio
            output_dict = self.get_params(batch_dict)
        else:
            # render audio
            resyn_audio, output_dict = self(batch_dict)
        losses = self.train_losses(output_dict, batch_dict, loss_weights)
        self.log_dict({'train/'+k: v for k, v in losses.items()}, on_epoch=True, on_step=False)
        batch_loss = sum(losses.values())
        self.log('train/total', batch_loss, prog_bar=True, on_epoch=True, on_step=False)
        return batch_loss

    def validation_step(self, batch_dict, batch_idx, dataloader_idx=0):
        # render audio
        resyn_audio, outputs = self(batch_dict)
        losses = self.train_losses(outputs, batch_dict)
        eval_losses = self.monitor_losses(outputs, batch_dict)
        losses.update(eval_losses)
        losses = {'val_{0}/{1}'.format(dataloader_idx, k): v for k, v in losses.items()}
        self.log_dict(losses, prog_bar=True, on_epoch=True, on_step=False, add_dataloader_idx=False)
        return losses

    def test_step(self, batch_dict, batch_idx, dataloader_idx=0):
        # render audio
        resyn_audio, outputs = self(batch_dict)
        losses = self.train_losses(outputs, batch_dict)
        eval_losses = self.monitor_losses(outputs, batch_dict)
        losses.update(eval_losses)
        losses = {'val_{0}/{1}'.format(dataloader_idx, k): v for k, v in losses.items()}
        self.log_dict(losses, prog_bar=True, on_epoch=True, on_step=False, add_dataloader_idx=False)
        return losses

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.estimator.parameters(), self.synth.parameters()), self.lr)
        # optimizer = torch.optim.Adam(self.estimator.parameters(), self.lr)
        return {
        "optimizer": optimizer,
        "lr_scheduler": {
            "scheduler": torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=self.decay_rate)
            }
        }