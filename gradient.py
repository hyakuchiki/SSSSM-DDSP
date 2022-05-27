from diffsynth.modelutils import construct_synth_from_conf
from diffsynth.loss import multispectrogram_loss
from omegaconf import OmegaConf, open_dict
import torch
import matplotlib.pyplot as plt
import numpy as np
import librosa
import librosa.display
import argparse
from copy import deepcopy
import os
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('out_dir',  type=str,   help='')
    parser.add_argument('synth_conf',   type=str,   help='')
    parser.add_argument('--audio_len',  type=float, default=4.0)
    parser.add_argument('--sr',         type=int,   default=16000)
    parser.add_argument('--batch_size', type=int,   default=100)
    parser.add_argument('--center',     type=float, default=0.4)

    args = parser.parse_args()

    device = 'cuda'
    conf = OmegaConf.load(args.synth_conf)
    with open_dict(conf):
        conf.sample_rate=args.sr
    synth = construct_synth_from_conf(conf).to(device)

    n_samples = int(args.audio_len * args.sr)

    target_params = {}
    for name, ps in synth.ext_param_sizes.items():
        target_params[name] = torch.ones(args.batch_size, 1, ps).to(device) * args.center
    for param_name in synth.fixed_param_names:
        param_value = getattr(synth, param_name)
        target_params[param_name] = param_value[None, None, :].expand(args.batch_size, -1, -1).to(device)

    target_audio, _output = synth(target_params, n_samples)

    for param_name, sweep_ps in synth.ext_param_sizes.items():
        for i in range(sweep_ps):
            sweep_params = deepcopy(target_params)
            x = torch.linspace(0, 1, args.batch_size, requires_grad=True)
            sweep_params[param_name][:, :, i] = x[:, None].to(device)
            sweep_name = param_name + '_' +str(i)
            save_dir = os.path.join(args.out_dir, f'gradsweep_{sweep_name}.png')
            def hook(grad):
                # grad [100, 1, 1]
                fig, ax = plt.subplots()
                ax.plot(x.detach(), grad[:, 0, i].detach().cpu().numpy())
                ax.axvline(x=args.center, c='red')
                fig.savefig(save_dir)
                plt.close(fig)
            sweep_params[param_name].register_hook(hook)
            x_audio, _output = synth(sweep_params, n_samples)
            spec_losses = multispectrogram_loss(x_audio, target_audio, reduce='none')
            losses = sum(spec_losses['logspec'].values()) # (batch)
            # plot loss
            fig, ax = plt.subplots()
            ax.plot(x.detach(), losses.detach().cpu().numpy())
            ax.axvline(x=args.center, c='red')
            fig.savefig(os.path.join(args.out_dir, f'losssweep_{sweep_name}.png'))
            torch.mean(losses).backward()
            plt.close(fig)