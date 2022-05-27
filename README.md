# Repository for Semi-Supervised Synthesizer Sound Matching with Differentiable DSP

Accompanying website: 

## Files

- gen_dataset.py
	- Used for dataset generation
- train.py
	- Training script
- test.py
	- Testing script
- configs
	- hydra configs for experiments

## Usage

### Dataset generation

Specify the synth architecture from configs/synth (in this case, h2of_fx_env which is the FX-Env setting from the paper).

```
python gen_dataset.py [generated in-domain dataset dir] configs/synth/h2of_fx_env.yaml
```

### Training

Edit configs/experiments/exp with dataset directory (id_base, data_cfgs.ood.base_dir).

Train a model with the `Synth` setting:

```
python train.py experiment=exp synth=h2of_fx_env
```

Change the loss function to parameter loss only (`P-loss` setting):

```
python train.py experiment=exp synth=h2of_fx_env loss=only_param
```

Resume with the `Real` setting:

```
python train.py experiment=exp synth=h2of_fx_env data.train_key=ood ckpt=[checkpoint file at 200th epoch of Synth]
```

Resume with the `Even` setting:

```
python train.py experiment=exp synth=h2of_fx_env data.train_key=id loss=even_spec_fro ckpt=[checkpoint file at 50th epoch of P-loss] trainer.max_epochs=200
```

```
python train.py experiment=exp synth=h2of_fx_env data.train_key=[id,ood] loss=even_spec_fro ckpt=[checkpoint file of the above run]
```