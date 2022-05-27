import hydra

@hydra.main(config_path="configs/", config_name="config.yaml")
def main(cfg):
    import os
    from omegaconf import open_dict
    import pytorch_lightning as pl
    import torch
    from plot import AudioLogger, SaveEvery
    import warnings
    from pytorch_lightning.callbacks import ModelCheckpoint
    from diffsynth.model import EstimatorSynth, EstimatorSynthFX
    from diffsynth.data import IdOodDataModule, MultiDataModule
    pl.seed_everything(cfg.seed, workers=True)
    warnings.simplefilter('ignore', RuntimeWarning)
    # load model
    model = EstimatorSynth(cfg.model, cfg.synth, cfg.loss, cfg.get('ext_f0', False))
    # loggers setup
    tb_logger = pl.loggers.TensorBoardLogger("tb_logs", "", default_hp_metric=False, version='')
    mf_logger = pl.loggers.MLFlowLogger(cfg.name, tracking_uri="file://" + hydra.utils.get_original_cwd() + "/mlruns")
    # load data
    datamodule = hydra.utils.instantiate(cfg.data)
    # trainer setup
    # keep every checkpoint_every epochs and best epoch
    checkpoint_callback = ModelCheckpoint(dirpath=os.getcwd(), monitor=cfg.monitor, save_top_k=-1, save_last=False, every_n_epochs=cfg.checkpoint_every)
    callbacks = [pl.callbacks.LearningRateMonitor(logging_interval='step'), AudioLogger(), checkpoint_callback]
    if cfg.ckpt is not None:
        cfg.ckpt = hydra.utils.to_absolute_path(cfg.ckpt)
    trainer = hydra.utils.instantiate(cfg.trainer, callbacks=callbacks, logger=[tb_logger, mf_logger])
    # log hyperparameters
    with open_dict(cfg):
        cfg.model.total_params = sum(p.numel() for p in model.parameters())
    mf_logger.log_hyperparams(cfg)
    # make model
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.ckpt)
    # torch.save(datamodule, os.path.join(os.getcwd(), 'datamodule.pt'))
    # return value used for optuna
    return trainer.callback_metrics[cfg.monitor]

if __name__ == "__main__":
    main()
