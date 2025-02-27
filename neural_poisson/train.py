import logging

import hydra
from omegaconf import DictConfig, OmegaConf

from lib.utils.config_utils import instantiate_callbacks, set_configs

log = logging.getLogger()


@hydra.main(version_base=None, config_path="../conf", config_name="train")
def train(cfg: DictConfig):
    cfg = set_configs(cfg)
    log.info("==> loading config ...")

    log.info(f"==> initializing logger <{cfg.logger._target_}> ...")
    logger = hydra.utils.instantiate(cfg.logger)

    log.info(f"==> initializing datamodule <{cfg.data._target_}> ...")
    datamodule = hydra.utils.instantiate(cfg.data)

    log.info(f"==> initializing model <{cfg.model._target_}> ...")
    model = hydra.utils.instantiate(cfg.model)

    log.info("==> initializing callbacks ...")
    callbacks = instantiate_callbacks(cfg.get("callbacks"))

    log.info("==> initializing trainer ...")
    trainer = hydra.utils.instantiate(cfg.trainer, logger=logger, callbacks=callbacks)

    if logger:
        log.info("==> logging hyperparameters ...")
        trainer.logger.log_hyperparams(OmegaConf.to_container(cfg))

    log.info("==> start training ...")
    trainer.fit(model=model, datamodule=datamodule, ckpt_path=cfg.get("ckpt_path"))


if __name__ == "__main__":
    try:
        train()
    except Exception as e:
        pass
