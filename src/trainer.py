import hydra
from omegaconf import DictConfig
from lightning import Trainer

def get_trainer(config: DictConfig):
    callbacks = list(hydra.utils.instantiate(config.training.callbacks).values())
    logger = hydra.utils.instantiate(config.training.logger)
    trainer = Trainer(**config.training.trainer,callbacks=callbacks)
    return trainer
