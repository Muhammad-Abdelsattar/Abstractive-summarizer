import os
import hydra
from omegaconf import DictConfig

from src.data import SummarizationDataModule
from src.model import AbstractiveSummarizationModule
from src.trainer import get_trainer


@hydra.main(config_path='.',config_name="config")
def run(config: DictConfig):
    data_module = SummarizationDataModule(config=config)
    model = AbstractiveSummarizationModule(config=config)
    trainer = get_trainer(config=config)
    trainer.fit(model=model,datamodule=data_module,ckpt_path=config.training.checkpoint)


if __name__ == "__main__":
    run()
