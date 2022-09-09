import hydra
import torch
from omegaconf import DictConfig

from aibedo.train import run_model
import os


@hydra.main(config_path="aibedo/configs/", config_name="main_config.yaml", version_base=None)
def main(config: DictConfig) -> float:
    """ Run/train model based on the config file configs/main_config.yaml (and any command-line overrides). """
    return run_model(config)


if __name__ == "__main__":
    os.environ['HYDRA_FULL_ERROR'] = "1"
    os.environ['OC_CAUSE'] = "1"
    torch.autograd.set_detect_anomaly(True)
    main()
