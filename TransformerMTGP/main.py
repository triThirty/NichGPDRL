import sys

import GPFC as TransformerGPmain
import torch
import random
import hydra
from omegaconf import DictConfig
import numpy as np


sys.path


@hydra.main(version_base=None, config_path="../conf", config_name="config")
def my_app(cfg: DictConfig) -> None:
    seed = cfg.exp.seeds
    device = cfg.exp.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    TransformerGPmain.main(
        # dataset_name,
        # seed,
        # num_pre_selection,
        # device,
        # score_based_algo,
        cfg
    )


if __name__ == "__main__":
    my_app()
    # for path in sys.path:
    #     print(path)
