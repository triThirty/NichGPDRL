import TransformerMTGP.GPFC as TransformerGPmain
import MTGP.GPFC as MTGPmain
import MTGP_KNN.GPFC as KNNmain
import torch
import random
import hydra
from omegaconf import DictConfig
import numpy as np


@hydra.main(version_base=None, config_path="conf", config_name="config")
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

    if cfg.exp.algo == "Transformer":
        TransformerGPmain.main(cfg)
    elif cfg.exp.algo == "Transformer_SSGP":
        TransformerGPmain.main(cfg)
    elif cfg.exp.algo == "KNN":
        KNNmain.main(cfg)
    elif cfg.exp.algo == "MTGP":
        MTGPmain.main(cfg)


if __name__ == "__main__":
    my_app()
