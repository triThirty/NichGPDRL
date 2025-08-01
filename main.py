import TransformerMTGP.GPFC as TransformerGPmain
from TransformerMTGP import main_experiment_transformerGP_all_generations_test_results
from MTGP import GPFC as MTGPmain, main_experiment_MTGP_all_generations_test_results
import MTGP_KNN.GPFC as KNNmain
from MTGP_KNN import main_experiment_knn_MTGP_all_generations_test_results
import torch
import random
from omegaconf import DictConfig, OmegaConf
import numpy as np


def my_app(cfg: DictConfig) -> None:
    seed = cfg.seeds
    device = cfg.device
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    if device == "cuda":
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if cfg.algo == "Transformer":
        TransformerGPmain.main(cfg)
        main_experiment_transformerGP_all_generations_test_results.main(
            cfg.scenarios, seed, "transformerGP_all_gen_test"
        )
    elif cfg.algo == "Transformer_SSGP":
        TransformerGPmain.main(cfg)
        main_experiment_transformerGP_all_generations_test_results.main(
            cfg.scenarios, seed, "transformerGP_all_gen_test"
        )
    elif cfg.algo == "KNN":
        KNNmain.main(cfg)
        main_experiment_knn_MTGP_all_generations_test_results.main(cfg)
    elif cfg.algo == "MTGP":
        MTGPmain.main(cfg)
        main_experiment_MTGP_all_generations_test_results.main(
            cfg.scenarios, seed, "GP_all_gen_test"
        )


def merge_conf(cli_conf: DictConfig) -> DictConfig:
    base_conf = OmegaConf.load("conf/defaults/base.yaml")
    algo_conf = OmegaConf.load(f"conf/exp/{cli_conf.algo}.yaml")
    conf = OmegaConf.merge(base_conf, algo_conf, cli_conf)
    return conf


if __name__ == "__main__":
    cli_conf = OmegaConf.from_cli()  # Default to KNN if not specified
    cfg = merge_conf(cli_conf)
    my_app(cfg)
