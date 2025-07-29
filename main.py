import sys
import argparse

import torch
import random

import MTGP.GPFC as GPmain
import MTGP_KNN.GPFC as KnnGPmain
import NichingMTGP.GPFC as NichingGPmain
import numpy as np

import main_experiment_GP_all_generations_test_results
import main_experiment_Integrated_DRL
import main_experiment_NichingMTGP_Integrated_DRL_R_S_single_agent_with_intermediate
import main_experiment_manualRule_using_validation
import main_experiment_NichingMTGP_Integrated_DRL
import main_experiment_NichingMTGP_Integrated_DRL_R_with_intermediate
import main_experiment_NichingMTGP_Integrated_DRL_S_with_intermediate
import main_training_R
import main_training_R_S_GPrule_single_agent
import main_training_S
import main_training_R_S_GPrule
import main_training_R_GPrule
import main_training_S_GPrule
import main_training_S_online_learning
import main_experiment_transformerGP_all_generations_test_results
import main_experiment_MTGP_all_generations_test_results
import main_experiment_knn_MTGP_all_generations_test_results
import os

sys.path

if __name__ == "__main__":
    # dataset_name = str(sys.argv[1])  # HH or HL or LH or LL
    # seed = int(sys.argv[2])  # a random number, e.g., 0
    # algo = str(sys.argv[3])  # as the following
    # is_pre_selection = bool(sys.argv[4])  # as the following

    parser = argparse.ArgumentParser(description="Run the main experiment")
    parser.add_argument(
        "--dataset_name",
        type=str or list,
        help="The name of the dataset to run the experiment on",
        default=["HH", "HL", "LH", "LL"],
    )
    parser.add_argument(
        "--seed",
        type=int or list,
        help="The random seed to use for the experiment",
        default=[
            0,
            1,
            2,
            4,
            8,
            16,
            32,
            40,
            64,
            128,
            256,
            512,
            1024,
            10,
            20,
            30,
            50,
            999,
            123,
            2025,
            30000,
            10000,
            20000,
            11000,
            23333,
            920083,
            908461,
            234815,
            882415,
            794609,
        ],
    )

    parser.add_argument(
        "--algo",
        type=str,
        help="The algorithm to run the experiment with",
    )

    parser.add_argument(
        "--enable_score_based_algo",
        # type=bool,
        action="store_true",
        help="If enabkle the score-based algorithm",
        default=False,
    )

    parser.add_argument(
        "--num_pre_selection",
        type=int,
        help="If the pre-selection is used",
        default=3,
    )

    parser.add_argument(
        "--device",
        type=str,
        help="The device to run the experiment on",
        default="cuda",
    )

    args = parser.parse_args()

    ds = args.dataset_name
    s = args.seed
    score_based_algo = args.enable_score_based_algo
    algo = args.algo
    num_pre_selection = args.num_pre_selection
    device = args.device

    if isinstance(s, list):
        for seed in s:
            torch.manual_seed(seed)
            np.random.seed(seed)
            random.seed(seed)
            os.environ["PYTHONHASHSEED"] = str(seed)
            if device == "cuda":
                torch.cuda.manual_seed(seed)
                torch.cuda.manual_seed_all(seed)

                torch.backends.cudnn.deterministic = True
                torch.backends.cudnn.benchmark = False

            if isinstance(ds, list):
                for dataset_name in ds:
                    print(f"Main process ID: {os.getpid()}")
                    if algo == "MTGP":
                        print("----------MTGP----------")
                        GPmain.main(dataset_name, seed)
                    elif algo == "KnnMTGP":
                        print("----------KnnMTGP----------")
                        KnnGPmain.main(dataset_name, seed)
                    elif algo == "NichingMTGP":
                        print("----------niching MTGP----------")
                        NichingGPmain.main(dataset_name, seed)
                    elif algo == "NichingGP_all_gen_test":
                        main_experiment_GP_all_generations_test_results.main(
                            dataset_name, seed, "NichingGP_all_gen_test"
                        )
                    elif algo == "GP_all_gen_test":
                        main_experiment_MTGP_all_generations_test_results.main(
                            dataset_name, seed, "GP_all_gen_test"
                        )
                    elif algo == "Knn_GP_all_gen_test":
                        main_experiment_knn_MTGP_all_generations_test_results.main(
                            dataset_name, seed, "Knn_GP_all_gen_test"
                        )
                    elif algo == "TransformerMTGP":
                        import TransformerMTGP.GPFC as TransformerGPmain

                        device = torch.device(device)
                        TransformerGPmain.main(
                            dataset_name,
                            seed,
                            num_pre_selection,
                            device,
                            score_based_algo,
                        )
                    elif algo == "transformerGP_all_gen_test":
                        main_experiment_transformerGP_all_generations_test_results.main(
                            dataset_name, seed, "transformerGP_all_gen_test"
                        )
                    elif algo == "MTGP_DRL_best_gen_test_for_CIM_paper":
                        main_experiment_GP_all_generations_test_results.main(
                            dataset_name, seed, "MTGP_DRL_best_gen_test_for_CIM_paper"
                        )
                    elif algo == "NichingMTGP_DRL_test":
                        main_experiment_NichingMTGP_Integrated_DRL.main(
                            dataset_name, seed
                        )
                    elif algo == "intermediate_DRL_R_S_single_agent_test":
                        main_experiment_NichingMTGP_Integrated_DRL_R_S_single_agent_with_intermediate.main(
                            dataset_name, seed
                        )
                    elif algo == "intermediate_DRL_R_test":
                        main_experiment_NichingMTGP_Integrated_DRL_R_with_intermediate.main(
                            dataset_name, seed
                        )
                    elif algo == "intermediate_DRL_S_test":
                        main_experiment_NichingMTGP_Integrated_DRL_S_with_intermediate.main(
                            dataset_name, seed
                        )
                    elif algo == "manualRule_test":
                        main_experiment_manualRule_using_validation.main(
                            dataset_name, seed
                        )
                    elif algo == "RL_test":
                        main_experiment_Integrated_DRL.main(dataset_name, seed)
                    elif (
                        algo == "GP_sequencing_RL_routing_test"
                    ):  # add by mengxu for revise the paper 2023.08.18
                        main_experiment_Integrated_DRL.main(dataset_name, seed)
                    elif algo == "RL_R":
                        main_training_R.training(dataset_name, seed)
                    elif algo == "RL_S":
                        main_training_S.training(dataset_name, seed)
                    elif algo == "RL_R_30":
                        # for 30 times training of RL_R
                        all_dataset_name = ["LH"]
                        for dataset_name in all_dataset_name:
                            for i in range(22, 30):
                                seed = i
                                main_training_R.training(dataset_name, seed)
                    elif algo == "RL_S_30":
                        # for 30 times training of RL_S
                        all_dataset_name = ["LL"]
                        for dataset_name in all_dataset_name:
                            for i in range(7, 11):
                                seed = i
                                main_training_S.training(dataset_name, seed)
                    elif algo == "GPRL_R_S":
                        main_training_R_S_GPrule.training(dataset_name, seed)
                    elif algo == "GPRL_R":
                        main_training_R_GPrule.training(dataset_name, seed)
                    elif algo == "GPRL_S":
                        main_training_S_GPrule.training(dataset_name, seed)
                    elif algo == "GPRL_S_online":
                        main_training_S_online_learning.training(dataset_name, seed)
                    elif algo == "GPRL_single_agent":
                        main_training_R_S_GPrule_single_agent.training(
                            dataset_name, seed
                        )
