import sys

import torch

import MTGP.GPFC as GPmain
import NichingMTGP.GPFC as NichingGPmain
import TransformerMTGP.GPFC as TransformerGPmain
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

sys.path

if __name__ == "__main__":
    dataset_name = str(sys.argv[1])  # HH or HL or LH or LL
    seed = int(sys.argv[2])  # a random number, e.g., 0
    algo = str(sys.argv[3])  # as the following

    # dataset_name = "HH"
    # seed = 2
    # algo = "MTGP"
    # algo = "TransformerMTGP"
    # algo = "GP_all_gen_test"
    # algo = "NichingMTGP"
    torch.manual_seed(seed)
    np.random.seed(seed)

    # algo = 'transformerGP_all_gen_test'
    # algo = 'GP_all_gen_test'

    import os

    print(f"Main process ID: {os.getpid()}")
    if algo == "MTGP":
        print("----------MTGP----------")
        GPmain.main(dataset_name, seed)
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
    elif algo == "TransformerMTGP":
        TransformerGPmain.main(dataset_name, seed)
    elif algo == "transformerGP_all_gen_test":
        main_experiment_transformerGP_all_generations_test_results.main(
            dataset_name, seed, "transformerGP_all_gen_test"
        )
    elif algo == "MTGP_DRL_best_gen_test_for_CIM_paper":
        main_experiment_GP_all_generations_test_results.main(
            dataset_name, seed, "MTGP_DRL_best_gen_test_for_CIM_paper"
        )
    elif algo == "NichingMTGP_DRL_test":
        main_experiment_NichingMTGP_Integrated_DRL.main(dataset_name, seed)
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
        main_experiment_manualRule_using_validation.main(dataset_name, seed)
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
        main_training_R_S_GPrule_single_agent.training(dataset_name, seed)
