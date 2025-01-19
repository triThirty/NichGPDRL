import sys
from tabulate import tabulate
import pandas as pd

import MTGP.LoadIndividual
import GPLS.LoadIndividual
import GSGP.LoadIndividual
import main_experiment_MTGP
from wilcoxonTest.wilcoxonTest import doWilcoxonTest
import numpy as np
from time import  strftime
from time import gmtime

sys.path


def load_DRL_training_time(randomSeeds, dataSetName): # save individual as txt by mengxu
    routing_folder = "./routing_models/scenario_" + dataSetName + "/running_time_" + str(
        randomSeeds) + "_small_state_dict3wc6m.npy"  # modified by mengxu 2022.10.31
    routing_training_time = np.load(routing_folder)

    sequencing_folder = "./sequencing_models/scenario_" + dataSetName + "/running_time_" + str(
        randomSeeds) + "_small_state_dict3wc6m.npy"  # modified by mengxu 2022.10.31
    sequencing_training_time = np.load(sequencing_folder)

    if routing_training_time > sequencing_training_time:
        training_time = routing_training_time
    else:
        training_time = sequencing_training_time

    return training_time

if __name__ == '__main__':
    # dataset_name = str(sys.argv[1])
    # mean_of_all_run(dataset_name)

    # all_dataset_name = ['HH', 'HL', 'LH', 'LL']
    # # all_dataset_name = ['HL', 'LH', 'LL']
    # for i in range(len(all_dataset_name)):
    #     print('Result on dataset: ' + all_dataset_name[i])
    #     dataset_name = all_dataset_name[i]
    #     mean_of_all_run(dataset_name)

    all_dataset_name = ['HH', 'HL', 'LH', 'LL']
    # all_dataset_name = ['HH', 'HL']
    algos = ['MTGP', 'DRL']
    runs = 30

    for dataset_index in range(len(all_dataset_name)):
        sum_MTGP = []
        sum_GPLS = []
        sum_GSGP = []
        sum_DRL = []
        dataset_name = all_dataset_name[dataset_index]
        print('\nDataset: ' + dataset_name + ': ')
        for run in range(runs):
            # print('Run ' + str(run) + ': ')
            for i in range(len(algos)):
                # print('Algo: ' + algos[i] + ': ')
                if algos[i] == 'MTGP':
                    training_time = MTGP.LoadIndividual.load_training_time(run, all_dataset_name[dataset_index])
                    sum_MTGP.append(training_time)
                    # if run==0:
                    #     sum_MTGP = training_time
                    # else:
                    #     sum_MTGP = sum_MTGP + training_time
                    # print(min_fitness)
                elif algos[i] == 'GPLS':
                    training_time = GPLS.LoadIndividual.load_training_time(run, all_dataset_name[dataset_index])
                    sum_GPLS.append(training_time)
                    # if run == 0:
                    #     sum_GPLS = training_time
                    # else:
                    #     sum_GPLS = sum_GPLS + training_time
                    # print(min_fitness)
                elif algos[i] == 'GSGP':
                    training_time = GSGP.LoadIndividual.load_training_time(run, all_dataset_name[dataset_index])
                    sum_GSGP.append(training_time)
                    # if run == 0:
                    #     sum_GSGP = training_time
                    # else:
                    #     sum_GSGP = sum_GSGP + training_time
                    # print(min_fitness)
                elif algos[i] == 'DRL':
                    training_time = load_DRL_training_time(run, all_dataset_name[dataset_index])
                    sum_DRL.append(training_time)
        # sum_dict = {'MTGP': sum_MTGP/runs, 'GPLS': sum_GPLS/runs}
        sum_dict = {'MTGP': sum_MTGP, 'DRL': sum_DRL}
        data = pd.DataFrame.from_dict(sum_dict)
        # calculate the average time of 30 runs
        avge_MTGP = np.mean(sum_MTGP)
        std_MTGP = np.std(sum_MTGP)
        avge_DRL = np.mean(sum_DRL)
        std_DRL = np.std(sum_DRL)
        print("Training time of MTGP: " + str(avge_MTGP) + '(' + str(std_MTGP) + ')')
        print("Training time of DRL: " + str(avge_DRL) + '(' + str(std_DRL) + ')')
        # print("Training time of MTGP: " + strftime("%H:%M:%S", gmtime(avge_MTGP)))
        # print("Training time of DRL: " + strftime("%H:%M:%S", gmtime(avge_DRL)))


        res = doWilcoxonTest(sum_MTGP, sum_DRL, 0.05)
        if res == 0:
            print('MTGP = DRL')
        elif res == 1:
            print('MTGP is significantly better than DRL')
        elif res == 2:
            print('DRL is significantly better than MTGP')
        else:
            print('There are something wrong here!')

        addressFinal = sys.path[
                           0] + '/experiment_result/scenario_' + dataset_name + '/mean_of_all_run_training_time_MTGP_vs_DRL_' + dataset_name + '.xlsx'
        data.to_excel(addressFinal, index=False)

        # res = doWilcoxonTest(sum_MTGP, sum_GPLS, 0.05)
        # if res == 0:
        #     print('MTGP = GPLS')
        # elif res == 1:
        #     print('MTGP is significantly better than GPLS')
        # elif res == 2:
        #     print('GPLS is significantly better than MTGP')
        # else:
        #     print('There are something wrong here!')
        #
        # addressFinal = sys.path[
        #                    0] + '/experiment_result/scenario_' + dataset_name + '/mean_of_all_run_training_results_MTGP_vs_GPLS_' + dataset_name + '.xlsx'
        # data.to_excel(addressFinal, index=False)

