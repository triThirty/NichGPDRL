import sys
from tabulate import tabulate
import pandas as pd

import MTGP.LoadIndividual
import GPLS.LoadIndividual
import GSGP.LoadIndividual
import main_experiment_MTGP
from wilcoxonTest.wilcoxonTest import doWilcoxonTest

sys.path


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
    algos = ['MTGP']
    runs = 30

    for dataset_index in range(len(all_dataset_name)):
        sum_MTGP = []
        sum_GPLS = []
        sum_GSGP = []
        dataset_name = all_dataset_name[dataset_index]
        print('\nDataset: ' + dataset_name + ': ')
        for run in range(runs):
            # print('Run ' + str(run) + ': ')
            for i in range(len(algos)):
                # print('Algo: ' + algos[i] + ': ')
                if algos[i] == 'MTGP':
                    min_fitness = MTGP.LoadIndividual.load_min_fitness(run, all_dataset_name[dataset_index])
                    if run==0:
                        sum_MTGP = min_fitness.T
                    else:
                        sum_MTGP = sum_MTGP + min_fitness.T
                    # print(min_fitness)
                elif algos[i] == 'GPLS':
                    min_fitness = GPLS.LoadIndividual.load_min_fitness(run, all_dataset_name[dataset_index])
                    if run == 0:
                        sum_GPLS = min_fitness.T
                    else:
                        sum_GPLS = sum_GPLS + min_fitness.T
                    # print(min_fitness)
                elif algos[i] == 'GSGP':
                    min_fitness = GSGP.LoadIndividual.load_min_fitness(run, all_dataset_name[dataset_index])
                    if run == 0:
                        sum_GSGP = min_fitness.T
                    else:
                        sum_GSGP = sum_GSGP + min_fitness.T

                    # print(min_fitness)
        # sum_dict = {'MTGP': sum_MTGP/runs, 'GPLS': sum_GPLS/runs}
        sum_dict = {'GP': sum_MTGP / runs}
        data = pd.DataFrame.from_dict(sum_dict)
        # print(data)

        # res = doWilcoxonTest(sum_MTGP, sum_GSGP, 0.05)
        # if res == 0:
        #     print('MTGP = GSGP')
        # elif res == 1:
        #     print('MTGP is significantly better than GSGP')
        # elif res == 2:
        #     print('GSGP is significantly better than MTGP')
        # else:
        #     print('There are something wrong here!')

        addressFinal = sys.path[
                           0] + '/experiment_result/scenario_' + dataset_name + '/mean_of_all_run_training_results_MTGP_' + dataset_name + '.xlsx'
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

