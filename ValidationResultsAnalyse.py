import sys
from tabulate import tabulate
import pandas as pd

import MTGP.LoadIndividual
import GPLS.LoadIndividual
import main_experiment_GPLS_MTGP_using_validation
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
    # all_dataset_name = ['HL']
    algos = ['MTGP']
    # algos = ['MTGP', 'GSGP']
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
                address = sys.path[
                              0] + '/experiment_result/scenario_' + dataset_name + '/all_gen_validation_results_MTGP_vs_GSGP_' + dataset_name + '_run_' + str(
                    run) + '.xlsx'
                objectives = pd.read_excel(address, engine='openpyxl')

                if algos[i] == 'MTGP':
                    if run==0:
                        sum_MTGP = objectives['MTGP']
                    else:
                        sum_MTGP = sum_MTGP + objectives['MTGP']
                elif algos[i] == 'GPLS':
                    if run == 0:
                        sum_GPLS = objectives['GPLS']
                    else:
                        sum_GPLS = sum_GPLS + objectives['GPLS']
                elif algos[i] == 'GSGP':
                    if run == 0:
                        sum_GSGP = objectives['GSGP']
                    else:
                        sum_GSGP = sum_GSGP + objectives['GSGP']

        sum_dict = {'MTGP': sum_MTGP / runs, 'GSGP': sum_GSGP / runs}
        data = pd.DataFrame.from_dict(sum_dict)
        # print(data)

        res = doWilcoxonTest(sum_MTGP, sum_GSGP, 0.05)
        if res == 0:
            print('MTGP = GSGP')
        elif res == 1:
            print('MTGP is significantly better than GSGP')
        elif res == 2:
            print('GSGP is significantly better than MTGP')
        else:
            print('There are something wrong here!')

        addressFinal = sys.path[
                           0] + '/experiment_result/scenario_' + dataset_name + '/mean_of_all_run_validation_results_MTGP_vs_GSGP_' + dataset_name + '.xlsx'
        data.to_excel(addressFinal, index=False)

        # sum_dict = {'MTGP': sum_MTGP/runs, 'GPLS': sum_GPLS/runs}
        # data = pd.DataFrame.from_dict(sum_dict)
        # # print(data)
        #
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
        #                    0] + '/experiment_result/scenario_' + dataset_name + '/mean_of_all_run_validation_results_MTGP_vs_GPLS_' + dataset_name + '.xlsx'
        # data.to_excel(addressFinal, index=False)

