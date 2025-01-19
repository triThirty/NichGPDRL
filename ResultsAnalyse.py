import os.path
import sys
from tabulate import tabulate
import pandas as pd
import numpy as np
import main_experiment_MTGP
from wilcoxonTest.wilcoxonTest import doWilcoxonTest

sys.path

def mean_of_all_run_all_gen_GP(dataset_name):

    runs = 30

    all30runs_all_gen_GP = []
    all_gen_GP = []
    # all30runs_MTGP_GPLS_RL.append([])
    # all30runs_MTGP_GPLS_RL.append([])
    # all30runs_MTGP_GPLS_RL.append([])

    sum_sum_sorted = []
    sum_sum = []
    for i in range(runs):
        seed = i
        address = sys.path[
                      0] + '/experiment_result/scenario_' + dataset_name + '/GP_all_gen_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        tardiness_eachRun_eachInstance = pd.read_excel(address, sheet_name='sum', engine='openpyxl')

        sum = tardiness_eachRun_eachInstance.mean()  # get mean on all the instances
        if(seed == 0):
            sum_sum = sum
            for gen in range(0, 51):
                GPgen = 'GP_gen_' + str(gen)
                all_gen_GP.append(sum[GPgen])
        else:
            sum_sum = sum_sum + sum
            for gen in range(0, 51):
                GPgen = 'GP_gen_' + str(gen)
                all_gen_GP[gen] = all_gen_GP[gen] + sum[GPgen]

        # for gen in range(0,51):
        #     GPgen = 'GP_gen_' + str(gen)
        #     all_gen_GP.append(sum[GPgen])
        # all30runs_all_gen_GP.append(all_gen_GP)
        # all30runs_MTGP_GPLS_RL[1].append(sum['gen_best_GPLS_test'])
        # all30runs_MTGP_GPLS_RL[2].append(sum['Integrated_DRL'])

        # sum_sorted = sum.sort_values()
        # if (seed == 0):
        #     sum_sum_sorted = sum_sorted
        # else:
        #     sum_sum_sorted = sum_sum_sorted + sum_sorted

    for gen in range(0, 51):
        all_gen_GP[gen] = all_gen_GP[gen] / runs
    sum_dict = {'GP': all_gen_GP}
    # sum_dict = {'MTGP': all30runs_MTGP_GPLS_RL[0], 'GPLS': all30runs_MTGP_GPLS_RL[1], 'RL': all30runs_MTGP_GPLS_RL[2]}
    data = pd.DataFrame.from_dict(sum_dict)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_run_all_gen_test_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

    sum_sum = sum_sum / runs
    print(sum_sum)
    # print(sum_sum.sort_values())

    # print('----MTGP,  GPLS,  DRL----')
    # mean_MTGP = np.mean(all30runs_all_gen_GP)
    # std_MTGP = np.std(all30runs_all_gen_GP)
    # print(str(mean_MTGP) + '(' + str(std_MTGP) + ')')

    # after_sorted = sum_sum.sort_values()

    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_run_all_gen_test_GP_' + dataset_name + '.xlsx'
    sum_sum.to_excel(addressFinal, index=False)

    # print('----MTGP,  GPLS,  DRL----')
    # mean_MTGP = np.mean(all30runs_MTGP_GPLS_RL[0])
    # std_MTGP = np.std(all30runs_MTGP_GPLS_RL[0])
    # mean_GPLS = np.mean(all30runs_MTGP_GPLS_RL[1])
    # std_GPLS = np.std(all30runs_MTGP_GPLS_RL[1])
    # mean_DRL = np.mean(all30runs_MTGP_GPLS_RL[2])
    # std_DRL = np.std(all30runs_MTGP_GPLS_RL[2])
    # print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
    #       str(mean_GPLS) + '(' + str(std_GPLS) + '), ' +
    #       str(mean_DRL) + '(' + str(std_DRL) + ')')
    #
    # after_sorted = sum_sum.sort_values()
    #
    # print('\nCompare MTGP with GPLS:')
    # res = doWilcoxonTest(all30runs_MTGP_GPLS_RL[0], all30runs_MTGP_GPLS_RL[1], 0.05)
    # if res == 0:
    #     print('MTGP = GPLS')
    # elif res == 1:
    #     print('MTGP is significantly better than GPLS')
    # elif res == 2:
    #     print('GPLS is significantly better than MTGP')
    # else:
    #     print('There are something wrong here!')
    #
    # print('\nCompare MTGP with RL:')
    # res = doWilcoxonTest(all30runs_MTGP_GPLS_RL[0], all30runs_MTGP_GPLS_RL[2], 0.05)
    # if res == 0:
    #     print('MTGP = RL')
    # elif res == 1:
    #     print('MTGP is significantly better than RL')
    # elif res == 2:
    #     print('RL is significantly better than MTGP')
    # else:
    #     print('There are something wrong here!')
    #
    # print('\nCompare GPLS with RL:')
    # res = doWilcoxonTest(all30runs_MTGP_GPLS_RL[1], all30runs_MTGP_GPLS_RL[2], 0.05)
    # if res == 0:
    #     print('GPLS = RL')
    # elif res == 1:
    #     print('GPLS is significantly better than RL')
    # elif res == 2:
    #     print('RL is significantly better than GPLS')
    # else:
    #     print('There are something wrong here!')
    #
    #
    # addressFinal = sys.path[0] + '/experiment_result/scenario_' + dataset_name + '/all_run_with_validation_MTGP_vs_GPLS_vs_RL_' + dataset_name + '.xlsx'

    # after_sorted.to_excel(addressFinal, index = True, index_label=['Algo', 'tardiness'])
    # print(sum_sum_sorted)

def mean_of_all_run_MTGP_GPLS(dataset_name):

    runs = 30

    all30runs_MTGP_GPLS_RL = []
    all30runs_MTGP_GPLS_RL.append([])
    all30runs_MTGP_GPLS_RL.append([])
    all30runs_MTGP_GPLS_RL.append([])

    sum_sum_sorted = []
    sum_sum = []
    for i in range(runs):
        seed = i
        address = sys.path[
                      0] + '/experiment_result/scenario_' + dataset_name + '/GP_all_gen_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        tardiness_eachRun_eachInstance = pd.read_excel(address, sheet_name='sum', engine='openpyxl')

        sum = tardiness_eachRun_eachInstance.mean()  # get mean on all the instances
        if(seed == 0):
            sum_sum = sum
        else:
            sum_sum = sum_sum + sum

        for gen in range(0,51):
            GPgen = 'GP_gen_' + str(gen)
        all30runs_MTGP_GPLS_RL[0].append(sum['gen_best_MTGP_test'])
        # all30runs_MTGP_GPLS_RL[1].append(sum['gen_best_GPLS_test'])
        # all30runs_MTGP_GPLS_RL[2].append(sum['Integrated_DRL'])

        # sum_sorted = sum.sort_values()
        # if (seed == 0):
        #     sum_sum_sorted = sum_sorted
        # else:
        #     sum_sum_sorted = sum_sum_sorted + sum_sorted

    sum_dict = {'MTGP': all30runs_MTGP_GPLS_RL[0]}
    # sum_dict = {'MTGP': all30runs_MTGP_GPLS_RL[0], 'GPLS': all30runs_MTGP_GPLS_RL[1], 'RL': all30runs_MTGP_GPLS_RL[2]}
    data = pd.DataFrame.from_dict(sum_dict)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_run_test_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

    sum_sum = sum_sum / runs
    print(sum_sum.sort_values())

    print('----MTGP,  GPLS,  DRL----')
    mean_MTGP = np.mean(all30runs_MTGP_GPLS_RL[0])
    std_MTGP = np.std(all30runs_MTGP_GPLS_RL[0])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + ')')

    after_sorted = sum_sum.sort_values()

    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_run_with_validation_MTGP_' + dataset_name + '.xlsx'

    # print('----MTGP,  GPLS,  DRL----')
    # mean_MTGP = np.mean(all30runs_MTGP_GPLS_RL[0])
    # std_MTGP = np.std(all30runs_MTGP_GPLS_RL[0])
    # mean_GPLS = np.mean(all30runs_MTGP_GPLS_RL[1])
    # std_GPLS = np.std(all30runs_MTGP_GPLS_RL[1])
    # mean_DRL = np.mean(all30runs_MTGP_GPLS_RL[2])
    # std_DRL = np.std(all30runs_MTGP_GPLS_RL[2])
    # print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
    #       str(mean_GPLS) + '(' + str(std_GPLS) + '), ' +
    #       str(mean_DRL) + '(' + str(std_DRL) + ')')
    #
    # after_sorted = sum_sum.sort_values()
    #
    # print('\nCompare MTGP with GPLS:')
    # res = doWilcoxonTest(all30runs_MTGP_GPLS_RL[0], all30runs_MTGP_GPLS_RL[1], 0.05)
    # if res == 0:
    #     print('MTGP = GPLS')
    # elif res == 1:
    #     print('MTGP is significantly better than GPLS')
    # elif res == 2:
    #     print('GPLS is significantly better than MTGP')
    # else:
    #     print('There are something wrong here!')
    #
    # print('\nCompare MTGP with RL:')
    # res = doWilcoxonTest(all30runs_MTGP_GPLS_RL[0], all30runs_MTGP_GPLS_RL[2], 0.05)
    # if res == 0:
    #     print('MTGP = RL')
    # elif res == 1:
    #     print('MTGP is significantly better than RL')
    # elif res == 2:
    #     print('RL is significantly better than MTGP')
    # else:
    #     print('There are something wrong here!')
    #
    # print('\nCompare GPLS with RL:')
    # res = doWilcoxonTest(all30runs_MTGP_GPLS_RL[1], all30runs_MTGP_GPLS_RL[2], 0.05)
    # if res == 0:
    #     print('GPLS = RL')
    # elif res == 1:
    #     print('GPLS is significantly better than RL')
    # elif res == 2:
    #     print('RL is significantly better than GPLS')
    # else:
    #     print('There are something wrong here!')
    #
    #
    # addressFinal = sys.path[0] + '/experiment_result/scenario_' + dataset_name + '/all_run_with_validation_MTGP_vs_GPLS_vs_RL_' + dataset_name + '.xlsx'
    after_sorted.to_excel(addressFinal, index=False)
    # after_sorted.to_excel(addressFinal, index = True, index_label=['Algo', 'tardiness'])
    # print(sum_sum_sorted)

def mean_of_all_run_MTGP_GSGP(dataset_name):

    runs = 30

    all30runs_MTGP_GSGP_RL = []
    all30runs_MTGP_GSGP_RL.append([])
    all30runs_MTGP_GSGP_RL.append([])
    all30runs_MTGP_GSGP_RL.append([])

    sum_sum_sorted = []
    sum_sum = []
    for i in range(runs):
        seed = i
        address = sys.path[
                      0] + '/experiment_result/scenario_' + dataset_name + '/MTGP_vs_GPLS_vs_RL_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        tardiness_eachRun_eachInstance = pd.read_excel(address, sheet_name='sum', engine='openpyxl')

        sum = tardiness_eachRun_eachInstance.mean()  # get mean on all the instances
        if(seed == 0):
            sum_sum = sum
        else:
            sum_sum = sum_sum + sum

        all30runs_MTGP_GSGP_RL[0].append(sum['gen_best_MTGP_test'])
        all30runs_MTGP_GSGP_RL[1].append(sum['gen_best_GSGP_test'])
        all30runs_MTGP_GSGP_RL[2].append(sum['Integrated_DRL'])

        # sum_sorted = sum.sort_values()
        # if (seed == 0):
        #     sum_sum_sorted = sum_sorted
        # else:
        #     sum_sum_sorted = sum_sum_sorted + sum_sorted

    sum_dict = {'MTGP': all30runs_MTGP_GSGP_RL[0], 'GSGP': all30runs_MTGP_GSGP_RL[1], 'RL': all30runs_MTGP_GSGP_RL[2]}
    data = pd.DataFrame.from_dict(sum_dict)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_run_test_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

    sum_sum = sum_sum / runs
    print(sum_sum.sort_values())

    print('----MTGP,  GSGP,  DRL----')
    mean_MTGP = np.mean(all30runs_MTGP_GSGP_RL[0])
    std_MTGP = np.std(all30runs_MTGP_GSGP_RL[0])
    mean_GSGP = np.mean(all30runs_MTGP_GSGP_RL[1])
    std_GSGP = np.std(all30runs_MTGP_GSGP_RL[1])
    mean_DRL = np.mean(all30runs_MTGP_GSGP_RL[2])
    std_DRL = np.std(all30runs_MTGP_GSGP_RL[2])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_GSGP) + '(' + str(std_GSGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    after_sorted = sum_sum.sort_values()

    print('\nCompare MTGP with GSGP:')
    res = doWilcoxonTest(all30runs_MTGP_GSGP_RL[0], all30runs_MTGP_GSGP_RL[1], 0.05)
    if res == 0:
        print('MTGP = GSGP')
    elif res == 1:
        print('MTGP is significantly better than GSGP')
    elif res == 2:
        print('GSGP is significantly better than MTGP')
    else:
        print('There are something wrong here!')

    print('\nCompare MTGP with RL:')
    res = doWilcoxonTest(all30runs_MTGP_GSGP_RL[0], all30runs_MTGP_GSGP_RL[2], 0.05)
    if res == 0:
        print('MTGP = RL')
    elif res == 1:
        print('MTGP is significantly better than RL')
    elif res == 2:
        print('RL is significantly better than MTGP')
    else:
        print('There are something wrong here!')

    print('\nCompare GSGP with RL:')
    res = doWilcoxonTest(all30runs_MTGP_GSGP_RL[1], all30runs_MTGP_GSGP_RL[2], 0.05)
    if res == 0:
        print('GSGP = RL')
    elif res == 1:
        print('GSGP is significantly better than RL')
    elif res == 2:
        print('RL is significantly better than GSGP')
    else:
        print('There are something wrong here!')


    addressFinal = sys.path[0] + '/experiment_result/scenario_' + dataset_name + '/all_run_with_validation_MTGP_vs_GSGP_vs_RL_' + dataset_name + '.xlsx'
    after_sorted.to_excel(addressFinal, index=False)
    # after_sorted.to_excel(addressFinal, index = True, index_label=['Algo', 'tardiness'])
    # print(sum_sum_sorted)

def mean_of_all_run_MTGP_RL(dataset_name):

    runs = 30

    all30runs_MTGP_RL = []
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])

    sum_sum_sorted = []
    sum_sum = []
    for i in range(runs):
        seed = i
        address = sys.path[
                      0] + '/experiment_result/scenario_' + dataset_name + '/best_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        tardiness_eachRun_eachInstance = pd.read_excel(address, sheet_name='sum', engine='openpyxl')

        sum = tardiness_eachRun_eachInstance.mean()  # get mean on all the instances
        if(seed == 0):
            sum_sum = sum
        else:
            sum_sum = sum_sum + sum

        all30runs_MTGP_RL[0].append(sum['gen_best_MTGP_test'])
        all30runs_MTGP_RL[1].append(sum['Integrated_DRL'])

        # sum_sorted = sum.sort_values()
        # if (seed == 0):
        #     sum_sum_sorted = sum_sorted
        # else:
        #     sum_sum_sorted = sum_sum_sorted + sum_sorted

    sum_dict = {'GP': all30runs_MTGP_RL[0], 'RL': all30runs_MTGP_RL[1]}
    data = pd.DataFrame.from_dict(sum_dict)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_run_test_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

    sum_sum = sum_sum / runs
    print(sum_sum.sort_values())

    print('----MTGP,  DRL----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[0])
    std_MTGP = np.std(all30runs_MTGP_RL[0])
    mean_DRL = np.mean(all30runs_MTGP_RL[1])
    std_DRL = np.std(all30runs_MTGP_RL[1])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    after_sorted = sum_sum.sort_values()

    print('\nCompare MTGP with RL:')
    res = doWilcoxonTest(all30runs_MTGP_RL[0], all30runs_MTGP_RL[1], 0.05)
    if res == 0:
        print('MTGP = RL')
    elif res == 1:
        print('MTGP is significantly better than RL')
    elif res == 2:
        print('RL is significantly better than MTGP')
    else:
        print('There are something wrong here!')


    addressFinal = sys.path[0] + '/experiment_result/scenario_' + dataset_name + '/all_run_without_validation_MTGP_RL_' + dataset_name + '.xlsx'
    after_sorted.to_excel(addressFinal, index=False)
    # after_sorted.to_excel(addressFinal, index = True, index_label=['Algo', 'tardiness'])
    # print(sum_sum_sorted)

def mean_of_top_n_NichingMTGP_RL(dataset_name):

    runs = 30

    all30runs_MTGP_RL = []
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    # all30runs_MTGP_RL.append([])
    # all30runs_MTGP_RL.append([])
    # all30runs_MTGP_RL.append([])
    # all30runs_MTGP_RL.append([])

    sum_sum_sorted = []
    sum_sum = []
    for i in range(runs):
        seed = i
        address = sys.path[
                      0] + '/experiment_result/scenario_' + dataset_name + '/intermediate_DRL_R_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        # address = sys.path[
        #               0] + '/experiment_result/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
        #     seed) + '_val.xlsx'
        if not os.path.exists(address):
            print("run " + str(i)+ "not exist!")
            continue

        tardiness_eachRun_eachInstance = pd.read_excel(address, sheet_name='sum', engine='openpyxl')

        sum = tardiness_eachRun_eachInstance.mean()  # get mean on all the instances
        if(seed == 0):
            sum_sum = sum
        else:
            sum_sum = sum_sum + sum

        all30runs_MTGP_RL[0].append(sum['Nichtop_0'])
        all30runs_MTGP_RL[1].append(sum['Nichtop_1'])
        all30runs_MTGP_RL[2].append(sum['Nichtop_2'])
        all30runs_MTGP_RL[3].append(sum['Nichtop_3'])
        all30runs_MTGP_RL[4].append(sum['Nichtop_4'])
        all30runs_MTGP_RL[5].append(sum['Nichtop_5'])
        all30runs_MTGP_RL[6].append(sum['Nichtop_6'])
        all30runs_MTGP_RL[7].append(sum['Nichtop_7'])
        all30runs_MTGP_RL[8].append(sum['Nichtop_6'])
        all30runs_MTGP_RL[9].append(sum['Nichtop_7'])
        all30runs_MTGP_RL[10].append(sum['iter_315'])
        all30runs_MTGP_RL[11].append(sum['iter_316'])
        all30runs_MTGP_RL[12].append(sum['iter_317'])
        all30runs_MTGP_RL[13].append(sum['iter_318'])
        all30runs_MTGP_RL[14].append(sum['iter_319'])

    # sum_dict = {'GP0': all30runs_MTGP_RL[0], 'GP1': all30runs_MTGP_RL[1], 'GP2': all30runs_MTGP_RL[2],
    #             'GP3': all30runs_MTGP_RL[3], 'RL': all30runs_MTGP_RL[4]}

    sum_dict = {'GP0': all30runs_MTGP_RL[0],'GP1': all30runs_MTGP_RL[1],'GP2': all30runs_MTGP_RL[2],
                'GP3': all30runs_MTGP_RL[3], 'GP4': all30runs_MTGP_RL[4],'GP5': all30runs_MTGP_RL[5],
                'GP6': all30runs_MTGP_RL[6],'GP7': all30runs_MTGP_RL[7],'GP8': all30runs_MTGP_RL[8],
                'GP9': all30runs_MTGP_RL[9],'DRL1': all30runs_MTGP_RL[10],'DRL2': all30runs_MTGP_RL[11],
                'DRL3': all30runs_MTGP_RL[12],'DRL4': all30runs_MTGP_RL[13],'DRL5': all30runs_MTGP_RL[14]}
    data = pd.DataFrame.from_dict(sum_dict)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_run_test_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

    sum_sum = sum_sum / runs
    # print(sum_sum.sort_values())

    print('----GP0,  NichGPRL----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[0])
    std_MTGP = np.std(all30runs_MTGP_RL[0])
    mean_DRL = np.mean(all30runs_MTGP_RL[4])
    std_DRL = np.std(all30runs_MTGP_RL[4])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    after_sorted = sum_sum.sort_values()

    print('\nCompare GP0 with NichGPRL:')
    res = doWilcoxonTest(all30runs_MTGP_RL[0], all30runs_MTGP_RL[4], 0.05)
    if res == 0:
        print('GP0 = NichGPRL')
    elif res == 1:
        print('GP0 is significantly better than NichGPRL')
    elif res == 2:
        print('NichGPRL is significantly better than GP0')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----GP1,  NichGPRL----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[1])
    std_MTGP = np.std(all30runs_MTGP_RL[1])
    mean_DRL = np.mean(all30runs_MTGP_RL[4])
    std_DRL = np.std(all30runs_MTGP_RL[4])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    after_sorted = sum_sum.sort_values()

    print('\nCompare GP1 with NichGPRL:')
    res = doWilcoxonTest(all30runs_MTGP_RL[1], all30runs_MTGP_RL[4], 0.05)
    if res == 0:
        print('GP1 = NichGPRL')
    elif res == 1:
        print('GP1 is significantly better than NichGPRL')
    elif res == 2:
        print('NichGPRL is significantly better than GP1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----GP2,  NichGPRL----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[2])
    std_MTGP = np.std(all30runs_MTGP_RL[2])
    mean_DRL = np.mean(all30runs_MTGP_RL[4])
    std_DRL = np.std(all30runs_MTGP_RL[4])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    after_sorted = sum_sum.sort_values()

    print('\nCompare GP2 with NichGPRL:')
    res = doWilcoxonTest(all30runs_MTGP_RL[2], all30runs_MTGP_RL[4], 0.05)
    if res == 0:
        print('GP2 = NichGPRL')
    elif res == 1:
        print('GP2 is significantly better than NichGPRL')
    elif res == 2:
        print('NichGPRL is significantly better than GP2')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----GP3,  NichGPRL----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[3])
    std_MTGP = np.std(all30runs_MTGP_RL[3])
    mean_DRL = np.mean(all30runs_MTGP_RL[4])
    std_DRL = np.std(all30runs_MTGP_RL[4])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    after_sorted = sum_sum.sort_values()

    print('\nCompare GP3 with NichGPRL:')
    res = doWilcoxonTest(all30runs_MTGP_RL[3], all30runs_MTGP_RL[4], 0.05)
    if res == 0:
        print('GP3 = NichGPRL')
    elif res == 1:
        print('GP3 is significantly better than NichGPRL')
    elif res == 2:
        print('NichGPRL is significantly better than GP3')
    else:
        print('There are something wrong here!')

def mean_of_top_n_NichingMTGP_all_RL(dataset_name):

    runs = 30

    all30runs_MTGP_RL = []
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])

    for i in range(runs):
        seed = i
        # address = sys.path[
        #               0] + '/experiment_result/scenario_' + dataset_name + '/intermediate_DRL_R_test_' + dataset_name + '_run_' + str(
        #     seed) + '_val.xlsx'
        address_RL1 = sys.path[0] + '/experiment_result/New1-OriginalDRL/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL1):
            print("run " + str(i)+ "not exist!")
            continue
        address_RL2 = sys.path[
                          0] + '/experiment_result/New3-OriginalDRL-GPsequencing/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL2):
            print("run " + str(i) + "not exist!")
            continue
        address_RL3 = sys.path[
                          0] + '/experiment_result/SequencingNichGProut1/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL3):
            print("run " + str(i) + "not exist!")
            continue
        address_RL4 = sys.path[
                          0] + '/experiment_result/SequencingNichGProut2/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL4):
            print("run " + str(i) + "not exist!")
            continue
        address_RL5 = sys.path[
                          0] + '/experiment_result/SequencingNichGProut3/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL5):
            print("run " + str(i) + "not exist!")
            continue
        address_RL6 = sys.path[
                          0] + '/experiment_result/SequencingNichGProut4/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL6):
            print("run " + str(i) + "not exist!")
            continue

        tardiness_eachRun_eachInstance_RL1 = pd.read_excel(address_RL1, sheet_name='sum', engine='openpyxl')
        sum_R1 = tardiness_eachRun_eachInstance_RL1.mean()
        tardiness_eachRun_eachInstance_RL2 = pd.read_excel(address_RL2, sheet_name='sum', engine='openpyxl')
        sum_R2 = tardiness_eachRun_eachInstance_RL2.mean()
        tardiness_eachRun_eachInstance_RL3 = pd.read_excel(address_RL3, sheet_name='sum', engine='openpyxl')
        sum_R3 = tardiness_eachRun_eachInstance_RL3.mean()
        tardiness_eachRun_eachInstance_RL4 = pd.read_excel(address_RL4, sheet_name='sum', engine='openpyxl')
        sum_R4 = tardiness_eachRun_eachInstance_RL4.mean()
        tardiness_eachRun_eachInstance_RL5 = pd.read_excel(address_RL5, sheet_name='sum', engine='openpyxl')
        sum_R5 = tardiness_eachRun_eachInstance_RL5.mean()
        tardiness_eachRun_eachInstance_RL6 = pd.read_excel(address_RL6, sheet_name='sum', engine='openpyxl')
        sum_R6 = tardiness_eachRun_eachInstance_RL6.mean()

        all30runs_MTGP_RL[0].append(sum_R4['Nichtop_0'])
        all30runs_MTGP_RL[1].append(sum_R4['Nichtop_1'])
        all30runs_MTGP_RL[2].append(sum_R4['Nichtop_2'])
        all30runs_MTGP_RL[3].append(sum_R4['Nichtop_3'])
        all30runs_MTGP_RL[4].append(sum_R1['Integrated_DRL'])
        all30runs_MTGP_RL[5].append(sum_R2['Integrated_DRL'])
        all30runs_MTGP_RL[6].append(sum_R3['Integrated_DRL'])
        all30runs_MTGP_RL[7].append(sum_R4['Integrated_DRL'])
        all30runs_MTGP_RL[8].append(sum_R5['Integrated_DRL'])
        all30runs_MTGP_RL[9].append(sum_R6['Integrated_DRL'])


    # sum_dict = {'GP0': all30runs_MTGP_RL[0], 'GP1': all30runs_MTGP_RL[1], 'GP2': all30runs_MTGP_RL[2],
    #             'GP3': all30runs_MTGP_RL[3], 'DRL1': all30runs_MTGP_RL[4], 'DRL2': all30runs_MTGP_RL[5]}
    sum_dict = {'GP0': all30runs_MTGP_RL[0],'GP1': all30runs_MTGP_RL[1],'GP2': all30runs_MTGP_RL[2],
                'GP3': all30runs_MTGP_RL[3],'DRL1': all30runs_MTGP_RL[4],'DRL2': all30runs_MTGP_RL[5],
                'DGP1': all30runs_MTGP_RL[6],'DGP2': all30runs_MTGP_RL[7],'DGP3': all30runs_MTGP_RL[8],
                'DGP4': all30runs_MTGP_RL[9]}
    data = pd.DataFrame.from_dict(sum_dict)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_run_test_NichGP_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

    print('----NichGP1,  NichGP2, NichGP3,  NichGP4----')
    mean_NichGP1 = np.mean(all30runs_MTGP_RL[0])
    std_NichGP1 = np.std(all30runs_MTGP_RL[0])
    mean_NichGP2 = np.mean(all30runs_MTGP_RL[1])
    std_NichGP2 = np.std(all30runs_MTGP_RL[1])
    mean_NichGP3 = np.mean(all30runs_MTGP_RL[2])
    std_NichGP3 = np.std(all30runs_MTGP_RL[2])
    mean_NichGP4 = np.mean(all30runs_MTGP_RL[3])
    std_NichGP4 = np.std(all30runs_MTGP_RL[3])
    print(str(mean_NichGP1) + '(' + str(std_NichGP1) + '), ' +
          str(mean_NichGP2) + '(' + str(std_NichGP2) + '), ' +
          str(mean_NichGP3) + '(' + str(std_NichGP3) + '), ' +
          str(mean_NichGP4) + '(' + str(std_NichGP4) + ')'
          )

    print('----DRL1,  DRL2----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[5])
    std_DRL = np.std(all30runs_MTGP_RL[5])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')


    print('\nCompare DRL1 with DRL2:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[5], 0.05)
    if res == 0:
        print('DRL1 = DRL2')
    elif res == 1:
        print('DRL1 is significantly better than DRL2')
    elif res == 2:
        print('DRL2 is significantly better than DRL1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL1,  DRL3----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[6])
    std_DRL = np.std(all30runs_MTGP_RL[6])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL1 with DRL3:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[6], 0.05)
    if res == 0:
        print('DRL1 = DRL3')
    elif res == 1:
        print('DRL1 is significantly better than DRL3')
    elif res == 2:
        print('DRL3 is significantly better than DRL1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL1,  DRL4----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL1 with DRL4:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('DRL1 = DRL4')
    elif res == 1:
        print('DRL1 is significantly better than DRL4')
    elif res == 2:
        print('DRL4 is significantly better than DRL1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL1,  DRL5----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[8])
    std_DRL = np.std(all30runs_MTGP_RL[8])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL1 with DRL5:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[8], 0.05)
    if res == 0:
        print('DRL1 = DRL5')
    elif res == 1:
        print('DRL1 is significantly better than DRL5')
    elif res == 2:
        print('DRL5 is significantly better than DRL1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL1,  DRL6----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[9])
    std_DRL = np.std(all30runs_MTGP_RL[9])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL1 with DRL6:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[9], 0.05)
    if res == 0:
        print('DRL1 = DRL6')
    elif res == 1:
        print('DRL1 is significantly better than DRL6')
    elif res == 2:
        print('DRL6 is significantly better than DRL1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL2,  DRL3----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[5])
    std_MTGP = np.std(all30runs_MTGP_RL[5])
    mean_DRL = np.mean(all30runs_MTGP_RL[6])
    std_DRL = np.std(all30runs_MTGP_RL[6])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL2 with DRL3:')
    res = doWilcoxonTest(all30runs_MTGP_RL[5], all30runs_MTGP_RL[6], 0.05)
    if res == 0:
        print('DRL2 = DRL3')
    elif res == 1:
        print('DRL2 is significantly better than DRL3')
    elif res == 2:
        print('DRL3 is significantly better than DRL2')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL2,  DRL4----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[5])
    std_MTGP = np.std(all30runs_MTGP_RL[5])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL2 with DRL4:')
    res = doWilcoxonTest(all30runs_MTGP_RL[5], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('DRL2 = DRL4')
    elif res == 1:
        print('DRL2 is significantly better than DRL4')
    elif res == 2:
        print('DRL4 is significantly better than DRL2')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL2,  DRL5----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[5])
    std_MTGP = np.std(all30runs_MTGP_RL[5])
    mean_DRL = np.mean(all30runs_MTGP_RL[8])
    std_DRL = np.std(all30runs_MTGP_RL[8])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL2 with DRL5:')
    res = doWilcoxonTest(all30runs_MTGP_RL[5], all30runs_MTGP_RL[8], 0.05)
    if res == 0:
        print('DRL2 = DRL5')
    elif res == 1:
        print('DRL2 is significantly better than DRL5')
    elif res == 2:
        print('DRL5 is significantly better than DRL2')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL2,  DRL6----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[5])
    std_MTGP = np.std(all30runs_MTGP_RL[5])
    mean_DRL = np.mean(all30runs_MTGP_RL[9])
    std_DRL = np.std(all30runs_MTGP_RL[9])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL2 with DRL6:')
    res = doWilcoxonTest(all30runs_MTGP_RL[5], all30runs_MTGP_RL[9], 0.05)
    if res == 0:
        print('DRL2 = DRL6')
    elif res == 1:
        print('DRL2 is significantly better than DRL6')
    elif res == 2:
        print('DRL6 is significantly better than DRL2')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL3,  DRL4----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[6])
    std_MTGP = np.std(all30runs_MTGP_RL[6])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL3 with DRL4:')
    res = doWilcoxonTest(all30runs_MTGP_RL[6], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('DRL3 = DRL4')
    elif res == 1:
        print('DRL3 is significantly better than DRL4')
    elif res == 2:
        print('DRL4 is significantly better than DRL3')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL3,  DRL5----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[6])
    std_MTGP = np.std(all30runs_MTGP_RL[6])
    mean_DRL = np.mean(all30runs_MTGP_RL[8])
    std_DRL = np.std(all30runs_MTGP_RL[8])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL3 with DRL5:')
    res = doWilcoxonTest(all30runs_MTGP_RL[6], all30runs_MTGP_RL[8], 0.05)
    if res == 0:
        print('DRL3 = DRL5')
    elif res == 1:
        print('DRL3 is significantly better than DRL5')
    elif res == 2:
        print('DRL5 is significantly better than DRL3')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL3,  DRL6----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[6])
    std_MTGP = np.std(all30runs_MTGP_RL[6])
    mean_DRL = np.mean(all30runs_MTGP_RL[9])
    std_DRL = np.std(all30runs_MTGP_RL[9])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL3 with DRL6:')
    res = doWilcoxonTest(all30runs_MTGP_RL[6], all30runs_MTGP_RL[9], 0.05)
    if res == 0:
        print('DRL3 = DRL6')
    elif res == 1:
        print('DRL3 is significantly better than DRL6')
    elif res == 2:
        print('DRL6 is significantly better than DRL3')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL4,  DRL5----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[7])
    std_MTGP = np.std(all30runs_MTGP_RL[7])
    mean_DRL = np.mean(all30runs_MTGP_RL[8])
    std_DRL = np.std(all30runs_MTGP_RL[8])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL4 with DRL5:')
    res = doWilcoxonTest(all30runs_MTGP_RL[7], all30runs_MTGP_RL[8], 0.05)
    if res == 0:
        print('DRL4 = DRL5')
    elif res == 1:
        print('DRL4 is significantly better than DRL5')
    elif res == 2:
        print('DRL5 is significantly better than DRL4')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL4,  DRL6----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[7])
    std_MTGP = np.std(all30runs_MTGP_RL[7])
    mean_DRL = np.mean(all30runs_MTGP_RL[9])
    std_DRL = np.std(all30runs_MTGP_RL[9])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL4 with DRL6:')
    res = doWilcoxonTest(all30runs_MTGP_RL[7], all30runs_MTGP_RL[9], 0.05)
    if res == 0:
        print('DRL4 = DRL6')
    elif res == 1:
        print('DRL4 is significantly better than DRL6')
    elif res == 2:
        print('DRL6 is significantly better than DRL4')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL5,  DRL6----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[8])
    std_MTGP = np.std(all30runs_MTGP_RL[8])
    mean_DRL = np.mean(all30runs_MTGP_RL[9])
    std_DRL = np.std(all30runs_MTGP_RL[9])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL5 with DRL6:')
    res = doWilcoxonTest(all30runs_MTGP_RL[8], all30runs_MTGP_RL[9], 0.05)
    if res == 0:
        print('DRL5 = DRL6')
    elif res == 1:
        print('DRL5 is significantly better than DRL6')
    elif res == 2:
        print('DRL6 is significantly better than DRL5')
    else:
        print('There are something wrong here!')

    print('--------------------------------------------------------------------------------')
    print('----NichGP1,  DRL4----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[0])
    std_MTGP = np.std(all30runs_MTGP_RL[0])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare NichGP1 with DRL4:')
    res = doWilcoxonTest(all30runs_MTGP_RL[0], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('NichGP1 = DRL4')
    elif res == 1:
        print('NichGP1 is significantly better than DRL4')
    elif res == 2:
        print('DRL4 is significantly better than NichGP1')
    else:
        print('There are something wrong here!')

    print('--------------------------------------------------------------------------------')
    print('----NichGP1,  DRL5----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[0])
    std_MTGP = np.std(all30runs_MTGP_RL[0])
    mean_DRL = np.mean(all30runs_MTGP_RL[8])
    std_DRL = np.std(all30runs_MTGP_RL[8])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare NichGP1 with DRL5:')
    res = doWilcoxonTest(all30runs_MTGP_RL[0], all30runs_MTGP_RL[8], 0.05)
    if res == 0:
        print('NichGP1 = DRL5')
    elif res == 1:
        print('NichGP1 is significantly better than DRL5')
    elif res == 2:
        print('DRL5 is significantly better than NichGP1')
    else:
        print('There are something wrong here!')


def mean_of_top_n_NichingMTGP_all_RL_new(dataset_name):

    runs = 30

    all30runs_MTGP_RL = []
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])


    for i in range(runs):
        seed = i
        # address = sys.path[
        #               0] + '/experiment_result/scenario_' + dataset_name + '/intermediate_DRL_R_test_' + dataset_name + '_run_' + str(
        #     seed) + '_val.xlsx'
        address_RL1 = sys.path[0] + '/experiment_result/DRL/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL1):
            print("run " + str(i)+ "not exist!")
            continue
        address_RL2 = sys.path[
                          0] + '/experiment_result/GPDRL/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL2):
            print("run " + str(i) + "not exist!")
            continue
        address_RL3 = sys.path[
                          0] + '/experiment_result/NichGPDRL/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL3):
            print("run " + str(i) + "not exist!")
            continue
        address_RL4 = sys.path[
                          0] + '/experiment_result/NichGPDRL-experience/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL4):
            print("run " + str(i) + "not exist!")
            continue


        tardiness_eachRun_eachInstance_RL1 = pd.read_excel(address_RL1, sheet_name='sum', engine='openpyxl')
        sum_R1 = tardiness_eachRun_eachInstance_RL1.mean()
        tardiness_eachRun_eachInstance_RL2 = pd.read_excel(address_RL2, sheet_name='sum', engine='openpyxl')
        sum_R2 = tardiness_eachRun_eachInstance_RL2.mean()
        tardiness_eachRun_eachInstance_RL3 = pd.read_excel(address_RL3, sheet_name='sum', engine='openpyxl')
        sum_R3 = tardiness_eachRun_eachInstance_RL3.mean()
        tardiness_eachRun_eachInstance_RL4 = pd.read_excel(address_RL4, sheet_name='sum', engine='openpyxl')
        sum_R4 = tardiness_eachRun_eachInstance_RL4.mean()

        all30runs_MTGP_RL[0].append(sum_R3['Integrated_DRL'])
        all30runs_MTGP_RL[1].append(sum_R3['Integrated_DRL'])
        all30runs_MTGP_RL[2].append(sum_R3['Integrated_DRL'])
        all30runs_MTGP_RL[3].append(sum_R3['Integrated_DRL'])
        all30runs_MTGP_RL[4].append(sum_R1['Integrated_DRL'])
        all30runs_MTGP_RL[5].append(sum_R2['Integrated_DRL'])
        all30runs_MTGP_RL[6].append(sum_R3['Integrated_DRL'])
        all30runs_MTGP_RL[7].append(sum_R4['Integrated_DRL'])



    # sum_dict = {'GP0': all30runs_MTGP_RL[0], 'GP1': all30runs_MTGP_RL[1], 'GP2': all30runs_MTGP_RL[2],
    #             'GP3': all30runs_MTGP_RL[3], 'DRL1': all30runs_MTGP_RL[4], 'DRL2': all30runs_MTGP_RL[5]}
    sum_dict = {'GP0': all30runs_MTGP_RL[0],'GP1': all30runs_MTGP_RL[1],'GP2': all30runs_MTGP_RL[2],
                'GP3': all30runs_MTGP_RL[3],'DRL': all30runs_MTGP_RL[4],'GPDRL': all30runs_MTGP_RL[5],
                'NichGPDRL': all30runs_MTGP_RL[6],'NichGPDRLe': all30runs_MTGP_RL[7]}
    data = pd.DataFrame.from_dict(sum_dict)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_run_test_NichGP_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

    print('----NichGP1,  NichGP2, NichGP3,  NichGP4----')
    mean_NichGP1 = np.mean(all30runs_MTGP_RL[0])
    std_NichGP1 = np.std(all30runs_MTGP_RL[0])
    mean_NichGP2 = np.mean(all30runs_MTGP_RL[1])
    std_NichGP2 = np.std(all30runs_MTGP_RL[1])
    mean_NichGP3 = np.mean(all30runs_MTGP_RL[2])
    std_NichGP3 = np.std(all30runs_MTGP_RL[2])
    mean_NichGP4 = np.mean(all30runs_MTGP_RL[3])
    std_NichGP4 = np.std(all30runs_MTGP_RL[3])
    print(str(mean_NichGP1) + '(' + str(std_NichGP1) + '), ' +
          str(mean_NichGP2) + '(' + str(std_NichGP2) + '), ' +
          str(mean_NichGP3) + '(' + str(std_NichGP3) + '), ' +
          str(mean_NichGP4) + '(' + str(std_NichGP4) + ')'
          )

    print('----DRL,  GPDRL----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[5])
    std_DRL = np.std(all30runs_MTGP_RL[5])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')


    print('\nCompare DRL with GPDRL:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[5], 0.05)
    if res == 0:
        print('DRL = GPDRL')
    elif res == 1:
        print('DRL is significantly better than GPDRL')
    elif res == 2:
        print('GPDRL is significantly better than DRL')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL,  NichGPDRL----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[6])
    std_DRL = np.std(all30runs_MTGP_RL[6])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL with NichGPDRL:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[6], 0.05)
    if res == 0:
        print('DRL = NichGPDRL')
    elif res == 1:
        print('DRL is significantly better than NichGPDRL')
    elif res == 2:
        print('NichGPDRL is significantly better than DRL')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL,  NichGPDRLe----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL with NichGPDRLe:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('DRL = NichGPDRLe')
    elif res == 1:
        print('DRL is significantly better than NichGPDRLe')
    elif res == 2:
        print('NichGPDRLe is significantly better than DRL')
    else:
        print('There are something wrong here!')


    print('\n')
    print('----GPDRL,  NichGPDRL----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[5])
    std_MTGP = np.std(all30runs_MTGP_RL[5])
    mean_DRL = np.mean(all30runs_MTGP_RL[6])
    std_DRL = np.std(all30runs_MTGP_RL[6])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare GPDRL with NichGPDRL:')
    res = doWilcoxonTest(all30runs_MTGP_RL[5], all30runs_MTGP_RL[6], 0.05)
    if res == 0:
        print('GPDRL = NichGPDRL')
    elif res == 1:
        print('GPDRL is significantly better than NichGPDRL')
    elif res == 2:
        print('NichGPDRL is significantly better than GPDRL')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----GPDRL,  NichGPDRLe----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[5])
    std_MTGP = np.std(all30runs_MTGP_RL[5])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare GPDRL with NichGPDRLe:')
    res = doWilcoxonTest(all30runs_MTGP_RL[5], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('GPDRL = NichGPDRLe')
    elif res == 1:
        print('GPDRL is significantly better than NichGPDRLe')
    elif res == 2:
        print('NichGPDRLe is significantly better than GPDRL')
    else:
        print('There are something wrong here!')


    print('\n')
    print('----NichGPDRL,  NichGPDRLe----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[6])
    std_MTGP = np.std(all30runs_MTGP_RL[6])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare NichGPDRL with NichGPDRLe:')
    res = doWilcoxonTest(all30runs_MTGP_RL[6], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('NichGPDRL = NichGPDRLe')
    elif res == 1:
        print('NichGPDRL is significantly better than NichGPDRLe')
    elif res == 2:
        print('NichGPDRLe is significantly better than NichGPDRL')
    else:
        print('There are something wrong here!')


def mean_of_top_n_MTGP_all_RL(dataset_name):

    runs = 30

    all30runs_MTGP_RL = []
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])
    all30runs_MTGP_RL.append([])

    for i in range(runs):
        seed = i
        # address = sys.path[
        #               0] + '/experiment_result/scenario_' + dataset_name + '/intermediate_DRL_R_test_' + dataset_name + '_run_' + str(
        #     seed) + '_val.xlsx'
        address_RL1 = sys.path[0] + '/experiment_result/New1-OriginalDRL/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL1):
            print("run " + str(i)+ "not exist!")
            continue
        address_RL2 = sys.path[
                          0] + '/experiment_result/GPDRLnew/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL2):
            print("run " + str(i) + "not exist!")
            continue
        address_RL3 = sys.path[
                          0] + '/experiment_result/SequencingGP1/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL3):
            print("run " + str(i) + "not exist!")
            continue
        address_RL4 = sys.path[
                          0] + '/experiment_result/SequencingGP2/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL4):
            print("run " + str(i) + "not exist!")
            continue
        address_RL5 = sys.path[
                          0] + '/experiment_result/SequencingGP3/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL5):
            print("run " + str(i) + "not exist!")
            continue
        address_RL6 = sys.path[
                          0] + '/experiment_result/SequencingGP4/scenario_' + dataset_name + '/top_n_NichingMTGP_DRL_test_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        if not os.path.exists(address_RL6):
            print("run " + str(i) + "not exist!")
            continue

        tardiness_eachRun_eachInstance_RL1 = pd.read_excel(address_RL1, sheet_name='sum', engine='openpyxl')
        sum_R1 = tardiness_eachRun_eachInstance_RL1.mean()
        tardiness_eachRun_eachInstance_RL2 = pd.read_excel(address_RL2, sheet_name='sum', engine='openpyxl')
        sum_R2 = tardiness_eachRun_eachInstance_RL2.mean()
        tardiness_eachRun_eachInstance_RL3 = pd.read_excel(address_RL3, sheet_name='sum', engine='openpyxl')
        sum_R3 = tardiness_eachRun_eachInstance_RL3.mean()
        tardiness_eachRun_eachInstance_RL4 = pd.read_excel(address_RL4, sheet_name='sum', engine='openpyxl')
        sum_R4 = tardiness_eachRun_eachInstance_RL4.mean()
        tardiness_eachRun_eachInstance_RL5 = pd.read_excel(address_RL5, sheet_name='sum', engine='openpyxl')
        sum_R5 = tardiness_eachRun_eachInstance_RL5.mean()
        tardiness_eachRun_eachInstance_RL6 = pd.read_excel(address_RL6, sheet_name='sum', engine='openpyxl')
        sum_R6 = tardiness_eachRun_eachInstance_RL6.mean()

        all30runs_MTGP_RL[0].append(sum_R4['Nichtop_0'])
        all30runs_MTGP_RL[1].append(sum_R4['Nichtop_1'])
        all30runs_MTGP_RL[2].append(sum_R4['Nichtop_2'])
        all30runs_MTGP_RL[3].append(sum_R4['Nichtop_3'])
        all30runs_MTGP_RL[4].append(sum_R1['Integrated_DRL'])
        all30runs_MTGP_RL[5].append(sum_R2['Integrated_DRL'])
        all30runs_MTGP_RL[6].append(sum_R3['Integrated_DRL'])
        all30runs_MTGP_RL[7].append(sum_R4['Integrated_DRL'])
        all30runs_MTGP_RL[8].append(sum_R5['Integrated_DRL'])
        all30runs_MTGP_RL[9].append(sum_R6['Integrated_DRL'])


    # sum_dict = {'GP0': all30runs_MTGP_RL[0], 'GP1': all30runs_MTGP_RL[1], 'GP2': all30runs_MTGP_RL[2],
    #             'GP3': all30runs_MTGP_RL[3], 'DRL1': all30runs_MTGP_RL[4], 'DRL2': all30runs_MTGP_RL[5]}
    sum_dict = {'GP0': all30runs_MTGP_RL[0],'GP1': all30runs_MTGP_RL[1],'GP2': all30runs_MTGP_RL[2],
                'GP3': all30runs_MTGP_RL[3],'DRL1': all30runs_MTGP_RL[4],'DRL2': all30runs_MTGP_RL[5],
                'DGP1': all30runs_MTGP_RL[6],'DGP2': all30runs_MTGP_RL[7],'DGP3': all30runs_MTGP_RL[8],
                'DGP4': all30runs_MTGP_RL[9]}
    data = pd.DataFrame.from_dict(sum_dict)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_run_test_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

    print('----NichGP1,  NichGP2, NichGP3,  NichGP4----')
    mean_NichGP1 = np.mean(all30runs_MTGP_RL[0])
    std_NichGP1 = np.std(all30runs_MTGP_RL[0])
    mean_NichGP2 = np.mean(all30runs_MTGP_RL[1])
    std_NichGP2 = np.std(all30runs_MTGP_RL[1])
    mean_NichGP3 = np.mean(all30runs_MTGP_RL[2])
    std_NichGP3 = np.std(all30runs_MTGP_RL[2])
    mean_NichGP4 = np.mean(all30runs_MTGP_RL[3])
    std_NichGP4 = np.std(all30runs_MTGP_RL[3])
    print(str(mean_NichGP1) + '(' + str(std_NichGP1) + '), ' +
          str(mean_NichGP2) + '(' + str(std_NichGP2) + '), ' +
          str(mean_NichGP3) + '(' + str(std_NichGP3) + '), ' +
          str(mean_NichGP4) + '(' + str(std_NichGP4) + ')'
          )

    print('----DRL1,  DRL2----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[5])
    std_DRL = np.std(all30runs_MTGP_RL[5])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')


    print('\nCompare DRL1 with DRL2:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[5], 0.05)
    if res == 0:
        print('DRL1 = DRL2')
    elif res == 1:
        print('DRL1 is significantly better than DRL2')
    elif res == 2:
        print('DRL2 is significantly better than DRL1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL1,  DRL3----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[6])
    std_DRL = np.std(all30runs_MTGP_RL[6])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL1 with DRL3:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[6], 0.05)
    if res == 0:
        print('DRL1 = DRL3')
    elif res == 1:
        print('DRL1 is significantly better than DRL3')
    elif res == 2:
        print('DRL3 is significantly better than DRL1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL1,  DRL4----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL1 with DRL4:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('DRL1 = DRL4')
    elif res == 1:
        print('DRL1 is significantly better than DRL4')
    elif res == 2:
        print('DRL4 is significantly better than DRL1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL1,  DRL5----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[8])
    std_DRL = np.std(all30runs_MTGP_RL[8])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL1 with DRL5:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[8], 0.05)
    if res == 0:
        print('DRL1 = DRL5')
    elif res == 1:
        print('DRL1 is significantly better than DRL5')
    elif res == 2:
        print('DRL5 is significantly better than DRL1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL1,  DRL6----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[4])
    std_MTGP = np.std(all30runs_MTGP_RL[4])
    mean_DRL = np.mean(all30runs_MTGP_RL[9])
    std_DRL = np.std(all30runs_MTGP_RL[9])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL1 with DRL6:')
    res = doWilcoxonTest(all30runs_MTGP_RL[4], all30runs_MTGP_RL[9], 0.05)
    if res == 0:
        print('DRL1 = DRL6')
    elif res == 1:
        print('DRL1 is significantly better than DRL6')
    elif res == 2:
        print('DRL6 is significantly better than DRL1')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL2,  DRL3----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[5])
    std_MTGP = np.std(all30runs_MTGP_RL[5])
    mean_DRL = np.mean(all30runs_MTGP_RL[6])
    std_DRL = np.std(all30runs_MTGP_RL[6])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL2 with DRL3:')
    res = doWilcoxonTest(all30runs_MTGP_RL[5], all30runs_MTGP_RL[6], 0.05)
    if res == 0:
        print('DRL2 = DRL3')
    elif res == 1:
        print('DRL2 is significantly better than DRL3')
    elif res == 2:
        print('DRL3 is significantly better than DRL2')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL2,  DRL4----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[5])
    std_MTGP = np.std(all30runs_MTGP_RL[5])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL2 with DRL4:')
    res = doWilcoxonTest(all30runs_MTGP_RL[5], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('DRL2 = DRL4')
    elif res == 1:
        print('DRL2 is significantly better than DRL4')
    elif res == 2:
        print('DRL4 is significantly better than DRL2')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL2,  DRL5----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[5])
    std_MTGP = np.std(all30runs_MTGP_RL[5])
    mean_DRL = np.mean(all30runs_MTGP_RL[8])
    std_DRL = np.std(all30runs_MTGP_RL[8])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL2 with DRL5:')
    res = doWilcoxonTest(all30runs_MTGP_RL[5], all30runs_MTGP_RL[8], 0.05)
    if res == 0:
        print('DRL2 = DRL5')
    elif res == 1:
        print('DRL2 is significantly better than DRL5')
    elif res == 2:
        print('DRL5 is significantly better than DRL2')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL2,  DRL6----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[5])
    std_MTGP = np.std(all30runs_MTGP_RL[5])
    mean_DRL = np.mean(all30runs_MTGP_RL[9])
    std_DRL = np.std(all30runs_MTGP_RL[9])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL2 with DRL6:')
    res = doWilcoxonTest(all30runs_MTGP_RL[5], all30runs_MTGP_RL[9], 0.05)
    if res == 0:
        print('DRL2 = DRL6')
    elif res == 1:
        print('DRL2 is significantly better than DRL6')
    elif res == 2:
        print('DRL6 is significantly better than DRL2')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL3,  DRL4----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[6])
    std_MTGP = np.std(all30runs_MTGP_RL[6])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL3 with DRL4:')
    res = doWilcoxonTest(all30runs_MTGP_RL[6], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('DRL3 = DRL4')
    elif res == 1:
        print('DRL3 is significantly better than DRL4')
    elif res == 2:
        print('DRL4 is significantly better than DRL3')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL3,  DRL5----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[6])
    std_MTGP = np.std(all30runs_MTGP_RL[6])
    mean_DRL = np.mean(all30runs_MTGP_RL[8])
    std_DRL = np.std(all30runs_MTGP_RL[8])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL3 with DRL5:')
    res = doWilcoxonTest(all30runs_MTGP_RL[6], all30runs_MTGP_RL[8], 0.05)
    if res == 0:
        print('DRL3 = DRL5')
    elif res == 1:
        print('DRL3 is significantly better than DRL5')
    elif res == 2:
        print('DRL5 is significantly better than DRL3')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL3,  DRL6----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[6])
    std_MTGP = np.std(all30runs_MTGP_RL[6])
    mean_DRL = np.mean(all30runs_MTGP_RL[9])
    std_DRL = np.std(all30runs_MTGP_RL[9])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL3 with DRL6:')
    res = doWilcoxonTest(all30runs_MTGP_RL[6], all30runs_MTGP_RL[9], 0.05)
    if res == 0:
        print('DRL3 = DRL6')
    elif res == 1:
        print('DRL3 is significantly better than DRL6')
    elif res == 2:
        print('DRL6 is significantly better than DRL3')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL4,  DRL5----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[7])
    std_MTGP = np.std(all30runs_MTGP_RL[7])
    mean_DRL = np.mean(all30runs_MTGP_RL[8])
    std_DRL = np.std(all30runs_MTGP_RL[8])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL4 with DRL5:')
    res = doWilcoxonTest(all30runs_MTGP_RL[7], all30runs_MTGP_RL[8], 0.05)
    if res == 0:
        print('DRL4 = DRL5')
    elif res == 1:
        print('DRL4 is significantly better than DRL5')
    elif res == 2:
        print('DRL5 is significantly better than DRL4')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL4,  DRL6----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[7])
    std_MTGP = np.std(all30runs_MTGP_RL[7])
    mean_DRL = np.mean(all30runs_MTGP_RL[9])
    std_DRL = np.std(all30runs_MTGP_RL[9])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL4 with DRL6:')
    res = doWilcoxonTest(all30runs_MTGP_RL[7], all30runs_MTGP_RL[9], 0.05)
    if res == 0:
        print('DRL4 = DRL6')
    elif res == 1:
        print('DRL4 is significantly better than DRL6')
    elif res == 2:
        print('DRL6 is significantly better than DRL4')
    else:
        print('There are something wrong here!')

    print('\n')
    print('----DRL5,  DRL6----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[8])
    std_MTGP = np.std(all30runs_MTGP_RL[8])
    mean_DRL = np.mean(all30runs_MTGP_RL[9])
    std_DRL = np.std(all30runs_MTGP_RL[9])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare DRL5 with DRL6:')
    res = doWilcoxonTest(all30runs_MTGP_RL[8], all30runs_MTGP_RL[9], 0.05)
    if res == 0:
        print('DRL5 = DRL6')
    elif res == 1:
        print('DRL5 is significantly better than DRL6')
    elif res == 2:
        print('DRL6 is significantly better than DRL5')
    else:
        print('There are something wrong here!')

    print('--------------------------------------------------------------------------------')
    print('----NichGP1,  DRL4----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[0])
    std_MTGP = np.std(all30runs_MTGP_RL[0])
    mean_DRL = np.mean(all30runs_MTGP_RL[7])
    std_DRL = np.std(all30runs_MTGP_RL[7])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare NichGP1 with DRL4:')
    res = doWilcoxonTest(all30runs_MTGP_RL[0], all30runs_MTGP_RL[7], 0.05)
    if res == 0:
        print('NichGP1 = DRL4')
    elif res == 1:
        print('NichGP1 is significantly better than DRL4')
    elif res == 2:
        print('DRL4 is significantly better than NichGP1')
    else:
        print('There are something wrong here!')

    print('--------------------------------------------------------------------------------')
    print('----NichGP1,  DRL5----')
    mean_MTGP = np.mean(all30runs_MTGP_RL[0])
    std_MTGP = np.std(all30runs_MTGP_RL[0])
    mean_DRL = np.mean(all30runs_MTGP_RL[8])
    std_DRL = np.std(all30runs_MTGP_RL[8])
    print(str(mean_MTGP) + '(' + str(std_MTGP) + '), ' +
          str(mean_DRL) + '(' + str(std_DRL) + ')')

    print('\nCompare NichGP1 with DRL5:')
    res = doWilcoxonTest(all30runs_MTGP_RL[0], all30runs_MTGP_RL[8], 0.05)
    if res == 0:
        print('NichGP1 = DRL5')
    elif res == 1:
        print('NichGP1 is significantly better than DRL5')
    elif res == 2:
        print('DRL5 is significantly better than NichGP1')
    else:
        print('There are something wrong here!')


def mean_of_all_run_RLandGP_diff_training_instances(dataset_name):

    runs = 30

    all30runs_RL = []
    all30runs_RL.append([])
    all30runs_RL.append([])
    all30runs_RL.append([])

    all30runs_GP = []
    all30runs_GP.append([])
    all30runs_GP.append([])
    all30runs_GP.append([])

    instancesNum = ['instances50', 'instances100', 'instances200']

    sum_sum_sorted = []
    sum_sum0 = []
    sum_sum1 = []
    sum_sum2 = []

    for i in range(runs):
        seed = i
        address0 = sys.path[
                      0] + '/experiment_result/' + instancesNum[0] + '/scenario_' + dataset_name + '/GP_vs_RL_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        tardiness_eachRun_eachInstance0 = pd.read_excel(address0, sheet_name='sum', engine='openpyxl')
        sum0 = tardiness_eachRun_eachInstance0.mean()  # get mean on all the instances
        if(seed == 0):
            sum_sum0 = sum0
        else:
            sum_sum0 = sum_sum0 + sum0
        all30runs_RL[0].append(sum0['Integrated_DRL'])
        all30runs_GP[0].append(sum0['gen_best_MTGP_test'])

        address1 = sys.path[
                       0] + '/experiment_result/' + instancesNum[1] + '/scenario_' + dataset_name + '/GP_vs_RL_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        tardiness_eachRun_eachInstance1 = pd.read_excel(address1, sheet_name='sum', engine='openpyxl')
        sum1 = tardiness_eachRun_eachInstance1.mean()  # get mean on all the instances
        if (seed == 0):
            sum_sum1 = sum1
        else:
            sum_sum1 = sum_sum1 + sum1
        all30runs_RL[1].append(sum1['Integrated_DRL'])
        all30runs_GP[1].append(sum1['gen_best_MTGP_test'])

        address2 = sys.path[
                       0] + '/experiment_result/' + instancesNum[2] + '/scenario_' + dataset_name + '/GP_vs_RL_' + dataset_name + '_run_' + str(
            seed) + '_val.xlsx'
        tardiness_eachRun_eachInstance2 = pd.read_excel(address2, sheet_name='sum', engine='openpyxl')
        sum2 = tardiness_eachRun_eachInstance2.mean()  # get mean on all the instances
        if (seed == 0):
            sum_sum2 = sum2
        else:
            sum_sum2 = sum_sum2 + sum2
        all30runs_RL[2].append(sum2['Integrated_DRL'])
        all30runs_GP[2].append(sum2['gen_best_MTGP_test'])

            # all30runs_RL[0].append(sum['gen_best_MTGP_test'])
            # all30runs_RL[1].append(sum['Integrated_DRL'])

            # sum_sorted = sum.sort_values()
            # if (seed == 0):
            #     sum_sum_sorted = sum_sorted
            # else:
            #     sum_sum_sorted = sum_sum_sorted + sum_sorted

    sum_dict_RL = {'RL50ins': all30runs_RL[0], 'RL100ins': all30runs_RL[1], 'RL200ins': all30runs_RL[2]}
    data = pd.DataFrame.from_dict(sum_dict_RL)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_RL_diffIns_run_test_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

    print('----RL50ins,  RL100ins,  RL200ins----')
    mean_RL50ins = np.mean(all30runs_RL[0])
    std_RL50ins = np.std(all30runs_RL[0])
    mean_RL100ins = np.mean(all30runs_RL[1])
    std_RL100ins = np.std(all30runs_RL[1])
    mean_RL200ins = np.mean(all30runs_RL[2])
    std_RL200ins = np.std(all30runs_RL[2])
    print(str(mean_RL50ins) + '(' + str(std_RL50ins) + '), ' +
          str(mean_RL100ins) + '(' + str(std_RL100ins) + '), ' +
          str(mean_RL200ins) + '(' + str(std_RL200ins) + ')')


    print('\nCompare RLins50 with RLins100:')
    res = doWilcoxonTest(all30runs_RL[0], all30runs_RL[1], 0.05)
    if res == 0:
        print('RLins50 = RLins100')
    elif res == 1:
        print('RLins50 is significantly better than RLins100')
    elif res == 2:
        print('RLins100 is significantly better than RLins50')
    else:
        print('There are something wrong here!')

    print('\nCompare RLins50 with RLins200:')
    res = doWilcoxonTest(all30runs_RL[0], all30runs_RL[2], 0.05)
    if res == 0:
        print('RLins50 = RLins200')
    elif res == 1:
        print('RLins50 is significantly better than RLins200')
    elif res == 2:
        print('RLins200 is significantly better than RLins50')
    else:
        print('There are something wrong here!')

    print('\nCompare RLins100 with RLins200:')
    res = doWilcoxonTest(all30runs_RL[1], all30runs_RL[2], 0.05)
    if res == 0:
        print('RLins100 = RLins200')
    elif res == 1:
        print('RLins100 is significantly better than RLins200')
    elif res == 2:
        print('RLins200 is significantly better than RLins100')
    else:
        print('There are something wrong here!')

    sum_dict_GP = {'RL50ins': all30runs_GP[0], 'RL100ins': all30runs_GP[1], 'RL200ins': all30runs_GP[2]}
    data = pd.DataFrame.from_dict(sum_dict_GP)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/all_GP_diffIns_run_test_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

    print('----GP50ins,  GP100ins,  GP200ins----')
    mean_GP50ins = np.mean(all30runs_GP[0])
    std_GP50ins = np.std(all30runs_GP[0])
    mean_GP100ins = np.mean(all30runs_GP[1])
    std_GP100ins = np.std(all30runs_GP[1])
    mean_GP200ins = np.mean(all30runs_GP[2])
    std_GP200ins = np.std(all30runs_GP[2])
    print(str(mean_GP50ins) + '(' + str(std_GP50ins) + '), ' +
          str(mean_GP100ins) + '(' + str(std_GP100ins) + '), ' +
          str(mean_GP200ins) + '(' + str(std_GP200ins) + ')')


    print('\nCompare GPins50 with GPins100:')
    res = doWilcoxonTest(all30runs_GP[0], all30runs_GP[1], 0.05)
    if res == 0:
        print('GPins50 = GPins100')
    elif res == 1:
        print('GPins50 is significantly better than GPins100')
    elif res == 2:
        print('GPins100 is significantly better than GPins50')
    else:
        print('There are something wrong here!')

    print('\nCompare GPins50 with GPins200:')
    res = doWilcoxonTest(all30runs_GP[0], all30runs_GP[2], 0.05)
    if res == 0:
        print('GPins50 = GPins200')
    elif res == 1:
        print('GPins50 is significantly better than GPins200')
    elif res == 2:
        print('GPins200 is significantly better than GPins50')
    else:
        print('There are something wrong here!')

    print('\nCompare GPins100 with GPins200:')
    res = doWilcoxonTest(all30runs_GP[1], all30runs_GP[2], 0.05)
    if res == 0:
        print('GPins100 = GPins200')
    elif res == 1:
        print('GPins100 is significantly better than GPins200')
    elif res == 2:
        print('GPins200 is significantly better than GPins100')
    else:
        print('There are something wrong here!')


    # addressFinal = sys.path[0] + '/experiment_result/scenario_' + dataset_name + '/all_run_without_validation_MTGP_RL_' + dataset_name + '.xlsx'
    # after_sorted.to_excel(addressFinal, index=False)
    # after_sorted.to_excel(addressFinal, index = True, index_label=['Algo', 'tardiness'])
    # print(sum_sum_sorted)


if __name__ == '__main__':
    # dataset_name = str(sys.argv[1])
    # mean_of_all_run(dataset_name)

    all_dataset_name = ['HH', 'HL', 'LH', 'LL']
    # all_dataset_name = ['LH']
    # all_dataset_name = ['HL']
    for i in range(len(all_dataset_name)):
        print('\nResult on dataset: ' + all_dataset_name[i])
        dataset_name = all_dataset_name[i]
        # mean_of_all_run_all_gen_GP(dataset_name)
        # mean_of_all_run_MTGP_RL(dataset_name)
        mean_of_top_n_NichingMTGP_all_RL_new(dataset_name)
        # mean_of_top_n_MTGP_all_RL(dataset_name)
        # mean_of_top_n_NichingMTGP_RL(dataset_name)
        # mean_of_all_run_RLandGP_diff_training_instances(dataset_name)
        # mean_of_all_run_MTGP_GSGP(dataset_name)


