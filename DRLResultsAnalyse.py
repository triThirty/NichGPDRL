import sys
from tabulate import tabulate
import pandas as pd
import numpy as np
import main_experiment_MTGP
from wilcoxonTest.wilcoxonTest import doWilcoxonTest
import os

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

def mean_of_intermediate_DRL_S_test(dataset_name,run):

    # runs = 1

    all30runs_RL = []
    num_intermediate = 180

    sum_sum_sorted = []
    sum_sum = []
    i = run
    # for i in range(runs):
    all30runs_RL.append([])
    seed = i
    address = sys.path[
                  0] + '/experiment_result/scenario_' + dataset_name + '/intermediate_DRL_S_test_' + dataset_name + '_run_' + str(
        seed) + '_val.xlsx'

    if not os.path.exists(address):
        print("run " + str(i) + "not exist!")
        return

    tardiness_eachRun_eachInstance = pd.read_excel(address, sheet_name='sum', engine='openpyxl')


    sum = tardiness_eachRun_eachInstance.mean()  # get mean on all the instances
    # if(seed == 0):
    #     sum_sum = sum
    # else:
    #     sum_sum = sum_sum + sum

    for idx in range(num_intermediate):
        name = 'iter_' + str(idx)
        all30runs_RL[0].append(sum[name])


    sum_dict = {'DRL_S_run_0': all30runs_RL[0]}
    data = pd.DataFrame.from_dict(sum_dict)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/run_' + str(
        seed) + '_intermediate_DRL_S_test_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)

def mean_of_intermediate_DRL_R_test(dataset_name,run):

    # runs = 1

    all30runs_RL = []
    num_intermediate = 320

    sum_sum_sorted = []
    sum_sum = []
    i = run
    # for i in range(runs):
    all30runs_RL.append([])
    seed = i
    address = sys.path[
                  0] + '/experiment_result/scenario_' + dataset_name + '/intermediate_DRL_R_test_' + dataset_name + '_run_' + str(
        seed) + '_val.xlsx'

    if not os.path.exists(address):
        print("run " + str(i) + "not exist!")
        return

    tardiness_eachRun_eachInstance = pd.read_excel(address, sheet_name='sum', engine='openpyxl')


    sum = tardiness_eachRun_eachInstance.mean()  # get mean on all the instances
    # if(seed == 0):
    #     sum_sum = sum
    # else:
    #     sum_sum = sum_sum + sum

    for idx in range(0,num_intermediate):
        name = 'iter_' + str(idx)
        all30runs_RL[0].append(sum[name])


    sum_dict = {'DRL_R_run_0': all30runs_RL[0]}
    data = pd.DataFrame.from_dict(sum_dict)
    # print(data)
    addressFinal = sys.path[
                       0] + '/experiment_result/scenario_' + dataset_name + '/run_' + str(
        seed) + '_intermediate_DRL_R_test_results_' + dataset_name + '.xlsx'
    data.to_excel(addressFinal, index=False)



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

    runs = 13
    # all_dataset_name = ['HH']
    all_dataset_name = ['HH','HL','LH']
    for run in range(12,runs):
        for i in range(len(all_dataset_name)):
            print('\nResult on dataset: ' + all_dataset_name[i])
            dataset_name = all_dataset_name[i]
            # mean_of_all_run_all_gen_GP(dataset_name)
            # mean_of_all_run_MTGP_RL(dataset_name)
            mean_of_intermediate_DRL_S_test(dataset_name, run)
            # mean_of_intermediate_DRL_R_test(dataset_name, run)
        # mean_of_all_run_RLandGP_diff_training_instances(dataset_name)
        # mean_of_all_run_MTGP_GSGP(dataset_name)


