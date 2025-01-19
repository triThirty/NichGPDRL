import simpy
import sys
sys.path 
# import matplotlib.pyplot as plt
# import matplotlib.animation as animation
# import torch
import numpy as np
from tabulate import tabulate
import pandas as pd
from pandas import DataFrame
import MTGP.LoadIndividual as mtload
import agent_machine
import agent_workcenter
import sequencing
import routing
import job_creation
# import breakdown_creation
# import heterogeneity_creation
import validation_S
import validation_R

'''
experiment of independent routing agents
'''

class shopfloor:
    def __init__(self, env, span, m_no, wc_no, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        m_per_wc = int(self.m_no / self.wc_no)
        '''STEP 2.1: create instances of machines'''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)
        #print(self.m_list)
        '''STEP 2.2: create instances of work centers'''
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            #print(x)
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) # create work centers
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) # add to machine list
            exec(expr2)
            cum_m_idx += m_per_wc
        #print(self.wc_list)

        '''STEP 3: initialize the job creator'''
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz
        if 'seed' in kwargs:
            # self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
            #     [5,25], 2, 0.9, seed=kwargs['seed'])
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [5, 25], 2, 0.9, seed=kwargs['seed'])
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [5, 25], 3, 0.9, seed=kwargs['seed'])
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 2, 0.9, seed=kwargs['seed'])
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 3, 0.9, seed=kwargs['seed'])
            #self.job_creator.output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])


        '''STEP 5: set sequencing or routing rules, and DRL'''
        # check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    print("Rule assigned to machine {} is invalid !".format(m.m_idx))
                    raise Exception

        # check if need to reset routing rule
        if 'routing_rule' in kwargs:
            print("Taking over: workcenters use {} routing rule".format(kwargs['routing_rule']))
            for wc in self.wc_list:
                order = "wc.job_routing = routing." + kwargs['routing_rule']
                try:
                    exec(order)
                except:
                    print("Rule assigned to workcenter {} is invalid !".format(wc.wc_idx))
                    raise Exception

        # specify the architecture of DRL
        if 'arch' and 'global_reward' in kwargs:
            arch = kwargs['arch'] + "=True"
            global_reward = 'global_reward={}'.format(kwargs['global_reward'])
            order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, {},{})".format(arch,global_reward)
            exec(order)

        # specify the architecture of DRL
        if 'DRL_S' in kwargs and kwargs['DRL_S']:
            print("---> DRL Sequencing mode ON <---")
            self.sequencing_brain = validation_S.DRL_sequencing(self.env, self.m_list, self.job_creator, \
                                                                    show=0, validated=1, reward_function='')

    def simulation(self):
        self.env.run()

class shopfloorMTGP:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, routing_tree, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        self.ifPrint = kwargs['ifPrint'] # added by mengxu
        # self.sequencingTree = sequencing_tree
        # self.routingTree = routing_tree
        # self.routingRule = kwargs['tree_routing']
        m_per_wc = int(self.m_no / self.wc_no)
        '''STEP 2.1: create instances of machines'''
        for i in range(m_no):
            expr1 = '''self.m_{} = agent_machine.machine(env, {}, print = 0)'''.format(i,i) # create machines
            exec(expr1)
            expr2 = '''self.m_list.append(self.m_{})'''.format(i) # add to machine list
            exec(expr2)
        #print(self.m_list)
        '''STEP 2.2: create instances of work centers'''
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            #print(x)
            expr1 = '''self.wc_{} = agent_workcenter.workcenter(env, {}, x)'''.format(i,i) # create work centers
            exec(expr1)
            expr2 = '''self.wc_list.append(self.wc_{})'''.format(i) # add to machine list
            exec(expr2)
            cum_m_idx += m_per_wc
        #print(self.wc_list)

        '''STEP 3: initialize the job creator'''
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz
        if 'seed' in kwargs:
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [5, 25], 2, 0.9, seed=kwargs['seed'])
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [5, 25], 3, 0.9, seed=kwargs['seed'])
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 2, 0.9, seed=kwargs['seed'])
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 3, 0.9, seed=kwargs['seed'])
            #self.job_creator.output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
            wc.setJobRoutingTree(routing_tree)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])
            m.setJobSequencingTree(sequencing_tree)


        '''STEP 5: set sequencing or routing rules, and DRL'''
        # check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            if self.ifPrint:
                print("Taking over: machines use {} sequencing rule".format(kwargs['sequencing_rule']))
            for m in self.m_list:
                order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                # order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print("Rule assigned to machine {} is invalid !".format(m.m_idx))
                    raise Exception

        # check if need to reset routing rule
        if 'routing_rule' in kwargs:
            if self.ifPrint:
                print("Taking over: workcenters use {} routing rule".format(kwargs['routing_rule']))
            for wc in self.wc_list:
                order = "wc.job_routing = routing." + kwargs['routing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print("Rule assigned to workcenter {} is invalid !".format(wc.wc_idx))
                    raise Exception

        # specify the architecture of DRL
        if 'arch' and 'global_reward' in kwargs:
            arch = kwargs['arch'] + "=True"
            global_reward = 'global_reward={}'.format(kwargs['global_reward'])
            order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, {},{})".format(arch,global_reward)
            exec(order)

    def simulation(self):
        self.env.run()


def best_rule_after_validation(dataSetName,randomSeeds):
    print('******************* Validation *******************')
    sum_record_validation = []
    iteration_validation = 100
    MTGP_validation = []
    best_individual_after_validation = []

    dict_best_individuals = mtload.load_individual_from_gen(randomSeeds, dataSetName)
    training_time = mtload.load_training_time(randomSeeds, dataSetName)
    min_fitness = mtload.load_min_fitness(randomSeeds, dataSetName)
    print('\nTraining time: ')
    print(training_time)
    print('Training min_fitness: ')
    print(min_fitness)
    print('\n')

    for run in range(iteration_validation):
        seed = np.random.randint(2000000000)
        sum_record_validation.append([])
        print('******************* ITERATION-{} *******************'.format(run))
        for idx in range(len(dict_best_individuals)):
            if run == 0:
                MTGP_validation.append(idx)
            individual = dict_best_individuals.get(idx)
            sequencing_rule_tree = individual[0]
            routing_rule_tree = individual[1]
            # print('\nsequencing_rule: ')
            # print(sequencing_rule_tree)
            # print('routing_rule: ')
            # print(routing_rule_tree)
            # create the environment instance for simulation
            env = simpy.Environment()
            spf = shopfloorMTGP(env, span, m_no, wc_no, sequencing_rule_tree, routing_rule_tree,
                                routing_rule='GP_pair_R_test', sequencing_rule='GP_pair_S_test', seed=seed, ifPrint=False,
                                dataset_name=dataSetName)

            # spf = shopfloor(env, span, m_no, wc_no, routing_rule = rule, seed = seed)
            spf.simulation()
            output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
            # add by mengxu to test
            # spf.job_creator.output()
            # spf.job_creator.final_output()
            # # add by mengxu to test
            sum_record_validation[run].append(cumulative_tard[-1])
            # max_record[run].append(tard_max)
            # rate_record[run].append(tard_rate)

    # get the overall performance
    avg = np.mean(sum_record_validation, axis=0)

    best_index = avg.argmin(axis=0)
    print('Best rule index: ' + str(best_index))
    return best_index

        # return dict_best_individuals.get(best_index)


        # winning_rate = np.zeros(len(title))
        # for idx in np.argmin(sum_record, axis=1):
        #     winning_rate[idx] += 1
        # winning_rate = np.around(winning_rate / iteration * 100, 2)
        # for rank, rule in enumerate(rank):
        #     print("{}, avg.: {} | max: {} | %: {}% | tardy %: {}% | winning rate: {}/{}%" \
        #           .format(title[rule], avg[rule], max[rule], ratio[rule], tardy_rate[rule], winning_rate_b[rule],
        #                   winning_rate[rule]))


# dictionary to store shopfloors and production record
spf_dict = {}
production_record = {}
# list of experiments
# benchmark = ['EA','CT','ET','TT','UT']
# benchmark = ['EA','CT','ET','TT','UT','SQ','GP_pair_R']
benchmark = []

MTGP = []
DRLs = ['validated']
reward_mechanism = [False]

# title = benchmark + ['Integrated_DRL']
span = 1000
m_no = 6
wc_no = 3
sum_record = []
benchmark_record = []
max_record = []
rate_record = []
iteration = 100 #original 1
# dont mess with above one-
export_result = 1

def main(dataset_name, seed):
# if __name__ == "__main__":
    dataSetName = str(sys.argv[1])
    randomSeeds = int(sys.argv[2])

    # Do validation and obtain the best evolved rule
    best_rule_index = best_rule_after_validation(dataSetName, randomSeeds)
    # MTGP rule test, test the best rule obtained from all the generations
    dict_best_individuals = mtload.load_individual_from_gen(randomSeeds, dataSetName)
    # training_time = mtload.load_training_time(randomSeeds, dataSetName)
    # min_fitness = mtload.load_min_fitness(randomSeeds, dataSetName)
    # print('\nTraining time: ')
    # print(training_time)
    # print('Training min_fitness: ')
    # print(min_fitness)
    # print('\n')


    for run in range(iteration):
        print('******************* ITERATION-{} *******************'.format(run))
        sum_record.append([])
        benchmark_record.append([])
        max_record.append([])
        rate_record.append([])
        seed = np.random.randint(2000000000)
        # run simulation with different rules
        for idx,rule in enumerate(benchmark):
            # create the environment instance for simulation
            env = simpy.Environment()
            if rule == 'GP_pair_R':
                spf = shopfloor(env, span, m_no, wc_no, routing_rule=rule, sequencing_rule='GP_pair_S', seed=seed, dataset_name=dataSetName)
            else:
                spf = shopfloor(env, span, m_no, wc_no, routing_rule=rule, seed=seed, dataset_name=dataSetName)
            # spf = shopfloor(env, span, m_no, wc_no, routing_rule = rule, seed = seed)
            spf.simulation()
            output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
            # add by mengxu to test
            # spf.job_creator.output()
            # spf.job_creator.final_output()
            # # add by mengxu to test
            sum_record[run].append(cumulative_tard[-1])
            benchmark_record[run].append(cumulative_tard[-1])
            max_record[run].append(tard_max)
            rate_record[run].append(tard_rate)

    # for run in range(iteration):
    #     for idx in range(len(dict_best_individuals)):
        algo = 'gen_best_MTGP_test'
    #     algo = 'gen_' + str(best_rule_index) + '_MTGP_test'
        if run == 0:
            MTGP.append(algo)
        individual = dict_best_individuals.get(best_rule_index)
        sequencing_rule_tree = individual[0]
        routing_rule_tree = individual[1]
        # print('\nsequencing_rule: ')
        # print(sequencing_rule_tree)
        # print('routing_rule: ')
        # print(routing_rule_tree)
        # create the environment instance for simulation
        env = simpy.Environment()
        spf = shopfloorMTGP(env, span, m_no, wc_no, sequencing_rule_tree, routing_rule_tree, routing_rule='GP_pair_R_test', sequencing_rule='GP_pair_S_test', seed=seed, ifPrint=False, dataset_name=dataSetName)

        # spf = shopfloor(env, span, m_no, wc_no, routing_rule = rule, seed = seed)
        spf.simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        # add by mengxu to test
        # spf.job_creator.output()
        # spf.job_creator.final_output()
        # # add by mengxu to test
        sum_record[run].append(cumulative_tard[-1])
        benchmark_record[run].append(cumulative_tard[-1])
        max_record[run].append(tard_max)
        rate_record[run].append(tard_rate)

        # and extra run with DRL
        # env = simpy.Environment()
        # spf = shopfloor(env, span, m_no, wc_no, DRL_R=True, DRL_S=True, seed=seed)
        # # spf = shopfloor(env, span, m_no, wc_no, arch = x, global_reward = reward_mechanism[idx], DRL_S=True, seed=seed)
        # # spf = shopfloor(env, span, m_no, wc_no, arch = x, global_reward = reward_mechanism[idx], seed = seed)
        # spf.simulation()
        # output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        # sum_record[run].append(cumulative_tard[-1])
        # max_record[run].append(tard_max)
        # rate_record[run].append(tard_rate)

        for idx,x in enumerate(DRLs):
            env = simpy.Environment()
            # spf = shopfloor(env, span, m_no, wc_no, DRL_R=True, DRL_S=True, seed=seed)
            spf = shopfloor(env, span, m_no, wc_no, arch = x, global_reward = reward_mechanism[idx], DRL_S=True, seed=seed, dataset_name=dataSetName)
            # spf = shopfloor(env, span, m_no, wc_no, arch = x, global_reward = reward_mechanism[idx], seed = seed)
            spf.simulation()
            output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
            # add by mengxu to test
            # spf.job_creator.output()
            # spf.job_creator.final_output()
            # # add by mengxu to test
            sum_record[run].append(cumulative_tard[-1])
            max_record[run].append(tard_max)
            rate_record[run].append(tard_rate)

    title = benchmark + MTGP + ['Integrated_DRL']

    print('-------------- Complete Record --------------')
    print(tabulate(sum_record, headers=title))
    print('-------------- Average Performance --------------')

    # get the performnce without DRL
    avg_b = np.mean(benchmark_record,axis=0)
    ratio_b = np.around(avg_b/avg_b.max()*100,2)
    winning_rate_b = np.zeros(len(title))
    for idx in np.argmin(benchmark_record,axis=1):
        winning_rate_b[idx] += 1
    winning_rate_b = np.around(winning_rate_b/iteration*100,2)

    # get the overall performance (include DRL)
    avg = np.mean(sum_record,axis=0)
    max = np.mean(max_record,axis=0)
    tardy_rate = np.around(np.mean(rate_record,axis=0)*100,2)
    ratio = np.around(avg/avg.min()*100,2)
    rank = np.argsort(ratio)
    winning_rate = np.zeros(len(title))
    for idx in np.argmin(sum_record,axis=1):
        winning_rate[idx] += 1
    winning_rate = np.around(winning_rate/iteration*100,2)
    for rank,rule in enumerate(rank):
        print("{}, avg.: {} | max: {} | %: {}% | tardy %: {}% | winning rate: {}/{}%"\
        .format(title[rule],avg[rule],max[rule],ratio[rule],tardy_rate[rule],winning_rate_b[rule],winning_rate[rule]))

    if export_result:
        df_win_rate = DataFrame([winning_rate], columns=title)
        #print(df_win_rate)
        df_sum = DataFrame(sum_record, columns=title)
        #print(df_sum)
        df_tardy_rate = DataFrame(rate_record, columns=title)
        #print(df_tardy_rate)
        df_max = DataFrame(max_record, columns=title)
        #print(df_max)
        df_before_win_rate = DataFrame([winning_rate_b], columns=title)
        address = sys.path[0] + '/experiment_result/scenario_' + dataSetName +'/MTGP_vs_RL_' + dataSetName + '_run_' + str(randomSeeds) + '_val.xlsx'
        # address = sys.path[0]+'/experiment_result/RAW_RA_val.xlsx'
        Excelwriter = pd.ExcelWriter(address,engine="xlsxwriter")
        dflist = [df_win_rate, df_sum, df_tardy_rate, df_max, df_before_win_rate]
        sheetname = ['win rate','sum', 'tardy rate', 'maximum','before win rate']

        for i,df in enumerate(dflist):
            df.to_excel(Excelwriter, sheet_name=sheetname[i], index=False)
        Excelwriter.save()
        print('export to {}'.format(address))

    # check the parameter and scenario setting
    spf.sequencing_brain.check_parameter()
    spf.routing_brain.check_parameter()
