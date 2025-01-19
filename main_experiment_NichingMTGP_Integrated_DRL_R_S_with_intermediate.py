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
import NichingMTGP.LoadIndividual as nichmtload
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

        dict_top_inds_MTGP_individuals = nichmtload.load_top_inds_from_final_gen(kwargs['seedOfRun'], kwargs['dataset_name'])

        # specify the architecture of DRL
        if 'arch' and 'global_reward' in kwargs:
            arch = kwargs['arch'] + "=True"
            global_reward = 'global_reward={}'.format(kwargs['global_reward'])
            seedOfRun = 'seedOfRun={}'.format(kwargs['seedOfRun'])
            model_address = 'model_address={}'.format(kwargs['model_address'])
            GPrule_action = 'GPrule_action=True'
            GPrules = 'GPrules={}'.format(dict_top_inds_MTGP_individuals)
            validated = 'validated=1'
            # dataset_name = 'dataset_name={}'.format(kwargs['dataset_name']) #modified by mengxu
            if 'intermediate_address_R' in kwargs:
                intermediate_address_R = 'intermediate_address={}'.format(kwargs['intermediate_address_R'])
                order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list,{},{},{},{},{},{},{})".format(
                    arch, global_reward, seedOfRun, model_address, GPrule_action, GPrules, intermediate_address_R)
            else:
                order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list,{},{},{},{},{},{})".format(
                    arch, global_reward, seedOfRun, model_address, GPrule_action, GPrules)
            exec(order)

        # specify the architecture of DRL
        if 'DRL_S' in kwargs and kwargs['DRL_S']:
            print("---> DRL Sequencing mode ON <---")
            if 'intermediate_address_S' in kwargs:
                self.sequencing_brain = validation_S.DRL_sequencing(self.env, self.m_list, self.job_creator, \
                                                                    show=0, validated=1, reward_function='',
                                                                    seedOfRun=kwargs['seedOfRun'],
                                                                    model_address=kwargs['model_address'],
                                                                    GPrule_action=True,
                                                                    GPrules=dict_top_inds_MTGP_individuals,
                                                                    intermediate_address=kwargs['intermediate_address_S'])
            else:
                self.sequencing_brain = validation_S.DRL_sequencing(self.env, self.m_list, self.job_creator, \
                                                                    show=0, validated=1, reward_function='',
                                                                    seedOfRun=kwargs['seedOfRun'],
                                                                    model_address=kwargs['model_address'],
                                                                    GPrule_action=True,
                                                                    GPrules=dict_top_inds_MTGP_individuals)


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



def main(dataset_name, seedOfRun):
# if __name__ == "__main__":
#     dataSetName = str(sys.argv[1])
#     seedOfRun = int(sys.argv[2])

    # dictionary to store shopfloors and production record
    spf_dict = {}
    production_record = {}
    # list of experiments
    # benchmark = ['EA','CT','ET','TT','UT']
    # benchmark = ['EA','CT','ET','TT','UT','SQ','GP_pair_R']
    benchmark = []

    MTGP = []
    DRLs = ['validated']
    num_DRLs_R = 160
    DRLs_R_title = []
    reward_mechanism = [False]

    # title = benchmark + ['Integrated_DRL']
    span = 1000
    m_no = 6
    wc_no = 3
    sum_record = []
    benchmark_record = []
    max_record = []
    rate_record = []
    iteration = 10 #original 1
    # dont mess with above one-
    export_result = 1

    dataSetName = dataset_name
    seedOfRun = int(seedOfRun)

    #===================================The following is about not using validation===================================
    # # only load the best one
    # best_MTGP_rule_index = 50
    # # MTGP rule test, test the best rule obtained from all the generations
    # dict_best_MTGP_individuals = nichmtload.load_individual_from_gen(seedOfRun, dataSetName)

    # load the top n
    # best_MTGP_rule_index = 50
    # MTGP rule test, test the best rule obtained from all the generations
    # dict_best_MTGP_individuals = nichmtload.load_individual_from_gen(seedOfRun, dataSetName)
    dict_top_inds_MTGP_individuals = nichmtload.load_top_inds_from_final_gen(seedOfRun, dataSetName)


    testSeeds = 123453
    # I think this is wrong, actually I should use totally same randomseed for test of all the runs 2022.10.27
    np.random.seed(int(testSeeds))


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
            # np.random.seed(int(seed))  # add by mengxu 2022.10.31
            env = simpy.Environment()
            if rule == 'GP_pair_R':
                spf = shopfloor(env, span, m_no, wc_no, routing_rule=rule, sequencing_rule='GP_pair_S', seed=seed, seedofRun=seedOfRun, dataset_name=dataSetName)
            else:
                spf = shopfloor(env, span, m_no, wc_no, routing_rule=rule, seed=seed, seedofRun=seedOfRun, dataset_name=dataSetName)
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

        # niching GP test
        # for i in range(len(dict_top_inds_MTGP_individuals)):
        #     algo = 'Nichtop_' + str(i)
        #     if run == 0:
        #         MTGP.append(algo)
        #     individual = dict_top_inds_MTGP_individuals.get(i)
        #     sequencing_rule_tree = individual[0]
        #     routing_rule_tree = individual[1]
        #     # np.random.seed(int(seed))  # add by mengxu 2022.10.31
        #     env = simpy.Environment()
        #     spf = shopfloorMTGP(env, span, m_no, wc_no, sequencing_rule_tree, routing_rule_tree,
        #                         routing_rule='GP_pair_R_test', sequencing_rule='GP_pair_S_test', seed=seed, seedOfRun = seedOfRun, ifPrint=False,
        #                         dataset_name=dataSetName)
        #
        #     spf.simulation()
        #     output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        #     sum_record[run].append(cumulative_tard[-1])
        #     benchmark_record[run].append(cumulative_tard[-1])
        #     max_record[run].append(tard_max)
        #     rate_record[run].append(tard_rate)


        # RL test for R
        for idx,x in enumerate(DRLs):
            for intermediate_address_R in range(num_DRLs_R):
                env = simpy.Environment()
                if run == 0:
                    intermediate_title = 'iter_'+str(intermediate_address_R)
                    DRLs_R_title.append(intermediate_title)
                spf_R = shopfloor(env, span, m_no, wc_no, arch=x, global_reward=reward_mechanism[idx], seed=seed,
                                seedOfRun=seedOfRun, model_address=x, dataset_name=dataSetName, intermediate_address_R=intermediate_address_R)
                # spf = shopfloor(env, span, m_no, wc_no, DRL_R=True, DRL_S=True, seed=seed, dataset_name=dataSetName)# which one should I use
                # spf = shopfloor(env, span, m_no, wc_no, arch = x, global_reward = reward_mechanism[idx], DRL_S=True, seed=seed, seedOfRun = seedOfRun, model_address=x, dataset_name=dataSetName)
                # spf = shopfloor(env, span, m_no, wc_no, arch = x, global_reward = reward_mechanism[idx], seed = seed)
                spf_R.simulation()
                output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf_R.job_creator.tardiness_output()
                # add by mengxu to test
                # spf.job_creator.output()
                # spf.job_creator.final_output()
                # # add by mengxu to test
                sum_record[run].append(cumulative_tard[-1])
                max_record[run].append(tard_max)
                rate_record[run].append(tard_rate)

    title = benchmark + MTGP + DRLs_R_title
    # title = benchmark + MTGP + ['Integrated_DRL']
    # title = benchmark + MTGP

    print('-------------- Complete Record --------------')
    print(tabulate(sum_record, headers=title))
    print('-------------- Average Performance --------------')

    # get the performnce without DRL
    # avg_b = np.mean(benchmark_record,axis=0)
    # ratio_b = np.around(avg_b/avg_b.max()*100,2)
    winning_rate_b = np.zeros(len(title))
    # for idx in np.argmin(benchmark_record,axis=1):
    #     winning_rate_b[idx] += 1
    # winning_rate_b = np.around(winning_rate_b/iteration*100,2)

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
        address = sys.path[0] + '/experiment_result/scenario_' + dataSetName +'/intermediate_DRL_R_test_' + dataSetName + '_run_' + str(seedOfRun) + '_val.xlsx'
        # address = sys.path[0]+'/experiment_result/RAW_RA_val.xlsx'
        Excelwriter = pd.ExcelWriter(address,engine="xlsxwriter")
        dflist = [df_win_rate, df_sum, df_tardy_rate, df_max, df_before_win_rate]
        sheetname = ['win rate','sum', 'tardy rate', 'maximum','before win rate']

        for i,df in enumerate(dflist):
            df.to_excel(Excelwriter, sheet_name=sheetname[i], index=False)
        Excelwriter.save()
        print('export to {}'.format(address))
