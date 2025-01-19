
import simpy
import sys
sys.path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from tabulate import tabulate
import NichingMTGP.LoadIndividual as nichmtload
import agent_machine
import agent_workcenter
import brain_workcenter_R
import brain_machine_S
import job_creation
import time
import sequencing
import routing
import breakdown_creation
import heterogeneity_creation
import validation_S # for co-training
import NichingMTGP.LoadIndividual as mtload

"""
THIS IS THE MODULE FOR ROUTING AGENT TRAINING
"""

class shopfloor:
    def __init__(self, env, span, m_no, wc_no, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        self.ifPrint = kwargs['ifPrint']  # added by mengxu
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
        # add by mengxu 2022.12.15
        if 'dataset_name' in kwargs:
            if kwargs['dataset_name'] == 'HH':
                self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                         [5, 25], 2, 0.99, random_seed=True,
                                                         ifPrint=self.ifPrint)  # ifPrint = self.ifPrint is added by mengxu to make it clearer when using MTGP to train
            elif kwargs['dataset_name'] == 'HL':
                self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                         [5, 25], 3, 0.99, random_seed=True,
                                                         ifPrint=self.ifPrint)
            elif kwargs['dataset_name'] == 'LH':
                self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                         [10, 20], 2, 0.99, random_seed=True,
                                                         ifPrint=self.ifPrint)
            elif kwargs['dataset_name'] == 'LL':
                self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                         [10, 20], 3, 0.99, random_seed=True,
                                                         ifPrint=self.ifPrint)

        '''STEP 4: initialize machines and work centers'''
        dict_top_inds_MTGP_individuals = mtload.load_top_inds_from_final_gen(kwargs['seed'], kwargs['dataset_name'])
        # dict_top_inds_MTGP_individuals_HH = mtload.load_top_inds_from_final_gen(kwargs['seed'], 'HH')
        # dict_top_inds_MTGP_individuals_HL = mtload.load_top_inds_from_final_gen(kwargs['seed'], 'HL')
        # dict_top_inds_MTGP_individuals_LH = mtload.load_top_inds_from_final_gen(kwargs['seed'], 'LH')
        # dict_top_inds_MTGP_individuals_LL = mtload.load_top_inds_from_final_gen(kwargs['seed'], 'LL')
        # dict_top_inds_MTGP_individuals = []
        # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_HH.get(0))
        # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_HL.get(0))
        # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_LH.get(0))
        # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_LL.get(0))
        if 'routing_rule' in kwargs:
            # if kwargs['dataset_name'] == 'HH':
            #     individual = dict_top_inds_MTGP_individuals[0]
            # elif kwargs['dataset_name'] == 'HL':
            #     individual = dict_top_inds_MTGP_individuals[1]
            # elif kwargs['dataset_name'] == 'LH':
            #     individual = dict_top_inds_MTGP_individuals[2]
            # elif kwargs['dataset_name'] == 'LL':
            #     individual = dict_top_inds_MTGP_individuals[3]
            individual = dict_top_inds_MTGP_individuals.get(0)
            routing_rule_tree = individual[1]
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
            if 'routing_rule' in kwargs:
                wc.setJobRoutingTree(routing_rule_tree)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])

        '''STEP 5-2: set up the brains for machines'''
        # dict_best_MTGP_individuals = mtload.load_individual_from_gen(kwargs['seed'], kwargs['dataset_name'])
        self.sqc_brain = brain_machine_S.sequencing_brain(self.env, self.job_creator, self.m_list, self.m_list,
                                                          self.span / 10, self.span, MC=1, reward_function=1, IQL=0,
                                                          I_DDQN=0, seed=kwargs['seed'],
                                                          dataset_name=kwargs['dataset_name'],
                                                          GPrule_action=True,
                                                          GPrules=dict_top_inds_MTGP_individuals)  # modified by mengxu 2022.10.10

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

        '''STEP 6: run the simulaiton'''
        start = time.time()
        env.run()
        # self.routing_brain.check_parameter()
        self.sqc_brain.check_parameter()
        end = time.time()
        running_time = end - start
        print('main_training_R running time: ' + str(running_time))
        # saveRunningTime(kwargs['seed'], kwargs['dataset_name'], running_time, len(self.wc_list), len(self.m_list), type='routing_models')
        saveRunningTime(kwargs['seed'], kwargs['dataset_name'], running_time, len(self.wc_list), len(self.m_list), type='sequencing_models')
        # hide by mengxu 2023.08.19
        # self.routing_brain.loss_record_output(save = 1, seed=kwargs['seed'])
        self.sqc_brain.loss_record_output(save=1, seed=kwargs['seed'], dataset_name=kwargs['dataset_name'])
        self.sqc_brain.culmulative_reward_record_output(save=1, seed=kwargs['seed'], dataset_name=kwargs['dataset_name'])
        # self.sqc_brain.reward_record_output(save=1, seed=kwargs['seed'], dataset_name=kwargs['dataset_name'])

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
                                                             [5, 25], 2, 0.99, seed=kwargs['seed'])
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [5, 25], 3, 0.99, seed=kwargs['seed'])
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 2, 0.99, seed=kwargs['seed'])
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 3, 0.99, seed=kwargs['seed'])
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


def saveRunningTime(randomSeeds, dataSetName, running_time, wc_num, m_num, **kwargs):
    address_seed = "./" + kwargs['type'] + "/scenario_" + dataSetName + "/running_time_" + str(
        randomSeeds) + "_small_state_dict" + '{}wc{}m'.format(wc_num, m_num)  # modified by mengxu 2022.10.31
    # address_seed = "./routing_models/scenario_" + dataSetName + "/running_time_" + str(randomSeeds) + "_small_state_dict" + '{}wc{}m'.format(wc_num, m_num)  # modified by mengxu 2022.10.31
    # # fileName1= './MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds)+'_running_time' + dataSetName
    np.save(address_seed, running_time)
    return


def training(dataset_name, seed):
    np.random.seed(int(seed))
    # create the environment instance for simulation
    env = simpy.Environment()
    # create the shop floor instance
    span = 5000
    m_no = 9
    wc_no = 3
    record_tardiness = []
    spf = shopfloor(env, span, m_no, wc_no, routing_rule='GP_pair_R_test', seed=seed, dataset_name=dataset_name,
                    ifPrint=False)
    output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
    record_tardiness.append(cumulative_tard[-1])


    dict_top_inds_MTGP_individuals = nichmtload.load_top_inds_from_final_gen(seed, dataset_name)
    MTGP = []
    # niching GP test
    for i in range(4):
        # for i in range(len(dict_top_inds_MTGP_individuals)):
        algo = 'Nichtop_' + str(i)
        MTGP.append(algo)
        individual = dict_top_inds_MTGP_individuals.get(i)
        sequencing_rule_tree = individual[0]
        routing_rule_tree = individual[1]
        np.random.seed(int(seed))  # add by mengxu 2022.10.31
        env = simpy.Environment()
        spf = shopfloorMTGP(env, span, m_no, wc_no, sequencing_rule_tree, routing_rule_tree,
                            routing_rule='GP_pair_R_test', sequencing_rule='GP_pair_S_test', seed=seed,
                            seedOfRun=seed, ifPrint=False,
                            dataset_name=dataset_name)

        spf.simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
        record_tardiness.append(cumulative_tard[-1])

    print("Tardiness: " + str(record_tardiness))
    # the following is to use the manual routing rule for training the DRL sequencing rule
    # spf = shopfloor(env, span, m_no, wc_no, seed=seed, dataset_name=dataset_name,ifPrint=False)
    # the following is to use the GP evolved best routing rule for training the DRL sequencing rule

