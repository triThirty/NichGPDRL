
import simpy
import sys
sys.path 
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from tabulate import tabulate

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
        # dict_top_inds_MTGP_individuals = mtload.load_top_inds_from_final_gen(kwargs['seed'], kwargs['dataset_name'])
        if 'sequencing_rule' in kwargs:
            # if kwargs['dataset_name'] == 'HH':
            #     individual = dict_top_inds_MTGP_individuals[0]
            # elif kwargs['dataset_name'] == 'HL':
            #     individual = dict_top_inds_MTGP_individuals[1]
            # elif kwargs['dataset_name'] == 'LH':
            #     individual = dict_top_inds_MTGP_individuals[2]
            # elif kwargs['dataset_name'] == 'LL':
            #     individual = dict_top_inds_MTGP_individuals[3]
            individual = dict_top_inds_MTGP_individuals.get(0)
            sequencing_rule_tree = individual[0]
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])
            if 'sequencing_rule' in kwargs:
                m.setJobSequencingTree(sequencing_rule_tree)

        '''STEP 5-1: set up the brains for workcenters'''
        # MTGP rule test, test the best rule obtained from all the generations
        self.routing_brain = brain_workcenter_R.routing_brain(self.env, self.job_creator, self.m_list, self.wc_list,
                                                              self.span / 5, self.span, seed=kwargs['seed'],
                                                              dataset_name=kwargs['dataset_name'],
                                                              GPrule_action=True,
                                                              Single_agent=True,
                                                              GPrules=dict_top_inds_MTGP_individuals)
        # self.routing_brain = brain_workcenter_R.routing_brain(self.env, self.job_creator, self.m_list, self.wc_list,
        #                                                       self.span/5, self.span, seed = kwargs['seed'],
        #                                                       dataset_name=kwargs['dataset_name'],
        #                                                       GPrule_action=True,
        #                                                       GPrules=dict_best_MTGP_individuals)

        '''STEP 5-2: set sequencing or routing rules, and DRL'''
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


        '''STEP 6: run the simulaiton'''
        start = time.time()
        env.run()
        self.routing_brain.check_parameter()
        # self.sqc_brain.check_parameter()
        end = time.time()
        running_time = end - start
        print('main_training_R running time: ' + str(running_time))
        saveRunningTime(kwargs['seed'], kwargs['dataset_name'], running_time, len(self.wc_list), len(self.m_list), type='routing_models')
        # saveRunningTime(kwargs['seed'], kwargs['dataset_name'], running_time, len(self.wc_list), len(self.m_list), type='sequencing_models')
        # hide by mengxu 2023.08.19
        self.routing_brain.loss_record_output(save = 1, seed=kwargs['seed'], dataset_name=kwargs['dataset_name'])
        # self.routing_brain.reward_record_output(save = 1, seed=kwargs['seed'], dataset_name=kwargs['dataset_name'])
        self.routing_brain.culmulative_reward_record_output(save=1, seed=kwargs['seed'], dataset_name=kwargs['dataset_name'])
        # self.sqc_brain.loss_record_output(save=1, seed=kwargs['seed'])


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
    span = 100000
    m_no = 9
    wc_no = 3
    # the following is to use the manual routing rule for training the DRL sequencing rule
    # spf = shopfloor(env, span, m_no, wc_no, seed=seed, dataset_name=dataset_name, ifPrint=False)
    # the following is to use the GP evolved routing rule for training the DRL sequencing rule
    spf = shopfloor(env, span, m_no, wc_no, seed=seed, dataset_name=dataset_name, ifPrint=False)
