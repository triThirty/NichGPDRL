import random

import simpy
import sys
sys.path #sometimes need this to refresh the path
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import torch
import numpy as np
from tabulate import tabulate

import agent_machine
import agent_workcenter
import brain_machine_S
import job_creation
import time
import breakdown_creation
import heterogeneity_creation

"""
THIS IS THE MODULE FOR TRAINING
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
        # self.job_creator = job_creation.creation\
        # (self.env, self.span, self.m_list, self.wc_list, [5,25], 3, 0.9, random_seed = True)

        if 'dataset_name' in kwargs:
            if kwargs['dataset_name'] == 'HH':
                self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                         [5, 25], 2, 0.9, random_seed=True,
                                                         ifPrint=self.ifPrint)  # ifPrint = self.ifPrint is added by mengxu to make it clearer when using MTGP to train
            elif kwargs['dataset_name'] == 'HL':
                self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                         [5, 25], 3, 0.9, random_seed=True,
                                                         ifPrint=self.ifPrint)
            elif kwargs['dataset_name'] == 'LH':
                self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                         [10, 20], 2, 0.9, random_seed=True,
                                                         ifPrint=self.ifPrint)
            elif kwargs['dataset_name'] == 'LL':
                self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                         [10, 20], 3, 0.9, random_seed=True,
                                                         ifPrint=self.ifPrint)

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])

        '''STEP 5: set up the brains for machines'''
        self.sqc_brain = brain_machine_S.sequencing_brain(self.env, self.job_creator, self.m_list, self.m_list, self.span/10, self.span, MC = 1,  reward_function = 1, IQL = 0,
                                                          I_DDQN = 0, seed=kwargs['seed'], dataset_name=kwargs['dataset_name'],
                                                          use_build_experience_strategy = False) #modified by mengxu 2022.10.10
        # self.sqc_brain = brain_machine_S.sequencing_brain(self.env, self.job_creator, self.m_list, self.m_list,
        #                                                   self.span / 10, self.span, MC=1, reward_function=1) #original

        '''STEP 6: the simulaiton'''
        start = time.time()
        env.run()
        self.sqc_brain.check_parameter()
        end = time.time()
        running_time = end - start
        print('main_training_S running time: ' + str(running_time))
        saveRunningTime(kwargs['seed'], kwargs['dataset_name'], running_time, len(self.wc_list), len(self.m_list))
        #hide by mengxu 2023.08.19
        # self.sqc_brain.loss_record_output(save = 1)

def saveRunningTime(randomSeeds, dataSetName, running_time, wc_num, m_num):
    address_seed = "./sequencing_models/scenario_" + dataSetName + "/running_time_" + str(randomSeeds) + "_small_state_dict" + '{}wc{}m'.format(wc_num, m_num)  # modified by mengxu 2022.10.31
    # fileName1= './MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds)+'_running_time' + dataSetName
    np.save(address_seed, running_time)
    return

def training(dataset_name, seed):
    np.random.seed(int(seed))
    torch.random.manual_seed(int(seed))
    random.seed(int(seed))
    # create the environment instance for simulation
    env = simpy.Environment()
    # create the shop floor instance
    span = 100000
    m_no = 6
    wc_no = 3
    spf = shopfloor(env, span, m_no, wc_no, seed=seed, dataset_name=dataset_name, ifPrint=False)



# span = 10000
# m_no = 6
# wc_no = 3
# spf = shopfloor(env, span, m_no, wc_no)
