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
import openpyxl
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
        # dict_top_inds_MTGP_individuals_HH = nichmtload.load_top_inds_from_final_gen(kwargs['seedOfRun'], 'HH')
        # dict_top_inds_MTGP_individuals_HL = nichmtload.load_top_inds_from_final_gen(kwargs['seedOfRun'], 'HL')
        # dict_top_inds_MTGP_individuals_LH = nichmtload.load_top_inds_from_final_gen(kwargs['seedOfRun'], 'LH')
        # dict_top_inds_MTGP_individuals_LL = nichmtload.load_top_inds_from_final_gen(kwargs['seedOfRun'], 'LL')
        # dict_top_inds_MTGP_individuals = []
        # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_HH.get(0))
        # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_HL.get(0))
        # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_LH.get(0))
        # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_LL.get(0))

        # specify the architecture of DRL
        if 'arch' and 'global_reward' in kwargs:
            arch = kwargs['arch'] + "=True"
            global_reward = 'global_reward={}'.format(kwargs['global_reward'])
            seedOfRun = 'seedOfRun={}'.format(kwargs['seedOfRun'])
            model_address = 'model_address={}'.format(kwargs['model_address'])
            GPrule_action = 'GPrule_action=False'
            GPrules = 'GPrules={}'.format(dict_top_inds_MTGP_individuals)
            validated = 'validated=1'
            # dataset_name = 'dataset_name={}'.format(kwargs['dataset_name']) #modified by mengxu
            order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list,{},{},{},{},{},{})".format(
                arch, global_reward, seedOfRun, model_address, GPrule_action, GPrules)
            exec(order)

        # specify the architecture of DRL
        if 'DRL_S' in kwargs and kwargs['DRL_S']:
            print("---> DRL Sequencing mode ON <---")
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

    def simulation(self):
        self.env.run()

class shopfloorMTGPSeq:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, **kwargs):
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
            # self.job_creator = job_creation.creation \
            #     (self.env, self.span, self.m_list, self.wc_list, [5, 25], 2, 0.9, random_seed=True, ifPrint = self.ifPrint)
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5,25], 2, 0.9, seed=kwargs['seed'], random_seed = True, ifPrint = self.ifPrint) #ifPrint = self.ifPrint is added by mengxu to make it clearer when using MTGP to train
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [5, 25], 3, 0.9, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 2, 0.9, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 3, 0.9, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
            #self.job_creator.output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
            # wc.setJobRoutingTree(routing_tree)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])
            m.setJobSequencingTree(sequencing_tree)


        '''STEP 5: set sequencing or routing rules, and DRL'''
        # check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            # if 'tree_sequencing' in kwargs:
            #     print(str(kwargs['tree_sequencing'])) #add by mengxu to check if this is right! 2022.10.15
            #     order = "m.tree_sequencing = " + str(kwargs['tree_sequencing'])
            #     try:
            #         exec(order)
            #     except:
            #         if self.ifPrint:
            #             print("Rule assigned to machine {} is invalid !".format(m.m_idx))
            #         raise Exception
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
        # if 'arch' and 'global_reward' in kwargs:
        #     arch = kwargs['arch'] + "=True"
        #     global_reward = 'global_reward={}'.format(kwargs['global_reward'])
        #     order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, {},{})".format(arch,global_reward)
        #     exec(order)

    def simulation(self):
        self.env.run()

class shopfloor_niching:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, routing_tree, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        self.ifPrint = kwargs['ifPrint'] # added by mengxu

        # for niching by mengxu 2023.10.18
        # todo: need to change to GP evolved rule later 2023.10.18
        self.sequencingDecisionSituations = []
        self.routingDecisionSituations = []
        # phenoCharacterisation = []
        # referenceSequencingRule = "TIS"
        # sequencingPhenoCharacterisation = SequencingPhenoCharacterisation.SequencingPhenoCharacterisation(referenceSequencingRule)
        # referenceRoutingRule = "WIQ"
        # routingPhenoCharacterisation = RoutingPhenoCharacterisation.RoutingPhenoCharacterisation(referenceRoutingRule)


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
            # self.job_creator = job_creation.creation \
            #     (self.env, self.span, self.m_list, self.wc_list, [5, 25], 2, 0.9, random_seed=True, ifPrint = self.ifPrint)
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5,25], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint = self.ifPrint) #ifPrint = self.ifPrint is added by mengxu to make it clearer when using MTGP to train
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5, 25], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [10, 20], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [10, 20], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
            #self.job_creator.output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
            wc.setJobRoutingTree(routing_tree)
            wc.setGetDecisionSituation(self.routingDecisionSituations)
        for i,m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i/m_per_wc)
            m.initialization(self.m_list,self.wc_list,self.job_creator,self.wc_list[wc_idx])
            m.setJobSequencingTree(sequencing_tree)
            m.setGetDecisionSituation(self.sequencingDecisionSituations)


        '''STEP 5: set sequencing or routing rules, and DRL'''
        # check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            # if 'tree_sequencing' in kwargs:
            #     print(str(kwargs['tree_sequencing'])) #add by mengxu to check if this is right! 2022.10.15
            #     order = "m.tree_sequencing = " + str(kwargs['tree_sequencing'])
            #     try:
            #         exec(order)
            #     except:
            #         if self.ifPrint:
            #             print("Rule assigned to machine {} is invalid !".format(m.m_idx))
            #         raise Exception
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
        # if 'arch' and 'global_reward' in kwargs:
        #     arch = kwargs['arch'] + "=True"
        #     global_reward = 'global_reward={}'.format(kwargs['global_reward'])
        #     order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, {},{})".format(arch,global_reward)
        #     exec(order)

    def simulation(self):
        self.env.run()

    def getDecisionSituations(self, num_decision):
        decisionSituations = []
        if len(self.sequencingDecisionSituations) < num_decision:
            print("Error in get enough number of self.sequencingDecisionSituations)")

        if len(self.routingDecisionSituations) < num_decision:
            print("Error in get enough number of self.routingDecisionSituations)")

        np.random.shuffle(self.sequencingDecisionSituations)
        subset_sequencingDecisionSituations = []
        tryTimes = 0
        while len(subset_sequencingDecisionSituations) < num_decision and tryTimes < len(self.sequencingDecisionSituations):
            # if len(self.sequencingDecisionSituations[tryTimes].getData()[3]) == 2: #todo: need to decide this number always can not get 7 by mengxu 2023.10.18
            subset_sequencingDecisionSituations.append(self.sequencingDecisionSituations[tryTimes])
            tryTimes = tryTimes + 1

        print("tryTimes for sequencing = " + str(tryTimes))
        print("size for sequencing = " + str(len(subset_sequencingDecisionSituations)))

        np.random.shuffle(self.routingDecisionSituations)
        subset_routingDecisionSituations = []
        tryTimes = 0
        while len(subset_routingDecisionSituations) < num_decision and tryTimes < len(self.routingDecisionSituations):
            # if len(self.routingDecisionSituations[tryTimes].getData()[1]) == 2:
            subset_routingDecisionSituations.append(self.routingDecisionSituations[tryTimes])
            tryTimes = tryTimes + 1
        print("tryTimes for routing = " + str(tryTimes))
        print("size for routing = " + str(len(subset_routingDecisionSituations)))

        decisionSituations.append(subset_sequencingDecisionSituations)
        decisionSituations.append(subset_routingDecisionSituations)
        return decisionSituations

    def routing_characterise(self, rule, decisionSituations):
        charlist = []
        for i in range(len(decisionSituations)):
            routingDecision = decisionSituations[i].clone()
            routing_data = routingDecision.getData()
            ranks_rule = routing.GP_pair_R_ranks(rule, routing_data[0], routing_data[1], routing_data[2], routing_data[3], routing_data[4],
                                                   routing_data[5], routing_data[6], routing_data[7], routing_data[8], routing_data[9])
            idxBest = 0
            for j in range(len(ranks_rule)):
                if ranks_rule[j] < ranks_rule[idxBest]:
                    idxBest = j
            charlist.append(idxBest)

        return charlist

    def sequencing_characterise(self, rule, decisionSituations):
        charlist = []

        for i in range(len(decisionSituations)):
            sequencingDecision = decisionSituations[i].clone()
            sequencing_data = sequencingDecision.getData()
            ranks_rule = sequencing.GP_pair_S_ranks(sequencing_data, rule)
            idxBest = 0
            for j in range(len(ranks_rule)):
                if ranks_rule[j] < ranks_rule[idxBest]:
                    idxBest = j

            charlist.append(idxBest)

        return charlist

class shopfloor_niching_seq:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, **kwargs):
        '''STEP 1: create environment instances and specifiy simulation span '''
        self.env=env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        self.ifPrint = kwargs['ifPrint'] # added by mengxu

        # todo: need to change to GP evolved rule later 2023.10.18
        self.sequencingDecisionSituations = []
        self.routingDecisionSituations = []

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
            # self.job_creator = job_creation.creation \
            #     (self.env, self.span, self.m_list, self.wc_list, [5, 25], 2, 0.9, random_seed=True, ifPrint = self.ifPrint)
            if 'dataset_name' in kwargs:
                if kwargs['dataset_name'] == 'HH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                        [5,25], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint = self.ifPrint) #ifPrint = self.ifPrint is added by mengxu to make it clearer when using MTGP to train
                elif kwargs['dataset_name'] == 'HL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [5, 25], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LH':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 2, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
                elif kwargs['dataset_name'] == 'LL':
                    self.job_creator = job_creation.creation(self.env, self.span, self.m_list, self.wc_list, \
                                                             [10, 20], 3, 0.99, seed=kwargs['seed'], random_seed = True, ifPrint=self.ifPrint)
            #self.job_creator.output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        '''STEP 4: initialize machines and work centers'''
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
        for i, m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i / m_per_wc)
            m.initialization(self.m_list, self.wc_list, self.job_creator, self.wc_list[wc_idx])
            m.setJobSequencingTree(sequencing_tree)
            m.setGetDecisionSituation(self.sequencingDecisionSituations)


        '''STEP 5: set sequencing or routing rules, and DRL'''
        # check if need to reset sequencing rule
        if 'sequencing_rule' in kwargs:
            # if 'tree_sequencing' in kwargs:
            #     print(str(kwargs['tree_sequencing'])) #add by mengxu to check if this is right! 2022.10.15
            #     order = "m.tree_sequencing = " + str(kwargs['tree_sequencing'])
            #     try:
            #         exec(order)
            #     except:
            #         if self.ifPrint:
            #             print("Rule assigned to machine {} is invalid !".format(m.m_idx))
            #         raise Exception
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
        # if 'arch' and 'global_reward' in kwargs:
        #     arch = kwargs['arch'] + "=True"
        #     global_reward = 'global_reward={}'.format(kwargs['global_reward'])
        #     order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, {},{})".format(arch,global_reward)
        #     exec(order)

    def simulation(self):
        self.env.run()

    def getDecisionSituations(self, num_decision):
        decisionSituations = []
        if len(self.sequencingDecisionSituations) < num_decision:
            print("Error in get enough number of self.sequencingDecisionSituations)")

        np.random.shuffle(self.sequencingDecisionSituations)
        subset_sequencingDecisionSituations = []
        tryTimes = 0
        while len(subset_sequencingDecisionSituations) < num_decision and tryTimes < len(self.sequencingDecisionSituations):
            if len(self.sequencingDecisionSituations[tryTimes].getData()[3]) == 3: #todo: need to decide this number always can not get 7 by mengxu 2023.10.18
                subset_sequencingDecisionSituations.append(self.sequencingDecisionSituations[tryTimes])
            tryTimes = tryTimes + 1

        print("tryTimes for sequencing = " + str(tryTimes))
        print("size for sequencing = " + str(len(subset_sequencingDecisionSituations)))

        decisionSituations.append(subset_sequencingDecisionSituations)
        return decisionSituations

    def sequencing_characterise(self, rule, decisionSituations):
        charlist = []

        for i in range(len(decisionSituations)):
            sequencingDecision = decisionSituations[i].clone()
            sequencing_data = sequencingDecision.getData()
            ranks_rule = sequencing.GP_pair_S_ranks(sequencing_data, rule)
            idxBest = 0
            for j in range(len(ranks_rule)):
                if ranks_rule[j] < ranks_rule[idxBest]:
                    idxBest = j

            charlist.append(idxBest)

        return charlist

def hammingDistance(vectorA, vectorB):
    dis=0
    length_A = len(vectorA)
    length_B = len(vectorB)
    length = length_A
    if length_A > length_B:
        length = length_B
    for i in range(length):
        if (vectorA[i] != vectorB[i]):
            dis = dis + 1
    return dis/length


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

    dataSetName = dataset_name
    seedOfRun = int(seedOfRun)

    save_selected_time = True
    save_selected_time_only_sequencing = True


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
    # dict_top_inds_MTGP_individuals_HH = nichmtload.load_top_inds_from_final_gen(seedOfRun, 'HH')
    # dict_top_inds_MTGP_individuals_HL = nichmtload.load_top_inds_from_final_gen(seedOfRun, 'HL')
    # dict_top_inds_MTGP_individuals_LH = nichmtload.load_top_inds_from_final_gen(seedOfRun, 'LH')
    # dict_top_inds_MTGP_individuals_LL = nichmtload.load_top_inds_from_final_gen(seedOfRun, 'LL')
    # dict_top_inds_MTGP_individuals = []
    # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_HH.get(0))
    # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_HL.get(0))
    # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_LH.get(0))
    # dict_top_inds_MTGP_individuals.append(dict_top_inds_MTGP_individuals_LL.get(0))

    env = simpy.Environment()
    rule_R = 'GP_pair_R_test'
    rule_S = 'GP_pair_S_test'
    individual = dict_top_inds_MTGP_individuals[0]
    sequencing_tree = individual[0]
    only_sequencing_rule = False
    if len(individual) == 2:
        routing_tree = individual[1]
        spf_niching = shopfloor_niching(env, span, m_no, wc_no, sequencing_tree, routing_tree, routing_rule=rule_R,
                            sequencing_rule=rule_S, seed=666, ifPrint=False, dataset_name=dataset_name)
    else:
        only_sequencing_rule = True
        rule_R = 'EA'
        spf_niching = shopfloor_niching_seq(env, span, m_no, wc_no, sequencing_tree, routing_rule=rule_R,
                                    sequencing_rule=rule_S, seed=666, ifPrint=False, dataset_name=dataset_name)
    spf_niching.simulation()
    decisionSituations = spf_niching.getDecisionSituations(100)
    all_sequencing_decisions_difference_all_NichingGP = []
    if len(individual) == 2:
        all_routing_decisions_difference_all_NichingGP = []

    all_run_sequencing_action_selection_times_all_DRL = []
    if len(individual) == 2:
        all_run_routing_action_selection_times_all_DRL = []

    # testSeeds = 123453
    testSeeds = 123453443
    np.random.seed(int(testSeeds))


    for run in range(iteration):
        all_sequencing_decisions_all_NichingGP = []
        if not only_sequencing_rule:
            all_routing_decisions_all_NichingGP = []
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

        length = len(dict_top_inds_MTGP_individuals)
        if length > 4:
            length = 4
        for i in range(length):
            algo = 'Nichtop_' + str(i)
            if run == 0:
                MTGP.append(algo)
            individual = dict_top_inds_MTGP_individuals.get(i)
            # individual = dict_top_inds_MTGP_individuals[i]
            sequencing_rule_tree = individual[0]
            if len(individual) == 2:
                routing_rule_tree = individual[1]
            # np.random.seed(int(seed))  # add by mengxu 2022.10.31
                env = simpy.Environment()
                spf = shopfloorMTGP(env, span, m_no, wc_no, sequencing_rule_tree, routing_rule_tree,
                                    routing_rule='GP_pair_R_test', sequencing_rule='GP_pair_S_test', seed=seed, seedOfRun = seedOfRun, ifPrint=False,
                                    dataset_name=dataSetName)
            else:
                env = simpy.Environment()
                spf = shopfloorMTGPSeq(env, span, m_no, wc_no, sequencing_rule_tree,
                                    routing_rule='EA', sequencing_rule='GP_pair_S_test', seed=seed,
                                    seedOfRun=seedOfRun, ifPrint=False,
                                    dataset_name=dataSetName)

            spf.simulation()
            output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
            sum_record[run].append(cumulative_tard[-1])
            benchmark_record[run].append(cumulative_tard[-1])
            max_record[run].append(tard_max)
            rate_record[run].append(tard_rate)

            if run == 0:
                all_sequencing_decisions_all_NichingGP.append(spf_niching.sequencing_characterise(sequencing_rule_tree, decisionSituations[0]))
                if not only_sequencing_rule:
                    all_routing_decisions_all_NichingGP.append(spf_niching.routing_characterise(routing_rule_tree, decisionSituations[1]))

        if run == 0:
            print('-------------- Compare difference between all sequencing decisions --------------')
            all_sequencing_decisions_difference_all_NichingGP_current_run = []
            if not only_sequencing_rule:
                all_routing_decisions_difference_all_NichingGP_current_run = []
            for i in range(len(all_sequencing_decisions_all_NichingGP)):
                for j in range(i + 1, len(all_sequencing_decisions_all_NichingGP)):
                    seq_dis = hammingDistance(all_sequencing_decisions_all_NichingGP[i],
                                              all_sequencing_decisions_all_NichingGP[j])
                    print("-------------sequencing----------------")
                    print(all_sequencing_decisions_all_NichingGP[i])
                    print(all_sequencing_decisions_all_NichingGP[j])
                    if not only_sequencing_rule:
                        rout_dis = hammingDistance(all_routing_decisions_all_NichingGP[i],
                                                   all_routing_decisions_all_NichingGP[j])
                        print("-------------routing----------------")
                        print(all_routing_decisions_all_NichingGP[i])
                        print(all_routing_decisions_all_NichingGP[j])
                    if not only_sequencing_rule:
                        print("Sequencing difference and routing difference between niching GP " + str(
                            i) + " and niching GP " + str(j) + " : " +
                              str(seq_dis) + " and " + str(rout_dis))
                    else:
                        print("Sequencing difference between niching GP " + str(
                            i) + " and niching GP " + str(j) + " : " +
                              str(seq_dis))
                    all_sequencing_decisions_difference_all_NichingGP_current_run.append(seq_dis)
                    if not only_sequencing_rule:
                        all_routing_decisions_difference_all_NichingGP_current_run.append(rout_dis)
            print('----------------------------')
            all_sequencing_decisions_difference_all_NichingGP.append(all_sequencing_decisions_difference_all_NichingGP_current_run)
            if not only_sequencing_rule:
                all_routing_decisions_difference_all_NichingGP.append(all_routing_decisions_difference_all_NichingGP_current_run)

        # RL test
        for idx,x in enumerate(DRLs):
            env = simpy.Environment()
            # spf = shopfloor(env, span, m_no, wc_no, DRL_R=True, DRL_S=True, seed=seed, dataset_name=dataSetName)# which one should I use
            spf = shopfloor(env, span, m_no, wc_no, arch = x, global_reward = reward_mechanism[idx], DRL_S=True, seed=seed, seedOfRun = seedOfRun, model_address=x, dataset_name=dataSetName)
            # spf = shopfloor(env, span, m_no, wc_no, arch = x, global_reward = reward_mechanism[idx], seed = seed)
            spf.simulation()
            print("Times trigger sequencing decisions: " + str(spf.sequencing_brain.times_using_sequencing_rule))
            if save_selected_time:
                all_run_sequencing_action_selection_times_all_DRL.append(spf.sequencing_brain.selected_times_each_action)
                if not only_sequencing_rule and not save_selected_time_only_sequencing:
                    all_run_routing_action_selection_times_all_DRL.append(spf.routing_brain.selected_times_each_action)
            output_time, cumulative_tard, tard_mean, tard_max, tard_rate = spf.job_creator.tardiness_output()
            # add by mengxu to test
            # spf.job_creator.output()
            # spf.job_creator.final_output()
            # # add by mengxu to test
            sum_record[run].append(cumulative_tard[-1])
            max_record[run].append(tard_max)
            rate_record[run].append(tard_rate)

    title = benchmark + MTGP + ['Integrated_DRL']
    # title = benchmark + MTGP

    print('-------------- Complete Record --------------')
    print(tabulate(sum_record, headers=title))
    print('-------------- Average Performance --------------')

    # get the performnce without DRL
    # avg_b = np.mean(benchmark_record,axis=0)
    # ratio_b = np.around(avg_b/avg_b.max()*100,2)
    # winning_rate_b = np.zeros(len(title))
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
        .format(title[rule],avg[rule],max[rule],ratio[rule],tardy_rate[rule],winning_rate[rule],winning_rate[rule]))

    if export_result:
        df_win_rate = DataFrame([winning_rate], columns=title)
        #print(df_win_rate)
        df_sum = DataFrame(sum_record, columns=title)
        #print(df_sum)
        df_tardy_rate = DataFrame(rate_record, columns=title)
        #print(df_tardy_rate)
        df_max = DataFrame(max_record, columns=title)
        #print(df_max)
        df_before_win_rate = DataFrame([winning_rate], columns=title)
        address = sys.path[0] + '/experiment_result/scenario_' + dataSetName +'/top_n_NichingMTGP_DRL_test_' + dataSetName + '_run_' + str(seedOfRun) + '_val.xlsx'
        # address = sys.path[0]+'/experiment_result/RAW_RA_val.xlsx'
        Excelwriter = pd.ExcelWriter(address,engine="xlsxwriter")
        dflist = [df_win_rate, df_sum, df_tardy_rate, df_max, df_before_win_rate]
        sheetname = ['win rate','sum', 'tardy rate', 'maximum','before win rate']

        for i,df in enumerate(dflist):
            df.to_excel(Excelwriter, sheet_name=sheetname[i], index=False)
        Excelwriter.save()
        print('export to {}'.format(address))

    dict_difference = {"sequencing": all_sequencing_decisions_difference_all_NichingGP[0], "routing": all_routing_decisions_difference_all_NichingGP[0]}
    address_dict_difference = sys.path[0] + '/experiment_result/scenario_' + dataSetName +'/each_pair_NichingGP_difference_' + dataSetName + '_run_' + str(seedOfRun) + '_val.xlsx'
    Excelwriter = pd.ExcelWriter(address_dict_difference,engine="xlsxwriter")
    DataFrame(dict_difference).to_excel(Excelwriter, index=False)
    Excelwriter.save()

    print("All sequencing difference: " + str(all_sequencing_decisions_difference_all_NichingGP))
    print("All routing difference: " + str(all_routing_decisions_difference_all_NichingGP))

    if save_selected_time:
        workbook_sequencing = openpyxl.Workbook()
        sheet_sequencing = workbook_sequencing.active
        for row in all_run_sequencing_action_selection_times_all_DRL:
            sheet_sequencing.append(row)
        address_sequencing = sys.path[
                                      0] + '/experiment_result/scenario_' + dataSetName + '/all_instance_sequencing_action_selection_times_all_DRL_' + dataSetName + '_run_' + str(
            seedOfRun) + '_val.xlsx'
        workbook_sequencing.save(address_sequencing)

        if not only_sequencing_rule and not save_selected_time_only_sequencing:
            workbook_routing = openpyxl.Workbook()
            sheet_routing = workbook_routing.active
            for row in all_run_routing_action_selection_times_all_DRL:
                sheet_routing.append(row)
            address_routing = sys.path[
                                          0] + '/experiment_result/scenario_' + dataSetName + '/all_instance_routing_action_selection_times_all_DRL_' + dataSetName + '_run_' + str(
                seedOfRun) + '_val.xlsx'
            workbook_routing.save(address_routing)
