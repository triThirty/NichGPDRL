import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import brain_machine_S as brain
import sequencing
#from ann_visualizer.visualize import ann_viz

'''
Load the trained parameters of sequencing agent
'''

class DRL_sequencing(brain.sequencing_brain):
    def __init__(self, env, machine_list, job_creator, *args, **kwargs):
        # initialize the environment and the workcenter to be controlled
        self.env = env
        # get list of alll machines, for collecting the global data
        self.m_list = machine_list
        self.job_creator = job_creator
        # each item is a network
        self.net_dict = {}
        self.kwargs = kwargs

        self.times_using_sequencing_rule = 0 # add by mengxu to check the time it used
        '''
        choose the trained parameters by its reward function
        '''
        if 'reward_function' in kwargs:
            pass
        else:
            print('WARNING: reward function is not specified')
            raise Exception
        # build action NN for each target machine
        if 'validated' in kwargs and kwargs['validated']:
            print("---> VALIDATION <---")
            if 'GPrule_action' in kwargs and kwargs['GPrule_action']:  # modified by mengxu
                self.GPrule_action = True
                self.func_list = [0,1,2,3]
                self.selected_times_each_action = [0, 0, 0, 0]
                self.sequencing_GPtree_list = []
                for idx in self.func_list:
                    dict_best_MTGP_individuals = kwargs['GPrules']
                    # individual = dict_best_MTGP_individuals[idx]
                    individual = dict_best_MTGP_individuals.get(idx)
                    sequencing_rule_tree = individual[0]
                    self.sequencing_GPtree_list.append(sequencing_rule_tree)
                for m in self.m_list:
                    m.GPrule_action = True  # add by mengxu
            else:
                self.GPrule_action = False
                self.func_list = [sequencing.SPT, sequencing.WINQ, sequencing.MS, sequencing.CR]  # original
                self.selected_times_each_action = [0, 0, 0, 0]
                for m in self.m_list:
                    m.GPrule_action = False  # add by mengxu
            # self.func_list = [sequencing.SPT, sequencing.WINQ, sequencing.MS, sequencing.CR]   # original
            self.output_size = len(self.func_list)
            if self.job_creator.pt_range[1]/self.job_creator.pt_range[0] > 2.5:
                scenario = "H"
            else:
                scenario = 'L'
            if self.job_creator.tightness == 2:
                scenario += "H"
            else:
                scenario += 'L'
            try:
                scenario += str(kwargs['reward_function'])
            except:
                pass
            # self.address_seed = "{}/sequencing_models/MC_rwd1.pt"
            seedOfRun = kwargs['seedOfRun']
            # self.address_seed = "{}/sequencing_models/scenario_" + scenario + "/run_" + str(seedOfRun) + "_MC_rwd1.pt"  # modified by mengxu
            if 'intermediate_address' in kwargs:
                self.address_seed = "{}/sequencing_models/scenario_" + scenario + "/run_" + str(
                    seedOfRun) + "_policies/run_" + str(seedOfRun) + "_MC_rwd_time" + str(
                    kwargs['intermediate_address']) + ".pt"  # modified by mengxu 2023.10.25
            else:
                self.address_seed = "{}/sequencing_models/scenario_" + scenario + "/run_" + str(seedOfRun) + "_MC_rwd1.pt"  # modified by mengxu
            # self.address_seed = "{}/sequencing_models/validated_" + scenario + ".pt" # original
            self.input_size = len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.network = brain.network_validated(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            self.build_state = self.state_multi_channel
            for m in self.m_list:
                m.job_sequencing = self.action_sqc_rule
        elif 'ext_validated' in kwargs and kwargs['ext_validated']:
            print("---> EXTENDED VALIDATION <---")
            if 'GPrule_action' in kwargs and kwargs['GPrule_action']:  # modified by mengxu
                self.GPrule_action = True
                self.func_list = [0,1,2,3]
                self.sequencing_GPtree_list = []
                for idx in self.func_list:
                    dict_best_MTGP_individuals = kwargs['GPrules']
                    individual = dict_best_MTGP_individuals.get(idx)
                    sequencing_rule_tree = individual[0]
                    self.sequencing_GPtree_list.append(sequencing_rule_tree)
                for m in self.m_list:
                    m.GPrule_action = True  # add by mengxu
            else:
                self.GPrule_action = False
                self.func_list = [sequencing.SPT, sequencing.WINQ, sequencing.MS, sequencing.CR]  # original
                for m in self.m_list:
                    m.GPrule_action = False  # add by mengxu
            # self.func_list = [sequencing.SPT, sequencing.WINQ, sequencing.MS, sequencing.CR]   # original
            self.output_size = len(self.func_list)
            if self.job_creator.pt_range[1]/self.job_creator.pt_range[0] > 2.5:
                scenario = "H"
            else:
                scenario = 'L'
            if self.job_creator.tightness==2:
                scenario += "H"
            else:
                scenario += 'L'
            try:
                scenario += str(kwargs['reward_function'])
            except:
                pass
            self.address_seed = "{}/sequencing_models/validated_" + scenario + "_ext{}.pt".format(self.job_creator.no_wcs)
            self.input_size = len(self.state_multi_channel(self.m_list[0].sequencing_data_generation()))
            self.network = brain.network_validated(self.input_size, self.output_size)
            self.network.network.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            self.build_state = self.state_multi_channel
            for m in self.m_list:
                m.job_sequencing = self.action_sqc_rule
        else:
            print("WARNING: ANN TYPE NOT SPECIFIED !!!!")

        # network_S_dict = torch.load(self.address_seed.format(sys.path[0])) # add by mengxu 2022.12.14 to print the network
        # print(torch.load(self.address_seed.format(sys.path[0]))) # add by mengxu 2022.12.14 to print the network
        # ann_viz(self.network, title = "network_S")
        print('--------------------------')
        #print("Dictionary of networks:\n",self.net_dict)
        # check if need to show the specific selection
        self.show = False
        if 'show' in kwargs and kwargs['show']:
            self.show = True

    def action_sqc_rule(self, local_data, **kwargs):
        s_t = self.build_state(local_data)
        value = self.network.forward(s_t.reshape([1,1,self.input_size]))
        a_t = torch.argmax(value)
        if self.GPrule_action: #modified by mengxu
            a_t = 3  # add by mengxu to test 2023.12.18
            self.selected_times_each_action[a_t] = self.selected_times_each_action[a_t] + 1
            if len(local_data[0]) >=2:
                self.times_using_sequencing_rule = self.times_using_sequencing_rule + 1
            if 'GPrule_action_data' in kwargs:
                sqc_data_RPrule_action = kwargs['GPrule_action_data']
                job_position = sequencing.GP_pair_S_test(sqc_data_RPrule_action,self.sequencing_GPtree_list[a_t])
        else:
            self.selected_times_each_action[a_t] = self.selected_times_each_action[a_t] + 1
            job_position = self.func_list[a_t](local_data) #original
        # job_position = self.func_list[a_t](local_data)
        return job_position

    def action_direct(self, sqc_data, **kwargs): # strategic idleness is prohibitted
        s_t = self.build_state(sqc_data)
        m_idx = sqc_data[-1]
        value = self.network.forward(s_t.reshape([1]+self.input_size_as_list),m_idx).squeeze()
        a_t = torch.argmax(value)
        job_position, j_idx = self.action_conversion(a_t)
        return job_position

    def check_parameter(self):
        print('------------------ Sequencing Brain Parameter Check ------------------')
        print("Collect from:",self.address_seed)
        print('Trained with Rwd.Func.:',self.kwargs['reward_function'])
        print('State function:',self.build_state.__name__)
        print('ANN architecture:',self.network.__class__.__name__)
        #ann_viz(self.network, title="network_S")
        print('*** SCENARIO:')
        print("Configuration: {} work centers, {} machines".format(len(self.job_creator.wc_list),len(self.m_list)))
        print("PT heterogeneity:",self.job_creator.pt_range)
        print('Due date tightness:',self.job_creator.tightness)
        print('Utilization rate:',self.job_creator.E_utliz)
        print('----------------------------------------------------------------------')
