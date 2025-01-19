import numpy as np
import sys
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import brain_workcenter_R as brain
import routing
#from ann_visualizer.visualize import ann_viz

'''
load trained parameters in experiment
'''

class DRL_routing(brain.routing_brain):
    def __init__(self, env, job_creator, wc_list, **kwargs):
        # initialize the environment and the workcenter to be controlled
        self.env = env
        self.job_creator = job_creator
        self.wc_list = wc_list
        for wc in self.wc_list:
            wc.job_routing = self.action_by_DRL
        # retrive the data of chosen workcenter
        self.m_per_wc = len(self.wc_list[0].m_list)
        all_m_no = len(self.wc_list[0].m_list)*len(wc_list)
        # state space, eah machine generate 4 types of data, along with the processing time of job
        self.input_size = self.m_per_wc*3 + 3
        # action space, consists of all selectable machines
        self.single_agent = False
        if 'GPrule_action' in kwargs and kwargs['GPrule_action']: # modified by mengxu
            self.GPrule_action = True
            self.GPrule_ensemble = False
            self.func_list = [0, 1, 2, 3]
            self.selected_times_each_action = [0, 0, 0, 0]
            # self.func_list = [0,1,2,3,4,5,6,7,8,9]
            # self.selected_times_each_action = [0,0,0,0,0,0,0,0,0,0]
            self.routing_GPtree_list = []
            if 'Single_agent' in kwargs and kwargs['Single_agent']: # modified by mengxu
                self.sequencing_GPtree_list = []
                self.single_agent = True
            for idx in self.func_list:
                dict_best_MTGP_individuals = kwargs['GPrules']
                # individual = dict_best_MTGP_individuals[idx]
                individual = dict_best_MTGP_individuals.get(idx)
                routing_rule_tree = individual[1]
                self.routing_GPtree_list.append(routing_rule_tree)
                if 'Single_agent' in kwargs and kwargs['Single_agent']:  # modified by mengxu
                    sequencing_rule_tree = individual[0]
                    self.sequencing_GPtree_list.append(sequencing_rule_tree)
            for m in self.wc_list:
                m.GPrule_action = True  # add by mengxu
                m.GPrule_ensemble = False
            self.output_size = len(self.routing_GPtree_list)
        elif 'GPrule_ensemble' in kwargs and kwargs['GPrule_ensemble']: # modified by mengxu
            self.GPrule_ensemble = True
            self.GPrule_action = False
            self.func_list = [0,1,2,3]
            self.func_weights_list = [1, 1, 1, 1]
            self.routing_GPtree_list = []
            for idx in self.func_list:
                dict_best_MTGP_individuals = kwargs['GPrules']
                # individual = dict_best_MTGP_individuals[idx]
                individual = dict_best_MTGP_individuals.get(idx)
                routing_rule_tree = individual[1]
                self.routing_GPtree_list.append(routing_rule_tree)
            for m in self.wc_list:
                m.GPrule_ensemble = True
                m.GPrule_action = False  # add by mengxu
            self.output_size = len(self.routing_GPtree_list)
        else:
            self.GPrule_action = False
            self.GPrule_ensemble = False
            for m in self.wc_list:
                m.GPrule_action = False  # add by mengxu
                m.GPrule_ensemble = False
            # action space, consists of all selectable machines
            self.output_size = self.m_per_wc #original
        # self.output_size = self.m_per_wc
        # specify the path to store the model
        self.path = sys.path[0]
        # specify the ANN and state function
        if 'validated' in kwargs and kwargs['validated']:
            # self.address_seed = "{}/routing_models/small_state_dict3wc6m.pt"
            self.action_NN = brain.build_network_small(self.input_size, self.output_size)
            seedOfRun = kwargs['seedOfRun']
            if self.job_creator.pt_range[1] / self.job_creator.pt_range[0] > 2.5:
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
            # dataset_name = kwargs['dataset_name']
            if 'intermediate_address' in kwargs:
                self.address_seed = "{}/routing_models/scenario_" + scenario + "/run_" + str(
                    seedOfRun) + "_policies/run_" + str(seedOfRun) + "_small_state_dict" + '{}wc{}m'.format(len(wc_list),self.m_per_wc*len(wc_list)) + "_time" + str(
                    kwargs['intermediate_address']) + ".pt"  # modified by mengxu 2023.10.25
            else:
                self.address_seed = "{}/routing_models/scenario_" + scenario + "/run_" + str(
                    seedOfRun) + "_small_state_dict" + '{}wc{}m'.format(len(wc_list),self.m_per_wc*len(wc_list)) + ".pt"  # modified by mengxu 2022.10.31
            self.action_NN = brain.build_network_small(self.input_size, self.output_size)
            # the following is the original
            # if self.m_per_wc == 2:
            #     # self.address_seed = "{}/routing_models/validated_2machine_small.pt"
            #     seedOfRun = kwargs['seedOfRun']
            #     if self.job_creator.pt_range[1] / self.job_creator.pt_range[0] > 2.5:
            #         scenario = "H"
            #     else:
            #         scenario = 'L'
            #     if self.job_creator.tightness == 2:
            #         scenario += "H"
            #     else:
            #         scenario += 'L'
            #     try:
            #         scenario += str(kwargs['reward_function'])
            #     except:
            #         pass
            #     # dataset_name = kwargs['dataset_name']
            #     if 'intermediate_address' in kwargs:
            #         self.address_seed = "{}/routing_models/scenario_" + scenario + "/run_" + str(
            #             seedOfRun) + "_policies/run_" + str(seedOfRun) + "_small_state_dict3wc6m_time" + str(kwargs['intermediate_address']) + ".pt"  # modified by mengxu 2023.10.25
            #     else:
            #         self.address_seed = "{}/routing_models/scenario_" + scenario + "/run_" + str(
            #             seedOfRun) + "_small_state_dict3wc6m.pt"  # modified by mengxu 2022.10.31
            #     self.action_NN = brain.build_network_small(self.input_size, self.output_size)
            # if self.m_per_wc == 3:
            #     self.address_seed = "{}/routing_models/validated_3machine_medium.pt"
            #     self.action_NN = brain.build_network_medium(self.input_size, self.output_size)
            # if self.m_per_wc == 4:
            #     self.address_seed = "{}/routing_models/validated_4machine_large.pt"
            #     self.action_NN = brain.build_network_large(self.input_size, self.output_size)
            self.build_state = self.state_deeper
            self.action_NN.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            # network_R_dict = torch.load(self.address_seed.format(sys.path[0])) #add by mengxu 2022.12.14 to print the network
            # print(torch.load(self.address_seed.format(sys.path[0])))#add by mengxu 2022.12.14 to print the network
            self.action_NN.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            print("---> VALIDATION mode ON <---")
        elif 'TEST' in kwargs and kwargs['TEST']:
            self.address_seed = "{}/routing_models/TEST_state_dict.pt"
            self.action_NN = brain.build_network_TEST(self.input_size, self.output_size)
            self.build_state = self.state_deeper
            self.action_NN.load_state_dict(torch.load(self.address_seed.format(sys.path[0])))
            self.action_NN.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
            print("---> TEST mode ON <---")

    def normalise_meng(self, vector):
        sum = np.sum(vector)
        vector_normalised = []
        for value in vector:
            vector_normalised.append(value/sum)
        return np.array(vector_normalised)

    def action_by_DRL(self, job_idx, routing_data, job_pt, job_slack, wc_idx, *args, **kwargs):
        s_t = self.build_state(routing_data, job_pt, job_slack, wc_idx)
        # input state to policy network, produce the state-action value
        value = self.action_NN.forward(s_t.reshape(1,1,self.input_size),wc_idx)
        # generate the action
        a_t = torch.argmax(value)
        if self.GPrule_action:  # modified by mengxu
            self.selected_times_each_action[a_t] = self.selected_times_each_action[a_t] + 1
            if 'GPrule_action_data' in kwargs:
                if self.single_agent:
                    for wc in self.wc_list:
                        for m in wc.m_list:
                            m.job_sequencing_tree = self.sequencing_GPtree_list[a_t]
                routing_data_GPrule_action = kwargs['GPrule_action_data']
                machine_position = routing.GP_pair_R_test(self.routing_GPtree_list[a_t], job_idx,
                                                          routing_data_GPrule_action, job_pt,
                                                          kwargs['next_pt'], kwargs['OWT'], kwargs['WKR'],
                                                          kwargs['NOR'], kwargs['weight_list'],
                                                          kwargs['waiting_time'], job_slack)
                machine_position = torch.tensor(machine_position) #todo: need double check 2023.12.04
                return machine_position
        elif self.GPrule_ensemble:  # modified by mengxu
            for i in range(len(self.func_weights_list)):
                weight_i = float(value[0][i].float())
                self.func_weights_list[i] = weight_i
            if 'GPrule_action_data' in kwargs:
                routing_data_GPrule_action = kwargs['GPrule_action_data']
                ensemble_priority = 0
                # print(self.func_weights_list)
                for i in range(len(self.func_weights_list)):
                    machine_priority = self.func_weights_list[i] * self.normalise_meng(routing.GP_pair_ensemble_R_test(
                                                                    self.routing_GPtree_list[i], job_idx,
                                                                    routing_data_GPrule_action, job_pt,
                                                                    kwargs['next_pt'], kwargs['OWT'], kwargs['WKR'],
                                                                    kwargs['NOR'], kwargs['weight_list'],
                                                                    kwargs['waiting_time'], job_slack))
                    ensemble_priority = ensemble_priority + machine_priority
                # print(ensemble_priority)
                machine_position = ensemble_priority.argmin()
                machine_position = torch.tensor(machine_position)
                return machine_position
        else:
            machine_position = a_t  # original
            return machine_position
        #print(value,a_t)
        #print('Policy NN choose action')
        # return a_t

    def check_parameter(self):
        print('------------------ Routing Brain Parameter Check ------------------')
        print("Collect from:",self.address_seed)
        print('State function:',self.build_state.__name__)
        print('ANN architecture:',self.action_NN.__class__.__name__)
        # ann_viz(self.action_NN, title="network_R")  # add by mengxu 2022.12.14 to print the network
        print('*** SCENARIO:')
        print("Configuration: {} work centers, {} machines".format(len(self.job_creator.wc_list),len(self.job_creator.m_list)))
        print("PT heterogeneity:",self.job_creator.pt_range)
        print('Due date tightness:',self.job_creator.tightness)
        print('Utilization rate:',self.job_creator.E_utliz)
        if self.GPrule_action:
            print('Times select each action:', self.selected_times_each_action)
        print('----------------------------------------------------------------------')
