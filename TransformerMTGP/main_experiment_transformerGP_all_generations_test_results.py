import simpy
import sys
import re

sys.path
import numpy as np
from tabulate import tabulate

import util.LoadIndividual as mtload
import util.agent_machine as agent_machine
import util.agent_workcenter as agent_workcenter
import util.sequencing as sequencing
import util.routing as routing
import util.job_creation as job_creation
from util import saveFile


class shopfloorMTGP:
    def __init__(self, env, span, m_no, wc_no, sequencing_tree, routing_tree, **kwargs):
        """STEP 1: create environment instances and specifiy simulation span"""
        self.env = env
        self.span = span
        self.m_no = m_no
        self.m_list = []
        self.wc_no = wc_no
        self.wc_list = []
        self.ifPrint = kwargs["ifPrint"]  # added by mengxu
        # self.sequencingTree = sequencing_tree
        # self.routingTree = routing_tree
        # self.routingRule = kwargs['tree_routing']
        m_per_wc = int(self.m_no / self.wc_no)
        """STEP 2.1: create instances of machines"""
        for i in range(m_no):
            expr1 = """self.m_{} = agent_machine.machine(env, {}, print = 0)""".format(
                i, i
            )  # create machines
            exec(expr1)
            expr2 = """self.m_list.append(self.m_{})""".format(i)  # add to machine list
            exec(expr2)
        # print(self.m_list)
        """STEP 2.2: create instances of work centers"""
        cum_m_idx = 0
        for i in range(wc_no):
            x = [self.m_list[m_idx] for m_idx in range(cum_m_idx, cum_m_idx + m_per_wc)]
            # print(x)
            expr1 = """self.wc_{} = agent_workcenter.workcenter(env, {}, x)""".format(
                i, i
            )  # create work centers
            exec(expr1)
            expr2 = """self.wc_list.append(self.wc_{})""".format(
                i
            )  # add to machine list
            exec(expr2)
            cum_m_idx += m_per_wc
        # print(self.wc_list)

        """STEP 3: initialize the job creator"""
        # env, span, machine_list, workcenter_list, number_of_jobs, pt_range, due_tightness, E_utliz
        if "seed" in kwargs:
            if "dataset_name" in kwargs:
                if kwargs["dataset_name"] == "HH":
                    self.job_creator = job_creation.creation(
                        self.env,
                        self.span,
                        self.m_list,
                        self.wc_list,
                        [5, 25],
                        2,
                        0.9,
                        seed=kwargs["seed"],
                        ifPrint=False,
                    )
                elif kwargs["dataset_name"] == "HL":
                    self.job_creator = job_creation.creation(
                        self.env,
                        self.span,
                        self.m_list,
                        self.wc_list,
                        [5, 25],
                        3,
                        0.9,
                        seed=kwargs["seed"],
                        ifPrint=False,
                    )
                elif kwargs["dataset_name"] == "LH":
                    self.job_creator = job_creation.creation(
                        self.env,
                        self.span,
                        self.m_list,
                        self.wc_list,
                        [10, 20],
                        2,
                        0.9,
                        seed=kwargs["seed"],
                        ifPrint=False,
                    )
                elif kwargs["dataset_name"] == "LL":
                    self.job_creator = job_creation.creation(
                        self.env,
                        self.span,
                        self.m_list,
                        self.wc_list,
                        [10, 20],
                        3,
                        0.9,
                        seed=kwargs["seed"],
                        ifPrint=False,
                    )
            # self.job_creator.output()
        else:
            print("WARNING: seed is not fixed !!")
            raise Exception

        """STEP 4: initialize machines and work centers"""
        for wc in self.wc_list:
            wc.print_info = 0
            wc.initialization(self.job_creator)
            wc.setJobRoutingTree(routing_tree)
        for i, m in enumerate(self.m_list):
            m.print_info = 0
            wc_idx = int(i / m_per_wc)
            m.initialization(
                self.m_list, self.wc_list, self.job_creator, self.wc_list[wc_idx]
            )
            m.setJobSequencingTree(sequencing_tree)

        """STEP 5: set sequencing or routing rules, and DRL"""
        # check if need to reset sequencing rule
        if "sequencing_rule" in kwargs:
            if self.ifPrint:
                print(
                    "Taking over: machines use {} sequencing rule".format(
                        kwargs["sequencing_rule"]
                    )
                )
            for m in self.m_list:
                order = "m.job_sequencing = sequencing." + kwargs["sequencing_rule"]
                # order = "m.job_sequencing = sequencing." + kwargs['sequencing_rule']
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print(
                            "Rule assigned to machine {} is invalid !".format(m.m_idx)
                        )
                    raise Exception

        # check if need to reset routing rule
        if "routing_rule" in kwargs:
            if self.ifPrint:
                print(
                    "Taking over: workcenters use {} routing rule".format(
                        kwargs["routing_rule"]
                    )
                )
            for wc in self.wc_list:
                order = "wc.job_routing = routing." + kwargs["routing_rule"]
                try:
                    exec(order)
                except:
                    if self.ifPrint:
                        print(
                            "Rule assigned to workcenter {} is invalid !".format(
                                wc.wc_idx
                            )
                        )
                    raise Exception

        # # specify the architecture of DRL
        # if 'arch' and 'global_reward' in kwargs:
        #     arch = kwargs['arch'] + "=True"
        #     global_reward = 'global_reward={}'.format(kwargs['global_reward'])
        #     order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, {},{})".format(arch,global_reward)
        #     exec(order)

    def simulation(self):
        self.env.run()


span = 1000
m_no = 6
wc_no = 3
sum_record = []
benchmark_record = []
max_record = []
rate_record = []
iteration = 100


def main(config):
    dataSetName = config.scenarios

    dict_best_MTGP_individuals = mtload.load_individual_from_gen(config)
    dict_best_MTGP_individuals_dict = mtload.load_individual_from_gen_json_format(
        config
    )

    testSeeds = 123453
    # I think this is wrong, actually I should use totally same randomseed for test of all the runs 2022.10.27
    np.random.seed(int(testSeeds))

    for run in range(iteration):
        print("******************* ITERATION-{} *******************".format(run))
        sum_record.append([])
        benchmark_record.append([])
        max_record.append([])
        rate_record.append([])
        seed = np.random.randint(2000000000)

        for idx, individual in enumerate(dict_best_MTGP_individuals):
            ind_dict = dict_best_MTGP_individuals_dict[idx]
            sequencing_rule_tree = individual[0]
            routing_rule_tree = individual[1]
            env = simpy.Environment()
            spf = shopfloorMTGP(
                env,
                span,
                m_no,
                wc_no,
                sequencing_rule_tree,
                routing_rule_tree,
                routing_rule="GP_pair_R_test",
                sequencing_rule="GP_pair_S_test",
                seed=seed,
                ifPrint=False,
                dataset_name=dataSetName,
            )

            spf.simulation()
            output_time, cumulative_tard, tard_mean, tard_max, tard_rate = (
                spf.job_creator.tardiness_output()
            )
            sum_record[run].append(cumulative_tard[-1])
            ind_dict["fitness"] += cumulative_tard[-1]
            benchmark_record[run].append(cumulative_tard[-1])
            max_record[run].append(tard_max)
            rate_record[run].append(tard_rate)

    for ind in dict_best_MTGP_individuals_dict:
        ind["fitness"] = ind["fitness"] / iteration
    saveFile.save_each_gen_best_individual_on_test_dataset(
        config, dict_best_MTGP_individuals_dict
    )
