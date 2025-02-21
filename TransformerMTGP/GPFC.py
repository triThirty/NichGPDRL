import simpy
from deap import base
from deap import creator
from deap import gp
import TransformerMTGP.multi_tree as mt
from TransformerMTGP import ea_simple_elitism
from TransformerMTGP.ParallelToolbox import ParallelToolbox
from TransformerMTGP.selection import *
import sys
from TransformerMTGP import saveFile
import time
import random
import torch

import numpy as np
import job_creation
import agent_machine
import agent_workcenter
import sequencing
import routing

from TransformerMTGP.model.model import MyNN


class shopfloor:
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
            # self.job_creator = job_creation.creation \
            #     (self.env, self.span, self.m_list, self.wc_list, [5, 25], 2, 0.9, random_seed=True, ifPrint = self.ifPrint)
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
                        random_seed=True,
                        ifPrint=self.ifPrint,
                    )  # ifPrint = self.ifPrint is added by mengxu to make it clearer when using MTGP to train
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
                        random_seed=True,
                        ifPrint=self.ifPrint,
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
                        random_seed=True,
                        ifPrint=self.ifPrint,
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
                        random_seed=True,
                        ifPrint=self.ifPrint,
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

        # specify the architecture of DRL
        # if 'arch' and 'global_reward' in kwargs:
        #     arch = kwargs['arch'] + "=True"
        #     global_reward = 'global_reward={}'.format(kwargs['global_reward'])
        #     order = "self.routing_brain = validation_R.DRL_routing(self.env, self.job_creator, self.wc_list, {},{})".format(arch,global_reward)
        #     exec(order)

    def simulation(self):
        self.env.run()


def connectedness(cluster):
    print(cluster)


def init_toolbox(toolbox, pset):
    REP.init_toolbox(toolbox, pset)
    toolbox.register("select", selElitistAndTournament, tournsize=TOURNAMENT_SIZE, elitism=ELITISM)


def init_stats():
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def evaluate(individual, rd, seed):
    # add by mengxu 2022.10.13 to add the training instances ===============================================
    # create the environment instance for simulation
    if individual.num_calculation >= 20:
        return [(individual.fitness.values[0],), individual.num_calculation]
    dataset_name = rd["dataset_name"]
    # create the shop floor instance
    rule_R = "GP_evolve_R"
    rule_S = "GP_evolve_S"
    true_fitness = individual.fitness.values[0] if individual.fitness.valid else 0
    for i in range(rd.get("num_iteration", 50)):
        seed += 1000
        env = simpy.Environment()
        spf = shopfloor(
            env,
            span,
            m_no,
            wc_no,
            individual[0],
            individual[1],
            routing_rule=rule_R,
            sequencing_rule=rule_S,
            seed=seed,
            ifPrint=False,
            dataset_name=dataset_name,
        )
        spf.simulation()
        output_time, cumulative_tard, tard_mean, tard_max, tard_rate = (
            spf.job_creator.tardiness_output()
        )
        true_fitness = true_fitness + 1 / (individual.num_calculation + 1) * (
            cumulative_tard[-1] - true_fitness
        )
        individual.num_calculation += 1
    return [(true_fitness,), individual.num_calculation]


def eval_wrapper(*args, **kwargs):
    rd = kwargs["rd"]
    return evaluate(*args, **kwargs, seed=rd["seed"])
    # return evaluate(*args, **kwargs, toolbox=rd['toolbox'], seed = rd['seed'])
    # return evaluate(*args, **kwargs, toolbox=rd['toolbox'], data=rd['data'], labels=rd['labels'])


# copies data over from parent process
def init_data(rundata):
    global rd
    rd = rundata


def GPFC_main(dataset_name, seed, num_pre_selection):
    rd["use_niching"] = use_niching
    rd["seed"] = seed
    rd["dataset_name"] = dataset_name
    num_features = 0  # the initial number of terminals is 0, then I will add more terminals into the pset
    pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset.context["array"] = np.array
    REP.init_primitives(pset)
    weights = (-1.0,)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    # set up toolbox
    toolbox = ParallelToolbox()  # base.Toolbox()
    init_toolbox(toolbox, pset)
    toolbox.register("evaluate", eval_wrapper)

    rd["toolbox"] = toolbox
    rd["only_sequencing_rule"] = only_sequencing_rule
    pop = toolbox.population(n=POP_SIZE)
    stats = init_stats()
    hof = tools.HallOfFame(1)
    seedRotate = True  # added by mengxu 2022.10.13
    # seedRotate = False # added by mengxu 2022.10.13

    transformer_model = MyNN(64, 1024, 1, 8, 3)
    adam = torch.optim.Adam(transformer_model.parameters(), lr=1e-3)
    times = 1

    (
        pop,
        logbook,
        min_fitness,
        best_ind_all_gen,
        top_inds_fitness_final_gen,
        top_inds_final_gen,
    ) = ea_simple_elitism.eaSimple(
        pop,
        toolbox,
        CXPB,
        MUTPB,
        REPPB,
        ELITISM,
        NGEN,
        seedRotate,
        rd,
        stats,
        halloffame=hof,
        verbose=True,
        seed=seed,
        dataset_name=dataset_name,
        transformer_model=transformer_model,
        optimizer=adam,
        start_gen=times,
        num_pre_selection=num_pre_selection,
    )
    best = hof[0]

    return min_fitness, best, best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen


POP_SIZE = 50
NGEN = 50
CXPB = 0.8
MUTPB = 0.15
REPPB = 0.05
ELITISM = 10
TOURNAMENT_SIZE = 4
MAX_HEIGHT = 8  # 8
REP = mt  # individual representation {mt (multi-tree) or vt (vector-tree)}
REP.MAX_HEIGHT = MAX_HEIGHT
N_TREES = 2
# N_TREES = 1
REP.N_TREES = N_TREES
only_sequencing_rule = False
rd = {}
use_niching = False

# create the shop floor instance
span = 1000
m_no = 6
wc_no = 3
ins_each_gen = 1  # added by mengxu followed the advice of Meng 2022.11.01


def main(dataset_name, seed, num_pre_selection):
    # if __name__ == "__main__":
    #     dataset_name = str(sys.argv[1])
    #     seed = int(sys.argv[2])
    random.seed(int(seed))
    np.random.seed(int(seed))
    saveFile.clear_individual_each_gen_to_txt(seed, dataset_name)
    start = time.time()
    min_fitness, p_one, best_ind_all_gen, top_inds_fitness_final_gen, top_inds_final_gen, = GPFC_main(dataset_name, seed, num_pre_selection)
    end = time.time()
    running_time = end - start
    saveFile.save_each_gen_best_individual_meng(seed, dataset_name, best_ind_all_gen)
    saveFile.save_each_gen_best_individual_json_format(
        seed, dataset_name, best_ind_all_gen
    )
    saveFile.saveMinFitness(seed, dataset_name, min_fitness)
    saveFile.saveRunningTime(seed, dataset_name, running_time)
    saveFile.save_top_inds_final_gen_meng(seed, dataset_name, top_inds_final_gen)
    saveFile.save_top_inds_fitness_final_gen(
        seed, dataset_name, top_inds_fitness_final_gen
    )
    saveFile.save_top_inds_with_fitness_final_gen_to_txt(
        seed, dataset_name, top_inds_final_gen, top_inds_fitness_final_gen
    )
    print(min_fitness)
    print("Training time: " + str(running_time))
    print("Training end!")
