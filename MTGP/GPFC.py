import simpy
from deap import base, creator, gp, tools
import MTGP_KNN.multi_tree as REP
from MTGP import ea_simple_elitism
from util.ParallelToolbox import ParallelToolbox
import util.saveFile as saveFile
import time

from functools import partial
import numpy as np
import util.job_creation as job_creation
import util.agent_machine as agent_machine
import util.agent_workcenter as agent_workcenter
import util.sequencing as sequencing
import util.routing as routing
import util.multi_tree as mt
from util.selection import (
    selElitistAndTournament,
)


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

    def simulation(self):
        self.env.run()


def connectedness(cluster):
    print(cluster)


def init_toolbox(toolbox, pset, config):
    REP.init_toolbox(toolbox, pset)
    toolbox.register(
        "select",
        selElitistAndTournament,
        tournsize=config.TOURNAMENT_SIZE,
        elitism=config.ELITISM,
    )


def init_stats():
    fitness_stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats = tools.MultiStatistics(fitness=fitness_stats)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    return stats


def evaluate(individual, **kwargs):
    # create the environment instance for simulation
    config = kwargs["config"]
    env = simpy.Environment()
    dataset_name = config.scenarios
    # create the shop floor instance
    spf = shopfloor(
        env,
        config.span,
        config.m_no,
        config.wc_no,
        individual[0],
        individual[1],
        routing_rule="GP_evolve_R",
        sequencing_rule="GP_evolve_S",
        seed=config.seeds,
        ifPrint=False,
        dataset_name=dataset_name,
    )
    spf.simulation()
    output_time, cumulative_tard, tard_mean, tard_max, tard_rate = (
        spf.job_creator.tardiness_output()
    )
    fitness = cumulative_tard[-1]

    for i in range(config.ins_each_gen - 1):
        seed = config.seeds + 1000
        env = simpy.Environment()
        spf = shopfloor(
            env,
            config.span,
            config.m_no,
            config.wc_no,
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
        fitness = fitness + cumulative_tard[-1]

    fitness = fitness / config.ins_each_gen
    scores = [fitness]
    return scores


# copies data over from parent process
def init_data(rundata):
    global rd
    rd = rundata


def GPFC_main(config):
    rd = {}
    rd["seed"] = config.seeds
    rd["dataset_name"] = config.scenarios
    num_features = 0  # the initial number of terminals is 0, then I will add more terminals into the pset
    pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset.context["array"] = np.array
    mt.init_primitives(pset)
    weights = (-1.0,)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    # set up toolbox
    toolbox = ParallelToolbox()  # base.Toolbox()
    init_toolbox(toolbox, pset, config)

    partial_evaluate = partial(evaluate, config=config)
    toolbox.register("evaluate", partial_evaluate)

    rd["toolbox"] = toolbox
    pop = toolbox.population(n=config.POP_SIZE)
    stats = init_stats()
    hof = tools.HallOfFame(1)
    seedRotate = True
    pop, logbook, min_fitness, best_ind_all_gen, all_individuals = (
        ea_simple_elitism.eaSimple(
            pop,
            toolbox,
            config.CXPB,
            config.MUTPB,
            config.REPPB,
            config.ELITISM,
            config.NGEN,
            seedRotate,
            rd,
            stats,
            halloffame=hof,
            verbose=True,
            seed=config.seeds,
            dataset_name=config.scenarios,
            config=config,
        )
    )
    best = hof[0]
    return min_fitness, best, best_ind_all_gen, all_individuals


def main(config, *args):
    saveFile.clear_individual_each_gen_to_txt(config)
    start = time.time()
    min_fitness, p_one, best_ind_all_gen, all_individuals = GPFC_main(config)
    end = time.time()
    running_time = end - start
    saveFile.save_each_gen_best_individual_json_format(config, best_ind_all_gen)
    saveFile.save_each_gen_best_individual_meng(config, best_ind_all_gen)
    print(min_fitness)
    print("Training time: " + str(running_time))
    print("Training end!")
