import simpy
from deap import base, creator, gp, tools

# import MTGP_KNN.multi_tree as REP
import util.multi_tree as REP
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
from util.selection import selElitistAndTournament
from util.shopfloor import evaluate


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


def GPFC_main(config):
    num_features = 0  # the initial number of terminals is 0, then I will add more terminals into the pset
    pset = gp.PrimitiveSet("MAIN", num_features, prefix="f")
    pset.context["array"] = np.array
    mt.init_primitives(pset)
    weights = (-1.0,)
    creator.create("FitnessMin", base.Fitness, weights=weights)
    # set up toolbox
    toolbox = ParallelToolbox()  # base.Toolbox()
    init_toolbox(toolbox, pset, config)

    toolbox.register("evaluate", evaluate)

    pop = toolbox.population(n=config.POP_SIZE)
    stats = init_stats()
    hof = tools.HallOfFame(1)
    pop, logbook, min_fitness, best_ind_all_gen, all_individuals = (
        ea_simple_elitism.eaSimple(
            pop,
            toolbox,
            config.CXPB,
            config.MUTPB,
            config.REPPB,
            config.ELITISM,
            config.NGEN,
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


def main(config):
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
