import simpy
from deap import base, creator, gp, tools
import MTGP_KNN.multi_tree as REP
from MTGP_KNN import ea_simple_elitism
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
from util.shopfloor import knn_shopfloor, evaluate


from util.deplicate_removal import (
    compute_phenotype,
    phyno_remove_duplicates,
)


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
    rd = {}
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

    reference_sequencing_rule = gp.PrimitiveTree.from_string(
        config[config.scenarios].reference_sequencing_rule, pset=pset
    )
    reference_routing_rule = gp.PrimitiveTree.from_string(
        config[config.scenarios].reference_routing_rule, pset=pset
    )
    stats = init_stats()
    hof = tools.HallOfFame(1)
    rd["decision_situations"] = []
    env = simpy.Environment()
    spf = knn_shopfloor(
        env,
        2000,
        12,
        config.wc_no,
        reference_sequencing_rule,
        reference_routing_rule,
        routing_rule="GP_evolve_R",
        sequencing_rule="GP_evolve_S",
        seed=np.random.randint(0, 1000000),
        ifPrint=False,
        dataset_name=config.scenarios,
    )
    spf.simulation()

    for routing_data, sequencing_data in zip(
        spf.decision_situations["routing"][-45::3],
        spf.decision_situations["sequencing"][-45::3],
    ):
        decision_situation = (routing_data, sequencing_data)
        rd["decision_situations"].append(decision_situation)

    pop = toolbox.population(n=config.POP_SIZE)
    compute_phenotype(pop, rd["decision_situations"])
    pop = phyno_remove_duplicates(pop)
    while len(pop) < config.POP_SIZE:
        new_ind = toolbox.individual()
        compute_phenotype([new_ind], rd["decision_situations"])
        pop.append(new_ind)
        pop = phyno_remove_duplicates(pop)

    pop, logbook, min_fitness, best_ind_all_gen, proportion_trend = (
        ea_simple_elitism.eaSimple(
            pop,
            toolbox,
            config.CXPB,
            config.MUTPB,
            config.REPPB,
            config.ELITISM,
            config.NGEN,
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
    return min_fitness, best, best_ind_all_gen, proportion_trend


def main(config):
    saveFile.clear_index_of_selected_inds_in_intermediate(config)
    saveFile.clear_individual_each_gen_to_txt(config)
    start = time.time()
    min_fitness, p_one, best_ind_all_gen, proportion_trend = GPFC_main(config)
    end = time.time()
    running_time = end - start
    saveFile.save_each_gen_best_individual_json_format(config, best_ind_all_gen)
    saveFile.save_each_gen_best_individual_meng(config, best_ind_all_gen)
    saveFile.save_surrogate_proportion_trend(config, proportion_trend)
    print(min_fitness)
    print("Training time: " + str(running_time))
    print("Training end!")
