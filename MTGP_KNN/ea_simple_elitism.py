import random

from deap import tools
import numpy as np

from MTGP_KNN.util.decistion_situation_generator import (
    KNN_train,
    predict,
)
from util.deplicate_removal import (
    compute_phenotype,
    phyno_remove_duplicates,
)
from util.functions import record
from util.statistics import statistics


def varAnd(population, toolbox, cxpb, mutpb, reppb):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb / (cxpb + mutpb + reppb)
    new_mutpb = mutpb / (cxpb + mutpb + reppb) + new_cxpb
    i = 1
    while i < len(offspring):
        randomValue = random.random()
        if randomValue < new_cxpb:  # crossover
            if offspring[i - 1] == offspring[i]:
                (offspring[i - 1],) = toolbox.mutate(offspring[i - 1])
                (offspring[i],) = toolbox.mutate(offspring[i])
            else:
                offspring[i - 1], offspring[i] = toolbox.mate(
                    offspring[i - 1], offspring[i]
                )
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
            i = i + 2
        elif new_cxpb <= randomValue < new_mutpb:  # mutation
            (offspring[i - 1],) = toolbox.mutate(offspring[i - 1])
            del offspring[i - 1].fitness.values
            i = i + 1
        else:
            i = i + 1
    return offspring


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    reppb,
    elitism,
    ngen,
    seedRotate,
    rd,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    seed=__debug__,
    dataset_name=__debug__,
    num_pre_selection=3,
    config=None,
):
    # initialise the random seed of each generation
    randomSeed_ngen = []
    for i in range((ngen + 1)):
        randomSeed_ngen.append(np.random.randint(2000000000))

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    min_fitness = []
    best_ind_all_gen = []
    all_individuals = []
    proportion_trend = []

    # Begin the generational process
    for gen in range(1, ngen + 1):
        if seedRotate:
            rd["seed"] = randomSeed_ngen[gen]

        # Step 3: Full Fitness Evaluation
        fitnesses = toolbox.multiProcess(
            toolbox.evaluate, population, config, rd["seed"]
        )
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit
        # Step 3: Full Fitness Evaluation

        record(
            halloffame,
            population,
            gen,
            stats,
            logbook,
            verbose,
            config,
            min_fitness,
            best_ind_all_gen,
        )

        # Step 4: Update Surrogate Model
        decision_matrix = [ind.decision_vector for ind in population]
        fitness_matrix = [ind.fitness.values[0] for ind in population]
        knn_model = KNN_train(
            X=decision_matrix, y=fitness_matrix, n_neighbors=config.n_neighbors
        )
        del decision_matrix
        del fitness_matrix
        # Step 4: Update Surrogate Model

        # Step 5-7: Produce Offspring from population in intermediate population
        parents = toolbox.select(population, len(population))
        pop_intermediate = []
        while len(pop_intermediate) < len(population) * num_pre_selection:
            offspring_intermediate = varAnd(parents, toolbox, cxpb, mutpb, reppb)
            compute_phenotype(offspring_intermediate, rd["decision_situations"])
            pop_intermediate.extend(offspring_intermediate)
            pop_intermediate = phyno_remove_duplicates(pop_intermediate)
            del offspring_intermediate
        pop_intermediate[:] = pop_intermediate[: len(population) * num_pre_selection]
        # Step 5-7: Produce Offspring from population in intermediate population

        # Step 8: Estimate Fitness using Surrogate
        predict(knn_model, pop_intermediate)
        # Step 8: Estimate Fitness using Surrogate

        # Step 9: Fill P with Best Rules from intermediate population
        population = sorted(pop_intermediate, key=lambda x: x.fitness.values[0])[
            : len(population)
        ]
        # Step 9: Fill P with Best Rules from intermediate population

        # Statistics
        statistics(toolbox, pop_intermediate, rd, config, population, proportion_trend)
        # Statistics
        del pop_intermediate

    return population, logbook, min_fitness, best_ind_all_gen, proportion_trend
