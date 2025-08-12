import random

from deap import tools
import numpy as np
from util.functions import record


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

        # Step 5-7: Produce Offspring from population in intermediate population
        parents = toolbox.select(population, len(population))
        pop_intermediate = []
        while len(pop_intermediate) < len(population):
            offspring_intermediate = varAnd(parents, toolbox, cxpb, mutpb, reppb)
            pop_intermediate.extend(offspring_intermediate)
            del offspring_intermediate
        pop_intermediate[:] = pop_intermediate[: len(population)]
        # Step 5-7: Produce Offspring from population in intermediate population

        # Replace the current population by the offspring
        population[:] = pop_intermediate

    return population, logbook, min_fitness, best_ind_all_gen, all_individuals
