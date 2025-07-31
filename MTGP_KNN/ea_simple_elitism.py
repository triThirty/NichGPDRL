from operator import attrgetter
import random
from deap import tools
import numpy as np
from MTGP_KNN import saveFile
from MTGP_KNN.selection import selElitistAndTournament
from MTGP_KNN.util.decistion_situation_generator import (
    compute_phenotype,
    KNN_train,
    generate_next_generation,
    remove_duplicates,
)

from memory_profiler import profile


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
            (offspring[i],) = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            i = i + 1
    return offspring


# todo: need to check if this is right
def sortPopulation(toolbox, population):
    populationCopy = [toolbox.clone(ind) for ind in population]
    popsize = len(population)

    for j in range(popsize):
        sign = False
        for i in range(popsize - 1 - j):
            sum_fit_i = np.sum(populationCopy[i].fitness.values)
            sum_fit_i_1 = np.sum(populationCopy[i + 1].fitness.values)
            if sum_fit_i > sum_fit_i_1:
                populationCopy[i], populationCopy[i + 1] = (
                    populationCopy[i + 1],
                    populationCopy[i],
                )
                sign = True
        if not sign:
            break

    # FOR CHECK
    # pop_fit = [np.sum(ind.fitness.values) for ind in
    #            populationCopy]
    # print(pop_fit)
    return populationCopy


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
):
    # initialise the random seed of each generation
    randomSeed_ngen = []
    for i in range((ngen + 1)):
        # for i in range((ngen+1)*ins_each_gen): # the *ins_each_gen is added by mengxu followed the advice of Meng 2022.11.01
        randomSeed_ngen.append(np.random.randint(2000000000))

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    min_fitness = []
    best_ind_all_gen = []  # add by mengxu
    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]

    rd["seed"] = randomSeed_ngen[0]
    fitnesses = toolbox.multiProcess(toolbox.evaluate, invalid_ind, rd)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    # for ind in population:
    compute_phenotype(population, rd["decision_situations"])

    decision_matrix = [ind.decision_vector for ind in population]
    fitness_matrix = [ind.fitness.values[0] for ind in population]
    knn_model = KNN_train(X=decision_matrix, y=fitness_matrix)
    del decision_matrix
    del fitness_matrix

    # all_individuals.append(invalid_ind)
    all_individuals = []

    pop_fit = [ind.fitness.values[0] for ind in population]
    min_fitness.append(min(pop_fit))
    # add by mengxu 2022.10.26
    best_index = np.argmin(pop_fit)
    best_ind_all_gen.append(population[best_index])  # add by mengxu
    p_one = population[best_index]
    saveFile.save_individual_each_gen_to_txt(seed, dataset_name, p_one, 0)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # population = sorted(population, key=lambda x: x.fitness.values[0])[
    #     : int(len(population) / 4)
    # ]
    # Begin the generational process
    for gen in range(1, ngen + 1):

        # Added by mengxu to do seed rotation
        if seedRotate:
            rd["seed"] = randomSeed_ngen[gen]
            # rd['seed'] = np.random.randint(2000000000)
        # Select the next generation individuals
        sorted_elite = sortPopulation(toolbox, population)[
            :elitism
        ]  # modified by mengxu 2022.10.29
        # sorted_elite = sorted(population, key=attrgetter("fitness"), reverse=True)[:elitism]

        if gen == 1:
            ELITISM = 10
            toolbox.register(
                "select", selElitistAndTournament, tournsize=7, elitism=ELITISM
            )

        offspring = toolbox.select(population, len(population) - elitism)

        pop_intermediate = []
        print("-------No OOM-----")
        while len(pop_intermediate) < len(population) * num_pre_selection:
            offspring_intermediate = varAnd(offspring, toolbox, cxpb, mutpb, reppb)
            compute_phenotype(offspring_intermediate, rd["decision_situations"])
            pop_intermediate.extend(offspring_intermediate)
            pop_intermediate = remove_duplicates(sorted_elite + pop_intermediate)[
                elitism:
            ]
            del offspring_intermediate
        pop_intermediate[:] = pop_intermediate[: len(population) * num_pre_selection]

        score_elite = generate_next_generation(
            pop_intermediate, population, elitism, toolbox, knn_model
        )
        del pop_intermediate
        population = sorted_elite + score_elite[: len(population) - elitism]

        fitnesses = toolbox.multiProcess(toolbox.evaluate, population, rd)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        decision_matrix = [ind.decision_vector for ind in population]
        fitness_matrix = [ind.fitness.values[0] for ind in population]
        knn_model = KNN_train(X=decision_matrix, y=fitness_matrix)
        del decision_matrix
        del fitness_matrix

        # modified by mengxu
        if halloffame is not None:
            halloffame.clear()  # add by mengxu
            halloffame.update(population)

        # add by mengxu 2022.10.26
        pop_fit = [ind.fitness.values[0] for ind in population]
        best_index = np.argmin(pop_fit)
        best_ind_all_gen.append(population[best_index])  # add by mengxu
        p_one = population[best_index]
        saveFile.save_individual_each_gen_to_txt(seed, dataset_name, p_one, gen)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        pop_fit = [ind.fitness.values[0] for ind in population]
        min_fitness.append(min(pop_fit))
    return population, logbook, min_fitness, best_ind_all_gen, all_individuals
