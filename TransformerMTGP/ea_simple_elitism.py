import random

import numpy as np
from deap import tools


from TransformerMTGP import saveFile
from TransformerMTGP.selection import selElitistAndTournament
from TransformerMTGP.niching.niching import niching_clear
from TransformerMTGP.model.surrogate import surrogate_train


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
        else:  # reproduction
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
    transformer_model=None,
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
    saveFile.save_all_individuals(seed, dataset_name, invalid_ind)

    rd["seed"] = randomSeed_ngen[0]
    fitnesses = toolbox.multiProcess(toolbox.evaluate, invalid_ind, rd)
    # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    surrogate_train(invalid_ind, transformer_model)
    transformer_model.eval()

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

    # add by mengxu for niching 2023.10.18
    if rd["use_niching"]:
        nich = niching_clear(0, 1)
        nich.initial_phenoCharacterisation(population[best_index])
        population = nich.clearPopulation(toolbox, population)

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

        # if gen == 1:
        # ELITISM = 10
        # toolbox.register("select", selElitistAndTournament, tournsize=TOURNAMENT_SIZE, elitism=ELITISM)

        offspring = toolbox.select(population, len(population) - elitism)

        # Vary the pool of individuals
        # print('ori',offspring[0][0])
        # print('ori',offspring[0][1])
        # print('ori',offspring[0][2])
        offspring = varAnd(offspring, toolbox, cxpb, mutpb, reppb)
        # print('after',offspring[0][0])
        # print('after',offspring[0][1])
        # print('after',offspring[0][2])
        # exit()
        saveFile.save_all_individuals(seed, dataset_name, offspring)

        # Evaluate the sorted_elite with an invalid fitness as we rotate seed, add by mengxu
        invalid_elite_ind = [ind for ind in sorted_elite]
        # invalid_elite_ind = sorted_elite #modified by mengxu, as we rotate seed, no matter it is valid or not valid, we need to re-evaluate
        for ind in invalid_elite_ind:
            del ind.fitness.values
        fitnesses_elite = toolbox.multiProcess(toolbox.evaluate, invalid_elite_ind, rd)
        # fitnesses_elite = toolbox.map(toolbox.evaluate, invalid_elite_ind)
        for ind, fit in zip(invalid_elite_ind, fitnesses_elite):
            ind.fitness.values = fit

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring]
        # invalid_ind = offspring #modified by mengxu, as we rotate seed, no matter it is valid or not valid, we need to re-evaluate
        for ind in invalid_ind:
            del ind.fitness.values
        fitnesses = toolbox.multiProcess(toolbox.evaluate, invalid_ind, rd)
        # fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Replace the current population by the offspring
        population[:] = invalid_elite_ind + invalid_ind
        # population[:] = sorted_elite+offspring

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

        # add by mengxu 2023.10.18 for niching---------------------------
        if rd["use_niching"]:
            nich.calculate_phenoCharacterisation(population[best_index])
            population = nich.clearPopulation(toolbox, population)
        # add by mengxu 2023.10.18 for niching---------------------------

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        pop_fit = [
            ind.fitness.values[0] for ind in population
        ]  ######selection from author
        min_fitness.append(min(pop_fit))

        if gen == ngen:
            # output top 5 individuals by niching GP 2023.10.19
            sorted_elite = sortPopulation(toolbox, population)
            top_inds_final_gen = []
            top_inds_fitness_final_gen = []
            for i in range(10):
                top_inds_final_gen.append(sorted_elite[i])
                top_inds_fitness_final_gen.append(sorted_elite[i].fitness.values[0])

    return (
        population,
        logbook,
        min_fitness,
        best_ind_all_gen,
        top_inds_fitness_final_gen,
        top_inds_final_gen,
    )
