import random

import numpy as np
from deap import tools


from TransformerMTGP import saveFile
from TransformerMTGP.model.surrogate import surrogate_train, surrogate_evaluate


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
            offspring[i - 1].num_calculation = 0
            offspring[i].num_calculation = 0
            i = i + 2
        elif new_cxpb <= randomValue < new_mutpb:  # mutation
            (offspring[i],) = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values
            offspring[i].num_calculation = 0
            i = i + 1
        else:  # reproduction
            del offspring[i].fitness.values
            offspring[i].num_calculation = 0
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


ind_archive_list = []
# ind_archive_weights_list = []


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
    optimizer=None,
    start_gen=1,
    num_pre_selection=0,
    device="cuda",
    reduce_scheduler=None,
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
    # invalid_ind = [ind for ind in population if not ind.fitness.valid]
    # invalid_ind = population

    rd["seed"] = randomSeed_ngen[0]
    rd["num_iteration"] = 1
    fitnesses = toolbox.multiProcess(toolbox.evaluate, population, rd)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit[0]
        ind.num_calculation = fit[1]

    # saveFile.save_all_individuals(seed, dataset_name, invalid_ind)
    surrogate_train(
        population,
        transformer_model,
        optimizer,
        toolbox=toolbox,
        rd=rd,
        device=device,
        reduce_scheduler=reduce_scheduler,
    )
    surrogate_evaluate(population, transformer_model, device)

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
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    ind_archive_list.extend(population)
    for gen in range(start_gen, ngen + 1):

        # Added by mengxu to do seed rotation
        if seedRotate:
            rd["seed"] = randomSeed_ngen[gen]
        sorted_elite = sorted(population, key=lambda x: x.score, reverse=True)[:elitism]

        offspring = toolbox.select(population, len(population) - elitism)

        # pre_selection_list = []
        # for _ in range(num_pre_selection):
        #     pre_selection_list.extend(varAnd(offspring, toolbox, cxpb, mutpb, reppb))

        offspring = varAnd(offspring, toolbox, cxpb, mutpb, reppb)
        # offspring = pre_selection_list + offspring
        surrogate_evaluate(offspring, transformer_model, device)
        # if num_pre_selection > 0:
        #     offspring = toolbox.select(offspring, len(population) - elitism)
        ind_archive_list.extend(offspring)
        population[:] = offspring + sorted_elite

        rd["num_iteration"] = 1
        rd["seed"] = np.random.randint(2000000000)
        fitnesses = toolbox.multiProcess(toolbox.evaluate, population, rd)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit[0]
            ind.num_calculation = fit[1]
        training_data = []
        training_data[:] = population + random.choices(
            ind_archive_list[-300:],
            weights=[
                1 / individual.fitness.values[0]
                for individual in ind_archive_list[-300:]
            ],
            k=len(population),
        )
        training_data[:] = population
        surrogate_train(
            training_data,
            transformer_model,
            optimizer,
            toolbox=toolbox,
            rd=rd,
            device=device,
            reduce_scheduler=reduce_scheduler,
        )
        surrogate_evaluate(population, transformer_model, device)

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
