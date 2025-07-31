import random

import numpy as np
from deap import tools
import torch
from copy import deepcopy


import TransformerMTGP.saveFile as saveFile
from TransformerMTGP.model.surrogate import (
    surrogate_evaluate,
    new_surrogate_train,
)
from model.model import MyNN, SharedEmbeddings

from MTGP_KNN.util.decistion_situation_generator import compute_phenotype

from TransformerMTGP.util.functions import (
    remove_duplicates,
    phyno_remove_duplicates,
)


def varAnd(population, toolbox, cxpb, mutpb, reppb, transformer_model, device):
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
            surrogate_evaluate([offspring[i]], transformer_model, device)
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
    start_gen=1,
    num_pre_selection=0,
    device="cuda",
):
    # initialise the random seed of each generation
    randomSeed_ngen = []
    for i in range((ngen + 1)):
        randomSeed_ngen.append(np.random.randint(2000000000))

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    min_fitness = []
    best_ind_all_gen = []  # add by mengxu

    rd["seed"] = randomSeed_ngen[0]
    fitnesses = toolbox.multiProcess(toolbox.evaluate, population, rd)
    for ind, fit in zip(population, fitnesses):
        ind.fitness.values = fit

    compute_phenotype(population, rd["decision_situations"])

    shared_emb = SharedEmbeddings()
    transformer_model = MyNN(64, 1024, 1, 8, 3, shared_emb)
    optimizer = torch.optim.Adam(
        params=list(transformer_model.parameters()),
        lr=1e-3,
    )

    new_surrogate_train(
        # surrogate_train(
        population,
        transformer_model,
        optimizer,
        device=device,
    )
    surrogate_evaluate(population, transformer_model, device)
    saveFile.save_all_individuals(seed, dataset_name, population, 0)

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
    for gen in range(start_gen, ngen + 1):

        # Added by mengxu to do seed rotation
        if seedRotate:
            rd["seed"] = randomSeed_ngen[gen]
        sorted_elite = sorted(population, key=lambda x: x.fitness.values[0])[:elitism]

        offspring = toolbox.select(population, len(population) - elitism)

        pop_intermediate = []
        while len(pop_intermediate) < len(population) * num_pre_selection:
            offspring_intermediate = varAnd(
                offspring, toolbox, cxpb, mutpb, reppb, transformer_model, device
            )
            compute_phenotype(offspring_intermediate, rd["decision_situations"])
            pop_intermediate.extend(offspring_intermediate)
            pop_intermediate = remove_duplicates(
                sorted_elite + deepcopy(pop_intermediate)
            )[elitism:]
            pop_intermediate = phyno_remove_duplicates(
                sorted_elite + deepcopy(pop_intermediate)
            )[elitism:]
        pop_intermediate[:] = pop_intermediate[: len(population) * num_pre_selection]

        surrogate_evaluate(pop_intermediate, transformer_model, device)
        score_elite = []
        while len(score_elite) < len(population) - elitism:
            score_elite.extend(
                toolbox.score_base_select(pop_intermediate, len(population) - elitism)
            )

            score_elite[:] = remove_duplicates(score_elite)
        del pop_intermediate
        population = sorted_elite + score_elite[: len(population) - elitism]

        rd["seed"] = randomSeed_ngen[gen]
        surrogate_evaluate(population, transformer_model, device)
        fitnesses = toolbox.multiProcess(toolbox.evaluate, population, rd)
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        transformer_model = MyNN(64, 1024, 1, 8, 3, shared_emb)
        optimizer = torch.optim.Adam(
            params=list(transformer_model.parameters()),
            lr=1e-3,
        )
        new_surrogate_train(
            population,
            transformer_model,
            optimizer,
            device=device,
        )
        surrogate_evaluate(population, transformer_model, device)
        saveFile.save_all_individuals(seed, dataset_name, population, gen)

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

    return population, logbook, min_fitness, best_ind_all_gen, [], []
