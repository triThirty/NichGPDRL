import random
import time

import numpy as np
from deap import tools
import torch
from copy import deepcopy


import util.saveFile as saveFile
from TransformerMTGP.model.surrogate import (
    surrogate_evaluate,
    new_surrogate_train,
)
from model.model import MyNN, SharedEmbeddings

from MTGP_KNN.util.decistion_situation_generator import compute_phenotype

from TransformerMTGP.util.functions import (
    remove_duplicates,
    phyno_remove_duplicates,
    calculate_ranking_accuracy,
    calculate_score_based_ind_proportion,
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
            i = i + 2
        elif new_cxpb <= randomValue < new_mutpb:  # mutation
            (offspring[i - 1],) = toolbox.mutate(offspring[i - 1])
            del offspring[i - 1].fitness.values
            surrogate_evaluate([offspring[i - 1]], transformer_model, device)
            i = i + 1
        else:
            i += 1
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
    config=None,
):
    # initialise the random seed of each generation
    randomSeed_ngen = []
    for i in range((ngen + 1)):
        randomSeed_ngen.append(np.random.randint(2000000000))

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    min_fitness = []
    best_ind_all_gen = []  # add by mengxu
    accuracy_trend = []
    proportion_trend = []

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

    pop_fit = [ind.fitness.values[0] for ind in population]
    min_fitness.append(min(pop_fit))
    # add by mengxu 2022.10.26
    best_index = np.argmin(pop_fit)
    best_ind_all_gen.append(population[best_index])  # add by mengxu
    p_one = population[best_index]
    saveFile.save_individual_each_gen_to_txt(config, p_one, 0)

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    for gen in range(start_gen, ngen + 1):

        print("Starting time: 0")
        start_time = time.time()
        # Added by mengxu to do seed rotation
        if seedRotate:
            rd["seed"] = randomSeed_ngen[gen]
        sorted_elite = sorted(population, key=lambda x: x.fitness.values[0])[:elitism]

        offspring = toolbox.select(population, len(population) - elitism)

        pop_intermediate = []
        end_time = time.time()
        print("slow point 1, time cost: ", end_time - start_time)
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

        end_time = time.time()
        print("slow point 2, time cost: ", end_time - start_time)
        surrogate_evaluate(pop_intermediate, transformer_model, device)
        score_elite = []

        # Some statistics from here
        fitnesses = toolbox.multiProcess(toolbox.evaluate, pop_intermediate, rd)
        for ind, fit in zip(pop_intermediate, fitnesses):
            ind.fitness.values = fit

        sorted_pop_intermediate_by_fitness = sorted(
            pop_intermediate, key=lambda x: x.fitness.values[0], reverse=True
        )[: len(population) - elitism]

        # sorted_pop_intermediate_by_score = sorted(
        #     pop_intermediate, key=lambda x: x.score
        # )[: len(population) - elitism]

        accuracy = calculate_ranking_accuracy(pop_intermediate)
        print(f"Ranking accuracy: {accuracy:.4f}")
        accuracy_trend.append(accuracy)

        while len(score_elite) < len(population) - elitism:
            score_elite.extend(
                toolbox.score_base_select(pop_intermediate, len(population) - elitism)
            )

            score_elite[:] = remove_duplicates(score_elite)
        del pop_intermediate
        population = sorted_elite + score_elite[: len(population) - elitism]

        score_based_ind_proportion = calculate_score_based_ind_proportion(
            sorted_pop_intermediate_by_fitness,
            score_elite[: len(population) - elitism],
        )
        proportion_trend.append(score_based_ind_proportion)
        print(f"Score-based individual proportion: {score_based_ind_proportion:.4f}")
        # End of statistics

        rd["seed"] = randomSeed_ngen[gen]
        end_time = time.time()
        print("slow point 3, time cost: ", end_time - start_time)
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
        end_time = time.time()
        print("slow point 4, time cost: ", end_time - start_time)
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
        saveFile.save_individual_each_gen_to_txt(config, p_one, gen)

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print(logbook.stream)

        pop_fit = [ind.fitness.values[0] for ind in population]
        min_fitness.append(min(pop_fit))

    return (
        population,
        logbook,
        min_fitness,
        best_ind_all_gen,
        accuracy_trend,
        proportion_trend,
    )
