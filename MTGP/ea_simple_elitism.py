import random

from deap import tools
import numpy as np
from util.functions import record
from TransformerMTGP.util.functions import varAnd
from TransformerMTGP.model.model import MyNN, SharedEmbeddings
import torch
from TransformerMTGP.model.surrogate import surrogate_evaluate

from util.deplicate_removal import remove_duplicates


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    reppb,
    elitism,
    ngen,
    stats=None,
    halloffame=None,
    verbose=__debug__,
    seed=__debug__,
    dataset_name=__debug__,
    config=None,
):
    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    min_fitness = []
    best_ind_all_gen = []
    all_individuals = []

    loaded_checkpoint = torch.load(f"data/checkpoint_{config.seeds}.pth")
    shared_emb = SharedEmbeddings()
    shared_emb.load_state_dict(loaded_checkpoint["embedding_state_dict"])
    transformer_model = MyNN(64, 1024, 1, 8, 3, shared_emb)
    transformer_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    transformer_model.eval()

    # Begin the generational process
    for gen in range(1, ngen + 1):
        # Step 3: Full Fitness Evaluation
        fitnesses = toolbox.multiProcess(toolbox.evaluate, population, config)
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
        surrogate_evaluate(population, transformer_model, "cpu")

        # Step 5-7: Produce Offspring from population in intermediate population
        parents = toolbox.select(population, len(population))  # Select parents
        elitism_pop = tools.selBest(population, elitism)  # Select elitism
        pop_intermediate = []
        while len(pop_intermediate) < len(population):
            offspring_intermediate = varAnd(
                parents, toolbox, cxpb, mutpb, reppb, config
            )
            pop_intermediate.extend(offspring_intermediate)
            pop_intermediate = remove_duplicates(pop_intermediate)
            del offspring_intermediate
        pop_intermediate[:] = pop_intermediate[: len(population) - elitism]
        # Step 5-7: Produce Offspring from population in intermediate population

        # Replace the current population by the offspring
        population[:] = elitism_pop + pop_intermediate

    return population, logbook, min_fitness, best_ind_all_gen, all_individuals
