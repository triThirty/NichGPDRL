import numpy as np
from deap import tools
import torch
from sklearn.neighbors import KNeighborsRegressor


from TransformerMTGP.model.surrogate import (
    surrogate_evaluate,
    new_surrogate_train,
)
from model.model import MyNN, SharedEmbeddings

from util.deplicate_removal import (
    phyno_remove_duplicates,
    compute_phenotype,
    remove_duplicates,
    remove_duplicates_from_list_a,
)
from util.functions import record
from util.statistics import statistics
from TransformerMTGP.util.functions import varAnd
from MTGP_KNN.util.decistion_situation_generator import (
    KNN_train,
    predict,
)


def eaSimple(
    population,
    toolbox,
    cxpb,
    mutpb,
    reppb,
    elitism,
    ngen,
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

    logbook = tools.Logbook()
    logbook.header = ["gen", "nevals"] + (stats.fields if stats else [])
    KNN_model = KNeighborsRegressor(
        n_neighbors=config.n_neighbors, p=2, weights="distance"
    )
    min_fitness = []
    best_ind_all_gen = []
    accuracy_trend = []
    proportion_trend = []
    shared_emb = SharedEmbeddings()

    training_dataset = []
    validation_dataset = []
    decision_matrix = []
    fitness_matrix = []

    # Begin the generational process
    for gen in range(start_gen, ngen + 1):

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

        # Step 4: Update Surrogate Model
        decision_matrix.extend([ind.decision_vector for ind in population])
        fitness_matrix.extend([ind.fitness.values[0] for ind in population])
        knn_model = KNN_train(X=decision_matrix, y=fitness_matrix, KNN_model=KNN_model)

        # Step 4.1: Initialize Transformer Model for each generation
        transformer_model = MyNN(64, 1024, 1, 8, 3, shared_emb)
        optimizer = torch.optim.Adam(
            params=list(transformer_model.parameters()),
            lr=1e-3,
        )
        # Step 4.1: Initialize Transformer Model for each generation
        intermediate_dataset = []
        for ind in population:
            if 20 <= len(ind[0]) + len(ind[1]) <= 150:
                intermediate_dataset.append(ind)
        intermediate_dataset = remove_duplicates_from_list_a(
            intermediate_dataset, training_dataset + validation_dataset
        )
        np.random.shuffle(intermediate_dataset)
        training_dataset.extend(
            intermediate_dataset[: int(len(intermediate_dataset) * 0.8)]
        )
        validation_dataset.extend(
            intermediate_dataset[int(len(intermediate_dataset) * 0.8) :]
        )
        training_dataset = remove_duplicates(training_dataset)
        validation_dataset = remove_duplicates(validation_dataset)
        new_surrogate_train(
            training_dataset,
            validation_dataset,
            transformer_model,
            optimizer,
            device,
        )
        surrogate_evaluate(population, transformer_model, device)
        # Step 4: Update Surrogate Model

        # Step 5-7: Produce Offspring from population in intermediate population
        parents = toolbox.select(population, len(population))  # Select parents
        elitism_pop = tools.selBest(population, elitism)  # Select elitism
        pop_intermediate = []
        while len(pop_intermediate) < len(population) * num_pre_selection:
            offspring_intermediate = varAnd(
                parents, toolbox, cxpb, mutpb, reppb, config
            )
            compute_phenotype(offspring_intermediate, rd["decision_situations"])
            pop_intermediate.extend(offspring_intermediate)
            pop_intermediate = phyno_remove_duplicates(pop_intermediate)
            del offspring_intermediate
        pop_intermediate[:] = pop_intermediate[: len(population) * num_pre_selection]
        # Step 5-7: Produce Offspring from population in intermediate population

        # Step 8: Estimate Fitness using Surrogate
        surrogate_evaluate(pop_intermediate, transformer_model, device)
        predict(knn_model, pop_intermediate)
        del transformer_model
        # Step 8: Estimate Fitness using Surrogate

        # Step 9: Fill P with Best Rules from intermediate population
        population = (
            elitism_pop
            # + sorted(pop_intermediate, key=lambda x: x.score)[
            + sorted(pop_intermediate, key=lambda x: x.fitness.values[0])[
                : len(population) - elitism
            ]
        )
        # Step 9: Fill P with Best Rules from intermediate population

        # Statistics
        # statistics(toolbox, pop_intermediate, config, population, proportion_trend)
        # Statistics
        del elitism_pop
        del pop_intermediate

    return (
        population,
        logbook,
        min_fitness,
        best_ind_all_gen,
        accuracy_trend,
        proportion_trend,
    )
