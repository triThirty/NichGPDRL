import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# from MTGP_KNN.GPFC import evaluate, shopfloor
from util.sequencing import GP_evolve_S
from util.routing import GP_evolve_R


def compute_phenotype(pop, decision_situations):
    for ind in pop:
        decision_vector = []
        for situation in decision_situations:
            selected_machine_index = GP_evolve_R(ind[0], *situation[0])
            decision_vector.append(selected_machine_index)
            job_position = GP_evolve_S(situation[1], ind[1])
            decision_vector.append(job_position)
        ind.decision_vector = decision_vector


def KNN_train(X, y, n_neighbors):
    KNN_model = KNeighborsRegressor(n_neighbors=n_neighbors, p=2, weights="distance")
    KNN_model.fit(X, y)
    return KNN_model


def predict(model, pop):
    for ind in pop:
        predicted_fitness = model.predict([ind.decision_vector])
        ind.fitness.values = (predicted_fitness[0],)


def hash_individual(ind):
    return hash(str(ind.decision_vector))


def remove_duplicates(population):
    unique_pop = []
    seen = set()

    for ind in population:
        h = hash_individual(ind)
        if h not in seen:
            seen.add(h)
            unique_pop.append(ind)

    return unique_pop


def generate_next_generation(pop_intermediate, population, elitism, toolbox, knn_model):
    predict(knn_model, pop_intermediate)
    score_elite = []
    while len(score_elite) < len(population) - elitism:
        score_elite.extend(toolbox.select(pop_intermediate, len(population) - elitism))

        score_elite[:] = remove_duplicates(score_elite)
    return score_elite
