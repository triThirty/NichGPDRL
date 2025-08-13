from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from util.deplicate_removal import phyno_remove_duplicates


def KNN_train(X, y, KNN_model):
    # KNN_model = KNeighborsRegressor(n_neighbors=n_neighbors, p=2, weights="distance")
    KNN_model.fit(X, y)
    return KNN_model


def predict(model, pop):
    for ind in pop:
        predicted_fitness = model.predict([ind.decision_vector])
        ind.fitness.values = (predicted_fitness[0],)


def generate_next_generation(pop_intermediate, population, elitism, toolbox, knn_model):
    predict(knn_model, pop_intermediate)
    score_elite = []
    while len(score_elite) < len(population) - elitism:
        score_elite.extend(toolbox.select(pop_intermediate, len(population) - elitism))

        score_elite[:] = phyno_remove_duplicates(score_elite)
    return score_elite
