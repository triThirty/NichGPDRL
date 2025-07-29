import numpy as np
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor

# from MTGP_KNN.GPFC import evaluate, shopfloor
from NichGPDRL.util.sequencing import GP_evolve_S
from NichGPDRL.util.routing import GP_evolve_R


def compute_phenotype(pop, decision_situations):
    for ind in pop:
        decision_vector = []
        for situation in decision_situations:
            selected_machine_index = GP_evolve_R(ind[0], *situation[0])
            decision_vector.append(selected_machine_index)
            job_position = GP_evolve_S(situation[1], ind[1])
            decision_vector.append(job_position)
        ind.decision_vector = decision_vector


def KNN_train(X, y):
    # y = np.array(y, dtype="str")
    # y = [",".join(item) for item in y.astype(str)]
    KNN_model = KNeighborsClassifier(n_neighbors=3, p=2)
    KNN_model.fit(X, y)
    return KNN_model


def predict(model, pop):
    for ind in pop:
        predicted_fitness = model.predict([ind.decision_vector])
        ind.fitness.values = (predicted_fitness[0],)
