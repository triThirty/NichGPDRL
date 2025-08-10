from util.sequencing import GP_evolve_S
from util.routing import GP_evolve_R
import numpy as np
from scipy.spatial.distance import pdist, squareform


def phyno_hash_individual(ind):
    return hash(str(ind.decision_vector))


def hash_individual(ind):
    return hash(str(ind[0]) + str(ind[1]))


def phyno_remove_duplicates(population):
    unique_pop = []
    seen = set()

    for i, ind in enumerate(population):
        h = phyno_hash_individual(ind)
        if h not in seen:
            seen.add(h)
            unique_pop.append(ind)

    return unique_pop


def remove_duplicates(population):
    unique_pop = []
    seen = set()

    for ind in population:
        h = hash_individual(ind)
        if h not in seen:
            seen.add(h)
            unique_pop.append(ind)

    return unique_pop


def calculate_ranking_accuracy(data):
    concordant_pairs = 0
    discordant_pairs = 0

    n = len(data)

    for i in range(n):
        for j in range(i + 1, n):
            item_i = data[i]
            item_j = data[j]

            fitness_diff = item_i.fitness.values[0] - item_j.fitness.values[0]
            score_diff = item_i.score - item_j.score

            if fitness_diff != 0 and fitness_diff * score_diff < 0:
                concordant_pairs += 1
            elif fitness_diff == 0 and score_diff == 0:
                concordant_pairs += 1
            else:
                discordant_pairs += 1
    accuracy = concordant_pairs / (concordant_pairs + discordant_pairs)
    return accuracy


def calculate_score_based_ind_proportion(
    sorted_pop_intermediate_by_fitness, sorted_pop_intermediate_by_score
):
    total_individuals = len(sorted_pop_intermediate_by_fitness)
    concordant_pairs = 0

    for ind in sorted_pop_intermediate_by_score:
        if ind in sorted_pop_intermediate_by_fitness:
            concordant_pairs += 1

    return concordant_pairs / total_individuals


def get_index_of_selected_inds_in_intermediate(
    sorted_pop_intermediate_by_fitness: list, selected_inds
):
    indices = []
    for ind in selected_inds:
        if ind in sorted_pop_intermediate_by_fitness:
            index = sorted_pop_intermediate_by_fitness.index(ind)
            indices.append(index)
    return indices


def compute_phenotype(pop, decision_situations):
    for ind in pop:
        decision_vector = []
        for situation in decision_situations:
            selected_machine_index = GP_evolve_R(ind[0], *situation[0])
            decision_vector.append(selected_machine_index)
            job_position = GP_evolve_S(situation[1], ind[1])
            decision_vector.append(job_position)
        ind.decision_vector = decision_vector


def phenotype_distance(offspring):
    data_matrix = []
    for ind in offspring:
        data_matrix.append(ind.decision_vector)
    data_matrix = np.array(data_matrix)
    distances_compressed = pdist(data_matrix, metric="euclidean")
    distance_matrix = squareform(distances_compressed)
    masked_matrix = distance_matrix + np.diag([np.inf] * distance_matrix.shape[0])
    min_indices = np.argmin(masked_matrix, axis=1)
    return min_indices
