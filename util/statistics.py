from util.deplicate_removal import (
    calculate_score_based_ind_proportion,
    get_index_of_selected_inds_in_intermediate,
)
from util import saveFile


def statistics(toolbox, pop_intermediate, config, population, proportion_trend):
    fitnesses = toolbox.multiProcess(toolbox.evaluate, pop_intermediate, config)
    for ind, fit in zip(pop_intermediate, fitnesses):
        ind.fitness.values = fit

    sorted_pop_intermediate_by_fitness = sorted(
        pop_intermediate, key=lambda x: x.fitness.values[0]
    )
    indices = get_index_of_selected_inds_in_intermediate(
        sorted_pop_intermediate_by_fitness, population
    )
    saveFile.save_index_of_selected_inds_in_intermediate(config, indices)
    score_based_ind_proportion = calculate_score_based_ind_proportion(
        sorted_pop_intermediate_by_fitness[: len(population)],
        population,
    )
    proportion_trend.append(score_based_ind_proportion)
