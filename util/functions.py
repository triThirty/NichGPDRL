import numpy as np
import util.saveFile as saveFile


def record(
    halloffame,
    population,
    gen,
    stats,
    logbook,
    verbose,
    config,
    min_fitness,
    best_ind_all_gen,
):
    if halloffame is not None:
        halloffame.clear()
        halloffame.update(population)

    pop_fit = [ind.fitness.values[0] for ind in population]
    best_index = np.argmin(pop_fit)
    best_ind_all_gen.append(population[best_index])
    p_one = population[best_index]
    saveFile.save_individual_each_gen_to_txt(config, p_one, gen)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=gen, nevals=len(population), **record)
    if verbose:
        print(logbook.stream)

    min_fitness.append(p_one.fitness.values[0])
