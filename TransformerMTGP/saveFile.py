import pickle
import numpy as np
import json


def save_all_individuals(
    randomSeeds, dataSetName, individuals, gen, with_fitness=False
):
    try:
        with open(
            "./TransformerMTGP/train/scenario_"
            + str(dataSetName)
            + "/"
            + str(randomSeeds)
            + "_all_individual_"
            + dataSetName
            + "_formula_format"
            + ".json",
            "r",
        ) as file:
            individuals_list = json.load(file)
    except FileNotFoundError:
        individuals_list = []
        with open(
            "./TransformerMTGP/train/scenario_"
            + str(dataSetName)
            + "/"
            + str(randomSeeds)
            + "_all_individual_"
            + dataSetName
            + "_formula_format"
            + ".json",
            "w",
        ) as file:
            json.dump(individuals_list, file)
    if not with_fitness:
        for key, ind in enumerate(individuals):
            individual_dict = {
                "T0": str(ind[0]),
                "T1": str(ind[1]),
                "fitness": ind.fitness.values[0],
                "num_calculation": ind.num_calculation,
                "score": getattr(ind, "score", 0),
                "gen": gen,
            }
            individuals_list.append(individual_dict)
    else:
        individuals_list = individuals
    with open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_all_individual_"
        + dataSetName
        + "_formula_format"
        + ".json",
        "w",
    ) as file:
        json.dump(individuals_list, file)


def save_all_intermedia_individuals(
    randomSeeds, dataSetName, individuals, gen, with_fitness=False
):
    try:
        with open(
            "./TransformerMTGP/train/scenario_"
            + str(dataSetName)
            + "/"
            + str(randomSeeds)
            + "_all_inter_pop_"
            + dataSetName
            + "_formula_format"
            + ".json",
            "r",
        ) as file:
            individuals_list = json.load(file)
    except FileNotFoundError:
        individuals_list = []
        with open(
            "./TransformerMTGP/train/scenario_"
            + str(dataSetName)
            + "/"
            + str(randomSeeds)
            + "_all_inter_pop_"
            + dataSetName
            + "_formula_format"
            + ".json",
            "w",
        ) as file:
            json.dump(individuals_list, file)
    if not with_fitness:
        for key, ind in enumerate(individuals):
            individual_dict = {
                "T0": str(ind[0]),
                "T1": str(ind[1]),
                "fitness": ind.fitness.values[0],
                "num_calculation": ind.num_calculation,
                "score:": ind.score,
                "gen": gen,
            }
            individuals_list.append(individual_dict)
    else:
        individuals_list = individuals
    with open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_all_inter_pop_"
        + dataSetName
        + "_formula_format"
        + ".json",
        "w",
    ) as file:
        json.dump(individuals_list, file)


def save_each_gen_best_individual_json_format(
    randomSeeds, dataSetName, best_ind_all_gen
):
    individual_dict = []

    for key, ind in enumerate(best_ind_all_gen):
        individual_dict.append(
            {
                "T0": str(ind[0]),
                "T1": str(ind[1]),
                "fitness": 0,
            }
        )

    with open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_meng_individual_"
        + dataSetName
        + "_formula_format"
        + ".json",
        "w",
    ) as fileName_individual:
        json.dump(individual_dict, fileName_individual)

    return


def save_each_gen_best_individual_on_test_dataset(
    randomSeeds, dataSetName, best_ind_all_gen_dict
):
    with open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_meng_individual_"
        + dataSetName
        + "_formula_format"
        + ".json",
        "w",
    ) as fileName_individual:
        json.dump(best_ind_all_gen_dict, fileName_individual)


def save_each_gen_best_individual_meng(randomSeeds, dataSetName, best_ind_all_gen):
    individual_dict = []

    for gen in range(len(best_ind_all_gen)):
        best_ind = best_ind_all_gen[gen]

        if len(best_ind) == 2:
            sequencing = best_ind[0]
            routing = best_ind[1]
        else:
            sequencing = best_ind[0]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        if len(best_ind) == 2:
            routing_list = []
            for i in range(len(routing)):
                routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        if len(best_ind) == 2:
            individual.append(routing_list)

        # individual_dict.__setitem__(gen, individual)
        individual_dict.append(individual)

    with open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_meng_individual_"
        + dataSetName
        + ".json",
        "w",
    ) as fileName_individual:
        json.dump(individual_dict, fileName_individual)


def save_top_inds_final_gen_meng(randomSeeds, dataSetName, top_inds_fitness_final_gen):
    individual_dict = []

    for gen in range(len(top_inds_fitness_final_gen)):
        best_ind = top_inds_fitness_final_gen[gen]

        if len(best_ind) == 2:
            sequencing = best_ind[0]
            routing = best_ind[1]
        else:
            sequencing = best_ind[0]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        if len(best_ind) == 2:
            routing_list = []
            for i in range(len(routing)):
                routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        if len(best_ind) == 2:
            individual.append(routing_list)

        # individual_dict.__setitem__(gen, individual)
        individual_dict.append(individual)

    # fileName_individual = open('./MTGP/train/' + str(randomSeeds) + '_meng_individual_' + dataSetName + '.pkl', "wb")
    # pickle.dump(individual_dict, fileName_individual)
    with open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_meng_top_individuals_final_gen_"
        + dataSetName
        + ".pkl",
        "wb",
    ) as fileName_individual:
        pickle.dump(individual_dict, fileName_individual)

    return


def save_individual_to_txt(
    randomSeeds, dataSetName, individuals
):  # save individual as txt by mengxu
    file = open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_"
        + dataSetName
        + ".txt",
        "w",
    )
    file.write("Individual:\n")
    file.write("Tree 0:\n")  # routing rule
    file.write(str(individuals[0]) + "\n")
    if len(individuals) == 2:
        file.write("Tree 1:\n")  # sequencing rule
        file.write(str(individuals[1]) + "\n")

    file.close()
    return


def clear_individual_each_gen_to_txt(
    randomSeeds, dataSetName
):  # save individual as txt by mengxu
    file = open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_"
        + dataSetName
        + "_each_gen.txt",
        "w",
    )  # 'w' represent coverage, 'a' denotes not coverage
    file.write("Best individuals from each gen:\n")
    file.close()
    return


def save_individual_each_gen_to_txt(
    randomSeeds, dataSetName, individuals, gen
):  # save individual as txt by mengxu
    file = open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_"
        + dataSetName
        + "_each_gen.txt",
        "a",
    )  # 'w' represent coverage, 'a' denotes not coverage
    file.write("\nGen: " + str(gen) + "\n")
    file.write("Individual:\n")
    file.write("Tree 0:\n")  # routing rule
    file.write(str(individuals[0]) + "\n")
    if len(individuals) == 2:
        file.write("Tree 1:\n")  # sequencing rule
        file.write(str(individuals[1]) + "\n")

    file.close()
    return


def save_top_inds_with_fitness_final_gen_to_txt(
    randomSeeds, dataSetName, individuals, fitnesses
):  # save individual as txt by mengxu
    file = open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_"
        + dataSetName
        + "_top_inds_with_fitness_final_gen.txt",
        "w",
    )  # 'w' represent coverage, 'a' denotes not coverage
    for i in range(len(individuals)):
        individual = individuals[i]
        file.write("Individual:" + str(i) + "\n")
        file.write("Tree 0:\n")  # routing rule
        file.write(str(individual[0]) + "\n")
        if len(individual) == 2:
            file.write("Tree 1:\n")  # sequencing rule
            file.write(str(individual[1]) + "\n")
        file.write("Fitness:\n")  # sequencing rule
        file.write(str(fitnesses[i]) + "\n")
        file.write("\n")

    file.close()
    return


def save_archive(randomSeeds, dataSetName, individuals):
    with open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_archive"
        + dataSetName
        + ".pickle",
        "wb",
    ) as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return


def save_pop(randomSeeds, dataSetName, individuals):
    with open(
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_pop"
        + dataSetName
        + ".pickle",
        "wb",
    ) as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return


def saveMinFitness(randomSeeds, dataSetName, min_fitness):
    fileName1 = (
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_min_fitness"
        + dataSetName
    )
    np.save(fileName1, min_fitness)
    return


def save_top_inds_fitness_final_gen(randomSeeds, dataSetName, min_fitness):
    fileName1 = (
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_top_inds_fitness_final_gen"
        + dataSetName
    )
    np.save(fileName1, min_fitness)
    return


def saveRunningTime(randomSeeds, dataSetName, running_time):
    fileName1 = (
        "./TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_running_time"
        + dataSetName
    )
    np.save(fileName1, running_time)
    return
