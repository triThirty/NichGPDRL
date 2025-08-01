# import pickle5 as pickle
import pickle
import numpy as np
import sys
import json


def load_individual_from_gen_json_format(randomSeeds, dataSetName):
    with open(
        sys.path[0]
        + "/TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_meng_individual_"
        + dataSetName
        + "_formula_format"
        + ".json",
        "r",
    ) as fileName_individual:
        dict = json.load(fileName_individual)

    return dict


def load_all_individuals_from_gen_json_format(randomSeeds, dataSetName):
    with open(
        sys.path[0]
        + "/TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_all_individual_"
        + dataSetName
        + "_formula_format"
        + ".json",
        "r",
    ) as fileName_individual:
        dict = json.load(fileName_individual)

    return dict


def load_individual_from_gen(randomSeeds, dataSetName):
    with open(
        sys.path[0]
        + "/TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_meng_individual_"
        + dataSetName
        + ".json",
        "r",
    ) as fileName_individual:
        dict = json.load(fileName_individual)
    return dict


def load_top_inds_from_final_gen(
    randomSeeds, dataSetName
):  # save individual as txt by mengxu
    # with open(
    #     sys.path[0]
    #     + "/TransformerMTGP/train/scenario_"
    #     + str(dataSetName)
    #     + "/"
    #     + str(randomSeeds)
    #     + "_meng_top_individuals_final_gen_"
    #     + dataSetName
    #     + ".pkl",
    #     "rb",
    # ) as fileName_individual:
    #     dict = pickle.load(fileName_individual)

    # # print(dict.items())
    # return dict
    with open(
        sys.path[0]
        + "/TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_meng_top_individuals_final_gen_"
        + dataSetName
        + ".json",
        "r",
    ) as fileName_individual:
        dict = json.load(fileName_individual)
    return dict


def load_training_time(randomSeeds, dataSetName):  # save individual as txt by mengxu
    folder = (
        "/TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_running_time"
        + dataSetName
        + ".npy"
    )
    training_time = np.load(folder)

    return training_time


def load_min_fitness(randomSeeds, dataSetName):  # save individual as txt by mengxu
    folder = (
        "/TransformerMTGP/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_min_fitness"
        + dataSetName
        + ".npy"
    )
    min_fitness = np.load(folder)

    return min_fitness
