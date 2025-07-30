# import pickle5 as pickle
import pickle
import json
import numpy as np
import sys


def load_individual_from_gen_json_format(randomSeeds, dataSetName):
    with open(
        sys.path[0]
        + "/MTGP_KNN/train/scenario_"
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


def load_individual_from_gen(randomSeeds, dataSetName):
    with open(
        sys.path[0]
        + "/MTGP_KNN/train/scenario_"
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


def load_training_time(randomSeeds, dataSetName):  # save individual as txt by mengxu
    folder = (
        "/MTGP_KNN/train/scenario_"
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
        "/MTGP_KNN/train/scenario_"
        + str(dataSetName)
        + "/"
        + str(randomSeeds)
        + "_min_fitness"
        + dataSetName
        + ".npy"
    )
    min_fitness = np.load(folder)

    return min_fitness
