import pickle
import numpy as np



def save_individual(randomSeeds, dataSetName,individuals):
    with open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_' + dataSetName+'.pickle', 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return

def save_each_gen_best_individual_meng(randomSeeds, dataSetName, best_ind_all_gen):
    individual_dict = {}

    for gen in range(len(best_ind_all_gen)):
        best_ind = best_ind_all_gen[gen]

        sequencing = best_ind[0]
        routing = best_ind[1]

        individual = []
        sequencing_list = []
        for i in range(len(sequencing)):
            sequencing_list.append(sequencing[i].name)

        routing_list = []
        for i in range(len(routing)):
            routing_list.append(routing[i].name)

        individual.append(sequencing_list)
        individual.append(routing_list)

        individual_dict.__setitem__(gen, individual)

    # fileName_individual = open('./MTGP/train/' + str(randomSeeds) + '_meng_individual_' + dataSetName + '.pkl', "wb")
    # pickle.dump(individual_dict, fileName_individual)
    with open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_meng_individual_' + dataSetName + '.pkl', "wb") as fileName_individual:
        pickle.dump(individual_dict , fileName_individual)

    return

def save_individual_to_txt(randomSeeds, dataSetName,individuals): # save individual as txt by mengxu
    file = open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_' + dataSetName+'.txt', 'w')
    file.write('Individual:\n')
    file.write('Tree 0:\n') #routing rule
    file.write(str(individuals[0]) + '\n')
    file.write('Tree 1:\n') #sequencing rule
    file.write(str(individuals[1]) + '\n')

    file.close()
    return

def clear_individual_each_gen_to_txt(randomSeeds, dataSetName): # save individual as txt by mengxu
    file = open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_' + dataSetName+'_each_gen.txt', 'w') # 'w' represent coverage, 'a' denotes not coverage
    file.write("Best individuals from each gen:\n")
    file.close()
    return

def save_individual_each_gen_to_txt(randomSeeds, dataSetName, individuals, gen): # save individual as txt by mengxu
    file = open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_' + dataSetName+'_each_gen.txt', 'a') # 'w' represent coverage, 'a' denotes not coverage
    file.write('\nGen: ' + str(gen) + '\n')
    file.write('Individual:\n')
    file.write('Tree 0:\n') #routing rule
    file.write(str(individuals[0]) + '\n')
    file.write('Tree 1:\n') #sequencing rule
    file.write(str(individuals[1]) + '\n')

    file.close()
    return


def save_archive(randomSeeds, dataSetName,individuals):
    with open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_archive' + dataSetName+'.pickle', 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return

def save_pop(randomSeeds, dataSetName,individuals):
    with open('./MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds) + '_pop' + dataSetName+'.pickle', 'wb') as file:
        pickle.dump(individuals, file, protocol=pickle.HIGHEST_PROTOCOL)
    file.close()
    return


def saveMinFitness(randomSeeds, dataSetName, min_fitness):
    fileName1= './MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds)+'_min_fitness' + dataSetName
    np.save(fileName1, min_fitness)
    return

def saveRunningTime(randomSeeds, dataSetName, running_time):
    fileName1= './MTGP/train/scenario_' + str(dataSetName) + '/' + str(randomSeeds)+'_running_time' + dataSetName
    np.save(fileName1, running_time)
    return


# def saveResults(fileName, *args, **kwargs):
#     f = open(fileName, 'w')#####Truncate file to zero length or create text file for writing
#     for i in args:
#         f.writelines(str(i)+'\n')
#     f.close()
#     return
#
# def saveLog (fileName, log):
#     f=open(fileName, 'wb')###open the file with binary model for writing
#     pickle.dump(log, f)
#     f.close()
#     return
#
# #def saveAllResults(randomSeeds, dataSetName, running_time,train_accuracy,test_accuracy,whole_accuracy, frequency_accuracy, tempEXA,
# # real_value_features,value_features_01):####store the time:
# def saveAlltime(randomSeeds, dataSetName, running_30):
#     fileName1='running_30' + dataSetName + '.txt'
#     saveResults(fileName1, running_30)
#     return
#
#
# def saveAlltrain(randomSeeds, dataSetName, train_accuracy_30):
#     fileName1= str(randomSeeds)+'train_accuracy_30' + dataSetName + '.txt'
#     saveResults(fileName1, 'train', train_accuracy_30)
#     return
#
# def saveAllrandomly(randomSeeds, dataSetName, randomly_test_accuracy_30):
#     fileName1= str(randomSeeds)+'randomly_test_accuracy_30' + dataSetName + '.txt'
#     saveResults(fileName1, 'randomly_test', randomly_test_accuracy_30)
#     return
# def saveAllfrequency(randomSeeds, dataSetName, frequency_accuracy_30):
#     fileName1= str(randomSeeds)+'frequency_accuracy_30' + dataSetName + '.txt'
#     saveResults(fileName1, 'frequency', frequency_accuracy_30)
#     return
#
# def saveAllwhole(randomSeeds, dataSetName, whole_accuracy_30):
#     fileName1= str(randomSeeds)+'whole_accuracy_30' + dataSetName + '.txt'
#     saveResults(fileName1, 'whole', whole_accuracy_30)
#     return
# def saveAllfeature(randomSeeds, dataSetName, value_features_01):
#     fileName1= str(randomSeeds)+'value_features_01_30' + dataSetName + '.txt'
#     saveResults(fileName1, 'feature', value_features_01)
#     return
#
# def saveAllfeature1(randomSeeds, dataSetName, EXA_01):
#     fileName1= str(randomSeeds)+'EXA_01' + dataSetName
#     np.save(fileName1, EXA_01)
#     return
#
# def saveAllfeature2(randomSeeds, dataSetName, EXA_array):
#     fileName1= str(randomSeeds)+'EXA_array' + dataSetName
#     np.save(fileName1, EXA_array)
#     return
#
# def saveAllfeature3(randomSeeds, dataSetName, front_training):
#     fileName1= str(randomSeeds)+'front_training' + dataSetName
#     np.save(fileName1, front_training)
#     return
# def saveAllfeature4(randomSeeds, dataSetName, front_testing):
#     fileName1= str(randomSeeds)+'front_testing' + dataSetName
#     np.save(fileName1, front_testing)
#     return
#
# def saveAllfeature5(randomSeeds, dataSetName, unique_number):
#     fileName1= str(randomSeeds)+'unique_number' + dataSetName
#     np.save(fileName1, unique_number)
#     return


# def saveAllfeature8(randomSeeds, dataSetName, ensemble_solution):
#     fileName1= str(randomSeeds)+'ensemble_solution' + dataSetName
#     np.save(fileName1, ensemble_solution)
#     return
#
#
# def saveAllfeature9(randomSeeds, dataSetName, encoding):
#     fileName1= str(randomSeeds)+'encoding' + dataSetName + '.txt'
#     saveResults(fileName1, encoding)
#     return
#
# def saveAllfeature10(randomSeeds, dataSetName, decoding):
#     fileName1= str(randomSeeds)+'decoding' + dataSetName + '.txt'
#     saveResults(fileName1, decoding)
#     return
#
# def saveAllhyp1(dataSetName, hyp_30_training):
#     fileName1= 'hyp_30_training' + dataSetName + '.txt'
#     saveResults(fileName1, hyp_30_training)
#     return
#
# def saveAllhyp2(dataSetName, hyp_30_testing):
#     fileName1= 'hyp_30_testing' + dataSetName + '.txt'
#     saveResults(fileName1, hyp_30_testing)
#     return
#
#
