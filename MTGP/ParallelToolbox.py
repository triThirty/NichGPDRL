import pickle

# import json

from deap import base
from multiprocessing import cpu_count, Pool

from functools import partial


##thanks TPOT
## https://github.com/EpistasisLab/tpot/pull/100/files
class ParallelToolbox(base.Toolbox):
    """Runs the TPOT genetic algorithm over multiple cores."""

    def __getstate__(self):
        self_dict = self.__dict__.copy()
        del self_dict["map"]
        return self_dict

    def __setstate__(self, state):
        self.__dict__.update(state)

    # created by mengxu 2022.11.28 for multiple processing
    def multiProcess(self, evaluate, invalid_ind, rd):
        cores = cpu_count()
        # print("cores: " + str(cores))
        pickle.dumps(invalid_ind)
        pickle.dumps(evaluate)
        partial_evaluate = partial(evaluate, rd=rd)
        fitnesses = Pool(processes=2).map(partial_evaluate, invalid_ind)
        # fitnesses = []
        # for ind in invalid_ind:
        #     evaluate(ind)
        #     fitnesses.append(ind.fitness.values)
        return fitnesses
        # cube_parts = self.inds_split(invalid_ind, cores)
        # # print("objective: " + objectives[i])
        # with Pool(cores) as p:
        #     ObjValue = p.map(evaluate, cube_parts)
        #     print("ObjValue: " + str(ObjValue))
        #     try:
        #         return concatenate(ObjValue)
        #     except ValueError:
        #         print("Array provided is smaller than # of cores available")
