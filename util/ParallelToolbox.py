import pickle

from deap import base
from multiprocessing import Pool
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
    def multiProcess(self, evaluate, invalid_ind, config, seed):
        pickle.dumps(invalid_ind)
        pickle.dumps(evaluate)
        partial_evaluate = partial(evaluate, config=config, seed=seed)
        fitnesses = Pool(processes=2).map(partial_evaluate, invalid_ind)
        return fitnesses
