import copy
import random

from util.multi_tree import maxheight, wrap
from deap import gp, creator
from deap import tools


def init_toolbox(toolbox, pset):
    creator.create("Individual", list, fitness=creator.FitnessMin, pset=pset)

    toolbox.register(
        "expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6
    )  # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    N_TREES = 2  # todo: only for test, need to be the same with original GPFC.py
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.tree, n=N_TREES
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)

    toolbox.register("expr_mut", gp.genFull, min_=2, max_=8)

    toolbox.register("mate", lim_xmate)
    toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut)


# the following is modified by mengxu
def xmate(ind1, ind2):
    i1 = random.randrange(len(ind1))
    ind1[i1], ind2[i1] = gp.cxOnePoint(ind1[i1], ind2[i1])

    # exchange the other tree
    i2 = 1 - i1  # only for individual with two tree
    ind1[i2], ind2[i2] = ind2[i2], ind1[i2]
    return ind1, ind2


def lim_xmate(ind1, ind2):
    return wrap(xmate, ind1, ind2)


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    ind[i1] = indx[0]
    return (ind,)


def lim_xmut(ind, expr):
    res = wrap(xmut, ind, expr=expr)
    return res
