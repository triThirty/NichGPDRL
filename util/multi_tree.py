import copy
import random
import numpy as np

from collections import defaultdict
from deap import gp, creator
from deap import tools
from functools import partial


def init_primitives(pset):
    # add function
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)

    # terminals for sequencing and routing in my paper
    pset.addTerminal(str("NIQ"))  # add by mengxu
    pset.addTerminal(str("WIQ"))  # add by mengxu
    pset.addTerminal(str("MWT"))  # add by mengxu
    pset.addTerminal(str("PT"))  # add by mengxu
    pset.addTerminal(str("NPT"))  # add by mengxu
    pset.addTerminal(str("OWT"))  # add by mengxu
    pset.addTerminal(str("WKR"))  # add by mengxu
    pset.addTerminal(str("NOR"))  # add by mengxu
    pset.addTerminal(str("TIS"))  # add by mengxu
    pset.addTerminal(str("SLACK"))  # add by mengxu


def init_toolbox(toolbox, pset):
    creator.create(
        "Individual",
        list,
        fitness=creator.FitnessMin,
        num_calculation=int,
        pset=pset,
    )

    toolbox.register(
        "expr", gp.genHalfAndHalf, pset=pset, min_=1, max_=6
    )  # original max = 6, modified by mengxu 2022.10.15 to check
    toolbox.register("tree", tools.initIterate, gp.PrimitiveTree, toolbox.expr)
    toolbox.register(
        "individual", tools.initRepeat, creator.Individual, toolbox.tree, n=N_TREES
    )
    toolbox.register("population", tools.initRepeat, list, toolbox.individual)
    toolbox.register("compile", gp.compile, pset=pset)
    toolbox.register("expr_mut", gp.genFull, min_=2, max_=8)

    toolbox.register("mate", lim_xmate)
    toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut)

    toolbox.register("score_mate", newlim_xmate)
    toolbox.register("score_mutate", newlim_xmut, expr=toolbox.expr_mut)


def maxheight(v):
    return max(i.height for i in v)


# stolen from gp.py....because you can't pickle decorated functions.
def wrap(func, *args, **kwargs):
    # MAX_HEIGHT = 8 #todo: only for test, need to be the same with original GPFC.py
    keep_inds = [copy.deepcopy(ind) for ind in args]
    new_inds = list(func(*args, **kwargs))
    for i, ind in enumerate(new_inds):
        if maxheight(ind) > MAX_HEIGHT:
            new_inds[i] = random.choice(keep_inds)
    return new_inds


__type__ = object


def newcxOnePoint(ind1, ind2):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    for i in range(len(ind1)):

        tree_1 = ind1[i]
        tree_2 = ind2[i]

        if len(tree_1) < 2 or len(tree_2) < 2:
            # No crossover on single node tree
            return ind1, ind2

        # List all available primitive types in each individual
        types1 = defaultdict(list)
        types2 = defaultdict(list)
        if tree_1.root.ret == __type__:
            # Not STGP optimization
            types1[__type__] = list(range(1, len(tree_1)))
            types2[__type__] = list(range(1, len(tree_2)))
            common_types = [__type__]
        else:
            for idx, node in enumerate(tree_1[1:], 1):
                types1[node.ret].append(idx)
            for idx, node in enumerate(tree_2[1:], 1):
                types2[node.ret].append(idx)
            common_types = set(types1.keys()).intersection(set(types2.keys()))

        if len(common_types) > 0:
            if i == 0:
                index1 = ind1.l_min
                index2 = ind2.l_max
            else:
                index1 = ind1.r_min
                index2 = ind2.r_max

            slice1 = tree_1.searchSubtree(index1)
            slice2 = tree_2.searchSubtree(index2)
            tree_1[slice1] = tree_2[slice2]

    return ind1, ind2


def newxmate(ind1, ind2):
    if len(ind1) == 2:
        ind1, ind2 = newcxOnePoint(ind1, ind2)
    else:
        if len(ind1) == 2:
            ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    return ind1, ind2


# the following is modified by mengxu
def xmate(ind1, ind2):
    if len(ind1) == 2:
        i1 = random.randrange(len(ind1))
        # todo: I think this is not same with my MTGP, as only the same type of tree can be used to do crossover
        ind1[i1], ind2[i1] = gp.cxOnePoint(ind1[i1], ind2[i1])

        # exchange the other tree
        i2 = 1 - i1  # only for individual with two tree
        ind1[i2], ind2[i2] = ind2[i2], ind1[i2]
    else:
        if len(ind1) == 2:
            ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    return ind1, ind2


def lim_xmate(ind1, ind2):
    return wrap(xmate, ind1, ind2)


def newlim_xmate(ind1, ind2):
    return wrap(
        newxmate,
        ind1,
        ind2,
    )


# score-based mutation
def mutUniform(individual, expr, pset):
    """Randomly select a point in the tree *individual*, then replace the
    subtree at that point as a root by the expression generated using method
    :func:`expr`.

    :param individual: The tree to be mutated.
    :param expr: A function object that can generate an expression when
                 called.
    :returns: A tuple of one tree.
    """
    # index = random.randrange(len(individual))
    tree1 = individual[0]
    index = individual.l_min
    slice_ = tree1.searchSubtree(index)
    type_ = tree1[index].ret
    tree1[slice_] = expr(pset=pset, type_=type_)
    individual[0] = tree1

    tree2 = individual[1]
    index = individual.r_min
    slice_ = tree2.searchSubtree(index)
    type_ = tree2[index].ret
    tree2[slice_] = expr(pset=pset, type_=type_)
    individual[1] = tree2
    return individual


def xmut(ind, expr):
    i1 = random.randrange(len(ind))
    indx = gp.mutUniform(ind[i1], expr, pset=ind.pset)
    ind[i1] = indx[0]
    return (ind,)


def newxmut(ind, expr):
    ind = mutUniform(ind, expr, pset=ind.pset)  # score-based mutation
    return (ind,)


def lim_xmut(ind, expr):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut, ind, expr=expr)
    return res


def newlim_xmut(ind, expr):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(newxmut, ind, expr=expr)
    return res


def protected_div(left, right):
    with np.errstate(divide="ignore", invalid="ignore"):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x


MAX_HEIGHT = 8
N_TREES = 2  # todo: only for test, need to be the same with original GPFC.py
