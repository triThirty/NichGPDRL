import copy
import random
import numpy as np

from collections import defaultdict
from deap import gp, creator
from deap import tools

# from MTGP.GPFC import N_TREES, MAX_HEIGHT


# def process_data(individual, toolbox, data):
#     no_instances = data.shape[0]
#     no_trees = len(individual)
#     feature_major = data.T
#     # [no_trees x no_instances]
#     # we do it this way so we can assign rows (constructed features) efficiently.
#     result = np.zeros(shape=(no_trees, no_instances))
#     for i, expr in enumerate(individual):
#         func = toolbox.compile(expr=expr)
#         vec = func(*feature_major)
#         if (not isinstance(vec, np.ndarray)) or vec.ndim == 0:
#             # it decided to just give us a constant back...
#             vec = np.repeat(vec, no_instances)
#         result[i] = vec
#     return result.T


def init_primitives(pset):
    # add function
    pset.addPrimitive(np.add, 2)
    pset.addPrimitive(np.subtract, 2)
    pset.addPrimitive(np.multiply, 2)
    pset.addPrimitive(protected_div, 2)
    pset.addPrimitive(np.maximum, 2)
    pset.addPrimitive(np.minimum, 2)
    # pset.addPrimitive(lf, 1)  # add by mengxu 2022.11.08 for GSGP
    # pset.addPrimitive(add_abs, 2)
    # pset.addPrimitive(sub_abs, 2)
    # pset.addPrimitive(mt_if, 3)
    # pset.addEphemeralConstant("rand", ephemeral=lambda: random.uniform(-1, 1))
    # add terminal
    # pset.addTerminal(1)  # add by mengxu //todo: the terminals seems not right, it already has three terminals in the set before I add my terminals in, need to modify
    # pset.addTerminal(2)  # add by mengxu
    # pset.addTerminal(3)  # add by mengxu

    # terminals for sequencing and routing in my paper
    pset.addTerminal(str("NIQ"))  # add by mengxu
    pset.addTerminal(str("WIQ"))  # add by mengxu
    pset.addTerminal(str("MWT"))  # add by mengxu
    pset.addTerminal(str("PT"))  # add by mengxu
    pset.addTerminal(str("NPT"))  # add by mengxu
    pset.addTerminal(str("OWT"))  # add by mengxu
    pset.addTerminal(str("WKR"))  # add by mengxu
    pset.addTerminal(str("NOR"))  # add by mengxu
    # pset.addTerminal('W')  # add by mengxu
    pset.addTerminal(str("TIS"))  # add by mengxu
    # pset.addTerminal('TRANT')  # add by mengxu

    # adviced terminal by Yi 2022.10.31
    pset.addTerminal(str("SLACK"))  # add by mengxu

    # pset.addTerminal('NIQ')  # add by mengxu
    # pset.addTerminal('WIQ')  # add by mengxu
    # pset.addTerminal('MWT')  # add by mengxu
    # pset.addTerminal('PT')  # add by mengxu
    # pset.addTerminal('NPT')  # add by mengxu
    # pset.addTerminal('OWT')  # add by mengxu
    # pset.addTerminal('WKR')  # add by mengxu
    # pset.addTerminal('NOR')  # add by mengxu
    # # pset.addTerminal('W')  # add by mengxu
    # pset.addTerminal('TIS')  # add by mengxu
    # # pset.addTerminal('TRANT')  # add by mengxu

    # # terminals for sequencing
    # pset.addTerminal('current_pt') #add by mengxu //todo: the terminals seems not right, it already has three terminals in the set before I add my terminals in, need to modify
    # pset.addTerminal('slack') #add by mengxu
    # pset.addTerminal('queue') #add by mengxu
    #
    # # terminals for routing
    # pset.addTerminal('time_in_system')  # add by mengxu
    # pset.addTerminal('que_size')  # add by mengxu


def lf(x):  # add by mengxu 2022.11.08
    return 1 / (1 + np.exp(-x))


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
    # toolbox.register("mate", xmate)
    # toolbox.register("mutate", xmut, expr=toolbox.expr_mut)

    toolbox.register("mate", lim_xmate)
    toolbox.register("mutate", lim_xmut, expr=toolbox.expr_mut)


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


def cxOnePoint(ind1, ind2):
    """Randomly select crossover point in each individual and exchange each
    subtree with the point as root between each individual.

    :param ind1: First tree participating in the crossover.
    :param ind2: Second tree participating in the crossover.
    :returns: A tuple of two trees.
    """
    if len(ind1) < 2 or len(ind2) < 2:
        # No crossover on single node tree
        return ind1, ind2

    # List all available primitive types in each individual
    types1 = defaultdict(list)
    types2 = defaultdict(list)
    if ind1.root.ret == __type__:
        # Not STGP optimization
        types1[__type__] = list(range(1, len(ind1)))
        types2[__type__] = list(range(1, len(ind2)))
        common_types = [__type__]
    else:
        for idx, node in enumerate(ind1[1:], 1):
            types1[node.ret].append(idx)
        for idx, node in enumerate(ind2[1:], 1):
            types2[node.ret].append(idx)
        common_types = set(types1.keys()).intersection(set(types2.keys()))

    if len(common_types) > 0:
        type_ = random.choice(list(common_types))

        index1 = random.choice(types1[type_])
        index2 = random.choice(types2[type_])

        slice1 = ind1.searchSubtree(index1)
        slice2 = ind2.searchSubtree(index2)
        ind1[slice1], ind2[slice2] = ind2[slice2], ind1[slice1]

    return ind1, ind2


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
            tree_1[slice1], tree_2[slice2] = tree_2[slice2], tree_1[slice1]

    return ind1, ind2


# the following is modified by mengxu
def xmate(ind1, ind2):
    if len(ind1) == 2:
        randomValue = random.random()
        if randomValue < 0.9:  # crossover
            i1 = random.randrange(len(ind1))
            # i2 = random.randrange(len(ind2))
            # todo: I think this is not same with my MTGP, as only the same type of tree can be used to do crossover
            ind1[i1], ind2[i1] = cxOnePoint(ind1[i1], ind2[i1])

            # exchange the other tree
            i2 = 1 - i1  # only for individual with two tree
            ind1[i2], ind2[i2] = ind2[i2], ind1[i2]
        else:
            ind1, ind2 = newcxOnePoint(ind1, ind2)
            del ind1.l_min
            del ind1.l_max
            del ind1.r_min
            del ind1.r_max
            del ind2.l_min
            del ind2.l_max
            del ind2.r_min
            del ind2.r_max
    else:
        if len(ind1) == 2:
            ind1[0], ind2[0] = gp.cxOnePoint(ind1[0], ind2[0])
    return ind1, ind2


# def xmate(ind1, ind2):
#     i1 = random.randrange(len(ind1))
#     i2 = random.randrange(len(ind2))
#     ind1[i1], ind2[i2] = gp.cxOnePoint(ind1[i1], ind2[i2])
#     return ind1, ind2


def lim_xmate(ind1, ind2):
    return wrap(xmate, ind1, ind2)


# def mutUniform(individual, expr, pset, mutate_point):
#     """Randomly select a point in the tree *individual*, then replace the
#     subtree at that point as a root by the expression generated using method
#     :func:`expr`.

#     :param individual: The tree to be mutated.
#     :param expr: A function object that can generate an expression when
#                  called.
#     :returns: A tuple of one tree.
#     """
#     # index = random.randrange(len(individual))
#     index = mutate_point
#     slice_ = individual.searchSubtree(index)
#     type_ = individual[index].ret
#     individual[slice_] = expr(pset=pset, type_=type_)
#     return (individual,)


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
    ind = individual[0]
    index = individual.l_max
    slice_ = ind.searchSubtree(index)
    type_ = ind[index].ret
    ind[slice_] = expr(pset=pset, type_=type_)
    individual[0] = ind

    ind = individual[1]
    index = individual.r_max
    slice_ = ind.searchSubtree(index)
    type_ = ind[index].ret
    ind[slice_] = expr(pset=pset, type_=type_)
    individual[1] = ind
    del individual.l_min
    del individual.l_max
    del individual.r_min
    del individual.r_max
    return (individual,)


def xmut(ind, expr):
    # print("The mutated point is:", ind.minimal_score_node_index)
    # mutate_point = 0
    # if len(ind[0]) > ind.minimal_score_node_index:
    #     i1 = 0
    #     mutate_point = ind.minimal_score_node_index
    # else:
    #     i1 = 1
    #     mutate_point = ind.minimal_score_node_index - len(ind[0])
    # i1 = random.randrange(len(ind))
    # indx = mutUniform(ind[i1], expr, pset=ind.pset, mutate_point=mutate_point)
    ind = mutUniform(ind, expr, pset=ind.pset)
    # ind[i1] = indx[0]
    # return (ind,)
    return ind


def lim_xmut(ind, expr):
    # have to put expr=expr otherwise it tries to use it as an individual
    res = wrap(xmut, ind, expr=expr)
    # print(res)
    return res


def add_abs(a, b):
    return np.abs(np.add(a, b))


def sub_abs(a, b):
    return np.abs(np.subtract(a, b))


def mt_if(a, b, c):
    return np.where(a < 0, b, c)


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
