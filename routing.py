import math

import simpy
import random
import numpy as np
import torch

'''
this module contains the machine routing rules used for comparison
routing agents may choose to follow one of following rules
or choose to use trained parameters for decision-making
'''

# Benchmark, as the worst possible case
def random_routing(idx, data, job_pt, job_slack, wc_idx, *args):
    machine_idx = np.random.randint(len(job_pt))
    return machine_idx

def TT(idx, data, job_pt, job_slack, wc_idx, *args): # shortest total waiting time
    # axis=0 means choose along columns
    # print("routing data:", data)
    rank = np.argmin(data, axis=0)
    machine_idx = rank[0]
    return machine_idx

def ET(idx, data, job_pt, job_slack, wc_idx, *args): # minimum exceution time
    machine_idx = np.argmin(job_pt)
    return machine_idx

def EA(idx, data, job_pt, job_slack, wc_idx, *args): # earliest available
    #print(data, np.transpose(data))
    rank = np.argmin(data, axis=0)
    machine_idx = rank[1]
    return machine_idx

def SQ(idx, data, job_pt, job_slack, wc_idx, *args): # shortest queue
    rank = np.argmin(data, axis=0)
    machine_idx = rank[2]
    return machine_idx

def CT(idx, data, job_pt, job_slack, wc_idx, *args): # earliest completion time
    #print(data,job_pt)
    completion_time = np.array(data)[:,1].clip(0) + np.array(job_pt)
    machine_idx = completion_time.argmin()
    return machine_idx

def UT(idx, data, job_pt, job_slack, wc_idx, *args): # lowest utilization rate
    rank = np.argmin(data, axis=0)
    machine_idx = rank[3]
    return machine_idx

def GP_R1(idx, data, job_pt, job_slack, wc_idx, *args): # genetic programming
    data = np.transpose(data)
    sec1 = min(2 * data[2] * np.max([data[2]*job_pt/data[1] , job_pt*data[0]*data[0]], axis=0))
    sec2 = data[2] * job_pt - data[1]
    sum = sec1 + sec2
    machine_idx = sum.argmin()
    return machine_idx

def GP_R2(idx, data, job_pt, job_slack, wc_idx, *args): # genetic programming
    data = np.transpose(data) #todo: check what's the data here!!!
    sec1 = data[2]*data[2], (data[2]+job_pt)*data[2]
    sec2 = np.min([data[1],args[0]/(data[1]*args[0]-1)],axis=0)
    sec3 = -data[2] * args[0]
    sec4 = data[2] * job_pt * np.max([data[0], np.min([data[1],job_pt],axis=0)/(args[0])],axis=0)
    sec5 = np.max([data[2]*data[2], np.ones_like(data[2])*(args[1]-args[0]-1), (data[2]+job_pt)*np.min([data[2],np.ones_like(data[2])*args[1]],axis=0)],axis=0)
    sum = sec1 - sec2 * np.max([sec3+sec4/sec5],axis=0)
    machine_idx = sum.argmin()
    return machine_idx

# subtract('PT', 'NIQ')
# multiply(add('PT', add('PT', add('WIQ', minimum('WIQ', add(multiply('MWT', 'WIQ'), maximum('MWT', 'NOR')))))), minimum(add(minimum('W', 'WIQ'), maximum('WIQ', 'OWT')), subtract('TIS', 'TIS')))
# def GP_pair_R(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, *args): # genetic programming rule 1
#     data = np.transpose(data)
#     sec1 = np.minimum(W, data[1])
#     sec2 = np.maximum(data[1], OWT)
#     sec3 = sec1 + sec2
#     sec4 = np.minimum(sec3, 0)
#
#     sec5 = data[2] * data[1]
#     sec6 = np.maximum(data[2], NOR)
#     sec7 = sec5 + sec6
#     sec8 = np.minimum(data[1], sec7)
#     sec9 = data[1] + sec8
#     sec10 = current_pt + sec9
#     sec11 = current_pt + sec10
#     sum = sec11 * sec4
#
#     job_position = sum.argmin()
#     return job_position

# minimum(add(subtract(add('PT', 'PT'), protected_div('PT', 'PT')), 'PT'), 'NIQ')
# add('PT', 'NIQ')
# def GP_pair_R(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, *args): # genetic programming rule 1
#     data = np.transpose(data)
#     sum = current_pt + data[0]
#
#     job_position = sum.argmin()
#     return job_position

# minimum(add(subtract('PT', protected_div('NIQ', add('WKR', 'PT'))), 'NIQ'), 'PT')
# add(multiply('PT', 'NIQ'), add(subtract('WKR', 'PT'), 'NIQ'))
# def GP_pair_R(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args): # genetic programming rule 1
#     data = np.transpose(data)
#     sec1 = current_pt * data[0]
#     sec2 = WKR - current_pt + data[0]
#     sum = sec1 + sec2
#
#     machine_idx = sum.argmin()
#     return machine_idx

def GP_pair_R_test(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args): # genetic programming rule 1
    data = np.transpose(data)  # todo: check what's the data here!!!
    individualvalue = treeNode_R_test(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W,
                                 TIS, SLACK)  # todo: actually, this should be used for sequencing rule
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        # print("Error here: GP_pair_R_test!")
        return 0  # todo: need to check if this is right!!! by mengxu 2022.10.15
    machine_idx = individualvalue.argmin()
    return machine_idx

def GP_pair_ensemble_R_test(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args): # genetic programming rule 1
    data = np.transpose(data)  # todo: check what's the data here!!!
    individualvalue = treeNode_R_test(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W,
                                 TIS, SLACK)  # todo: actually, this should be used for sequencing rule
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        print("Error here: GP_pair_ensemble_R_test!")
        return 0  # todo: need to check if this is right!!! by mengxu 2022.10.15
    return individualvalue
    # machine_idx = individualvalue.argmin()
    # return machine_idx

def treeNode_R_test(tree, index, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK):
    if tree[index] == 'add':
        return treeNode_R_test(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK) + treeNode_R_test(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
    elif tree[index] == 'subtract':
        return treeNode_R_test(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK) - treeNode_R_test(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
    elif tree[index] == 'multiply':
        return treeNode_R_test(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK) * treeNode_R_test(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
    elif tree[index] == 'protected_div':
        return protected_div(treeNode_R_test(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK), treeNode_R_test(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK))
    elif tree[index] == 'maximum':
        return np.maximum(treeNode_R_test(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK), treeNode_R_test(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK))
    elif tree[index] == 'minimum':
        return np.minimum(treeNode_R_test(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK), treeNode_R_test(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK))
    elif tree[index] == 'lf': # add by mengxu 2022.11.08
        ref = treeNode_R_test(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        if isinstance(ref, (np.int64, np.float64, float, int)):
            return 1 / (1 + np.exp(-ref))
        else:
            for i in range(len(ref)):
                ref[i] = 1 / (1 + np.exp(-ref[i]))
            return ref
    elif tree[index] == 'NIQ':
        return data[0]
    elif tree[index] == 'WIQ':
        return data[1]
    elif tree[index] == 'MWT':
        return data[2]
    elif tree[index] == 'PT':
        return current_pt
    elif tree[index] == 'NPT':
        return next_pt
    elif tree[index] == 'OWT':
        return OWT
    elif tree[index] == 'WKR':
        return WKR
    elif tree[index] == 'NOR':
        return NOR
    elif tree[index] == 'W':
        return W
    elif tree[index] == 'TIS':
        return TIS
    elif tree[index] == 'SLACK':
        return SLACK

def GP_pair_R_ranks(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args): # genetic programming rule 1
    data = np.transpose(data)  # todo: check what's the data here!!!
    individualvalue = treeNode_R_test(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W,
                                 TIS, SLACK)  # todo: actually, this should be used for sequencing rule
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        # print("Error here: GP_pair_R_test!")
        return [0]  # todo: need to check if this is right!!! by mengxu 2022.10.15
    ranks = [0 for i in range(len(individualvalue))]

    for i in range(len(individualvalue)):
        machine_idx = individualvalue.argmin()
        ranks[machine_idx] = i
        # print("individualvalue[job_position]: " + str(individualvalue[job_position]))
        individualvalue[machine_idx] = 10000000
    return ranks  # todo: need to check by mengxu 2023.10.18
def GP_evolve_R_ranks(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args): # genetic programming evolved sequencing rule
    data = np.transpose(data)  # todo: check what's the data here!!!
    individualvalue = treeNode_R(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)  # todo: actually, this should be used for sequencing rule
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return [0] #todo: need to check if this is right!!! by mengxu 2022.10.15

    ranks = [0 for i in range(len(individualvalue))]

    for i in range(len(individualvalue)):
        machine_idx = individualvalue.argmin()
        ranks[machine_idx] = i
        # print("individualvalue[job_position]: " + str(individualvalue[job_position]))
        individualvalue[machine_idx] = 10000000
    return ranks  # todo: need to check by mengxu 2023.10.18


def GP_evolve_R(tree_R, idx, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK, *args): # genetic programming evolved sequencing rule
    data = np.transpose(data)  # todo: check what's the data here!!!
    individualvalue = treeNode_R(tree_R, 0, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)  # todo: actually, this should be used for sequencing rule
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return 0 #todo: need to check if this is right!!! by mengxu 2022.10.15
    machine_idx = individualvalue.argmin()
    return machine_idx


def treeNode_R(tree, index, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK):
    if tree[index].arity == 2: #todo: meed to double check as this is for multiple machines to calculate the priority together!
        if tree[index].name == 'add':
            return treeNode_R(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK) + treeNode_R(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        elif tree[index].name == 'subtract':
            return treeNode_R(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK) - treeNode_R(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        elif tree[index].name == 'multiply':
            return treeNode_R(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK) * treeNode_R(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
        elif tree[index].name == 'protected_div':
            return protected_div(treeNode_R(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK), treeNode_R(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK))
        elif tree[index].name == 'maximum':
            return np.maximum(treeNode_R(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK), treeNode_R(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK))
        elif tree[index].name == 'minimum':
            return np.minimum(treeNode_R(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK), treeNode_R(tree, index+2, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK))
    elif tree[index].arity == 1:
        if tree[index].name == 'lf': # add by mengxu 2022.11.08
            ref = treeNode_R(tree, index+1, data, current_pt, next_pt, OWT, WKR, NOR, W, TIS, SLACK)
            if isinstance(ref, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-ref))
            else:
                for i in range(len(ref)):
                    ref[i] = 1 / (1 + np.exp(-ref[i]))
                return ref
    elif tree[index].arity == 0:
        if tree[index].name == 'NIQ':
            return data[0]
        elif tree[index].name == 'WIQ':
            return data[1]
        elif tree[index].name == 'MWT':
            return data[2]
        elif tree[index].name == 'PT':
            return current_pt
        elif tree[index].name == 'NPT':
            return next_pt
        elif tree[index].name == 'OWT':
            return OWT
        elif tree[index].name == 'WKR':
            return WKR
        elif tree[index].name == 'NOR':
            return NOR
        elif tree[index].name == 'W':
            return W
        elif tree[index].name == 'TIS':
            return TIS
        elif tree[index].name == 'SLACK':
            return SLACK

# def GP_evolve_R(tree_R, idx, data, job_pt, job_slack, wc_idx, *args): # genetic programming evolved sequencing rule
#     data = np.transpose(data)  # todo: check what's the data here!!!
#     individualvalue = treeNode_R(tree_R, 0, data, job_pt, job_slack, wc_idx)  # todo: actually, this should be used for sequencing rule
#     if isinstance(individualvalue, int):
#         return 0 #todo: need to check if this is right!!! by mengxu 2022.10.15
#     job_position = individualvalue.argmin()
#     return job_position


# def treeNode_R(tree, index, data, job_pt, job_slack, wc_idx):
#     if tree[index].arity == 2:
#         if tree[index].name == 'add':
#             return treeNode_R(tree, index+1, data, job_pt, job_slack, wc_idx) + treeNode_R(tree, index+2, data, job_pt, job_slack, wc_idx)
#         elif tree[index].name == 'subtract':
#             return treeNode_R(tree, index+1, data, job_pt, job_slack, wc_idx) - treeNode_R(tree, index+2, data, job_pt, job_slack, wc_idx)
#         elif tree[index].name == 'multiply':
#             return treeNode_R(tree, index+1, data, job_pt, job_slack, wc_idx) * treeNode_R(tree, index+2, data, job_pt, job_slack, wc_idx)
#         elif tree[index].name == 'protected_div':
#             return protected_div(treeNode_R(tree, index+1, data, job_pt, job_slack, wc_idx), treeNode_R(tree, index+2, data, job_pt, job_slack, wc_idx))
#         elif tree[index].name == 'maximum':
#             return np.maximum(treeNode_R(tree, index+1, data, job_pt, job_slack, wc_idx), treeNode_R(tree, index+2, data, job_pt, job_slack, wc_idx))
#         elif tree[index].name == 'minimum':
#             return np.minimum(treeNode_R(tree, index+1, data, job_pt, job_slack, wc_idx), treeNode_R(tree, index+2, data, job_pt, job_slack, wc_idx))
#         elif tree[index].arity == 0:
#             if tree[index].name == 'NIQ':
#                 return data[0]
#             elif tree[index].name == 'WIQ':
#                 return data[1]
#             elif tree[index].name == 'MWT':
#                 return data[2]
#             elif tree[index].name == 'PT':
#                 return data[3]
#             elif tree[index].name == 'NPT':
#                 return data[4]
#             elif tree[index].name == 'OWT':
#                 return data[5]
#             elif tree[index].name == 'NOR':
#                 return data[6]
#             elif tree[index].name == 'TIS':
#                 return data[7]

def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x