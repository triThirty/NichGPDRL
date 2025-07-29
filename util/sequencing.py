import math

import simpy
import random
import numpy as np
import math
from deap import gp
# import MTGP.multi_tree as mt

'''
this module contains the job sequencing rules used in the experiment
sequencing agents may choose to follow one of following rules
or choose to use trained parameters for decision-making
'''

# Benchmark, as the worst possible case
def random_sequencing(data):
    job_position = np.random.randint(len(data[0]))
    return job_position

def SPT(data): # shortest processing time
    job_position = np.argmin(data[0])
    return job_position

def LPT(data): # longest processing time
    job_position = np.argmax(data[0])
    return job_position

def LRO(data): # least remaining operations / highest completion rate
    job_position = np.argmax(data[10])
    return job_position

def LWKR(data): # least work remaining
    job_position = np.argmin(data[0] + data[1])
    return job_position

def LWKRSPT(data): # remaining work + SPT
    job_position = np.argmin(data[0]*2 + data[1])
    return job_position

def LWKRMOD(data): # remaining work + MOD
    due = data[2]
    operational_finish = data[0] + data[3]
    MOD = np.max([due,operational_finish],axis=0)
    job_position = np.argmin(data[0] + data[1] + MOD)
    return job_position

def EDD(data):
    # choose the job with earlist due date
    job_position = np.argmin(data[2])
    return job_position

def COVERT(data): # cost over time
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0,None)
    priority = (1 - cost / (0.05*average_pt)).clip(0,None) / data[0]
    job_position = priority.argmax()
    return job_position

def CR(data):
    time_till_due = data[5]
    CR = time_till_due / (data[0] + data[1])
    job_position = CR.argmin()
    return job_position

def CRSPT(data): # CR+SPT
    CRSPT = data[5] / (data[0] + data[1]) + data[0]
    job_position = CRSPT.argmin()
    return job_position

def MS(data):
    slack = data[6]
    job_position = slack.argmin()
    return job_position

def MDD(data): # The modified due date is a job's original due date or its early finish time, whichever is larger
    due = data[2]
    finish = data[1] + data[3]
    MDD = np.max([due,finish],axis=0)
    job_position = MDD.argmin()
    return job_position

def MON(data):
    # Montagne's heuristic, this rule combines SPT with additional slack factor
    due_over_pt = np.array(data[2])/np.sum(data[0])
    priority = due_over_pt/np.array(data[0])
    job_position = priority.argmax()
    return job_position

def MOD(data): # The modified operational due date
    due = data[2]
    operational_finish = data[0] + data[3]
    MOD = np.max([due,operational_finish],axis=0)
    job_position = MOD.argmin()
    return job_position

def NPT(data): # next processing time
    job_position = np.argmin(data[9])
    return job_position

def ATC(data): # http://www.growingscience.com/ijiec/Vol7/IJIEC_2015_23.pdf
    #print(data)
    average_pt = data[0].mean()
    cost = (data[2] - data[3] - data[0]).clip(0,None)
    #print(average_pt, AT)
    priority = np.exp( - cost / (0.05*average_pt)) / data[0]
    #print(priority)
    job_position = priority.argmax()
    return job_position

def AVPRO(data): # average processing time per operation
    AVPRO = (data[0] + data[1]) / (data[10] + 1)
    job_position = AVPRO.argmin()
    return job_position

def SRMWK(data): # slack per remaining work, identical to CR
    SRMWK = data[6] / (data[0] + data[1])
    job_position = SRMWK.argmin()
    return job_position

def SRMWKSPT(data): # slack per remaining work + SPT, identical to CR+SPT
    SRMWKSPT = data[6] / (data[0] + data[1]) + data[0]
    job_position = SRMWKSPT.argmin()
    return job_position

def WINQ(data): # WINQ
    job_position = data[7].argmin()
    return job_position

def PTWINQ(data): # PT + WINQ
    sum = data[0] + data[7]
    job_position = sum.argmin()
    return job_position

def PTWINQS(data): # PT + WINQ + Slack
    sum = data[0] + data[6] + data[7]
    job_position = sum.argmin()
    return job_position

def DPTWINQNPT(data): # 2PT + WINQ + NPT
    sum = data[0]*2 + data[7] + data[9]
    job_position = sum.argmin()
    return job_position

def DPTLWKR(data): # 2PT + LWKR
    sum = data[0]*2 + data[1]
    job_position = sum.argmin()
    return job_position

def DPTLWKRS(data): # 2PT + LWKR + slack
    sum = data[0]*2 + data[1] + data[6]
    job_position = sum.argmin()
    return job_position

def FIFO(dummy): # first in, first out, data is not needed
    job_position = 0
    return job_position

def GP_S1(data): # genetic programming rule 1
    sec1 = data[0] + data[1]
    sec2 = (data[7]*2-1) / data[0]
    sec3 = (data[7] + data[1] + (data[0]+data[1])/(data[7]-data[1])) / data[0]
    sum = sec1-sec2-sec3
    job_position = sum.argmin()
    return job_position

def GP_S2(data): # genetic programming rule 2
    NIQ = len(data[0])
    sec1 = NIQ * (data[0]-1)
    sec2 = data[0] + data[1] * np.max([data[0],data[7]],axis=0)
    sec3 = np.max([data[7],NIQ+data[7]],axis=0)
    sec4 = (data[8]+1+np.max([data[1],np.ones_like(data[1])*(NIQ-1)],axis=0)) * np.max([data[7],data[1]],axis=0)
    sum = sec1 * sec2 + sec3 * sec4
    job_position = sum.argmin()
    return job_position

def GP_S3(data): # genetic programming rule 1
    sec1 = data[0] + data[1]
    sec2 = (data[7]*2-1) / data[0]
    sec3 = (data[7] + data[1] + (data[0]+data[1])/(data[7]-data[1])) / data[0]
    sum = sec1-sec2-sec3
    job_position = sum.argmin()
    return job_position


# subtract('PT', 'NIQ')
# multiply(add('PT', add('PT', add('WIQ', minimum('WIQ', add(multiply('MWT', 'WIQ'), maximum('MWT', 'NOR')))))), minimum(add(minimum('W', 'WIQ'), maximum('WIQ', 'OWT')), subtract('TIS', 'TIS')))
# def GP_pair_S(data, tree_S):
#     sum = data[3] - data[0]
#     machine_idx = sum.argmin()
#     return machine_idx

# minimum(add(subtract(add('PT', 'PT'), protected_div('PT', 'PT')), 'PT'), 'NIQ')
# add('PT', 'NIQ')
# def GP_pair_S(data, tree_S):
#     sum = np.minimum(3*data[3] - 1, data[0])
#     machine_idx = sum.argmin()
#     return machine_idx

# minimum(add(subtract('PT', protected_div('NIQ', add('WKR', 'PT'))), 'NIQ'), 'PT')
# add(multiply('PT', 'NIQ'), add(subtract('WKR', 'PT'), 'NIQ'))
# def GP_pair_S(data, tree_S):
#     # data[0] = np.array([data[0] for i in range(len(data[3]))])
#     # data[1] = np.array([data[1] for i in range(len(data[3]))])
#     # data[2] = np.array([data[2] for i in range(len(data[3]))])
#     new_data = []
#     new_data.append(np.array([data[0] for i in range(len(data[3]))]))
#     new_data.append(np.array([data[1] for i in range(len(data[3]))]))
#     new_data.append(np.array([data[2] for i in range(len(data[3]))]))
#     for i in range(3, len(data)):
#         new_data.append(data[i])
#     sec1 = new_data[6] + new_data[3]
#     sec2 = protected_div(new_data[0], sec1)
#     sec3 = new_data[3] - sec2
#     sec4 = sec3 + new_data[0]
#     sum  = np.minimum(sec4, new_data[3])
#     job_position = sum.argmin()
#     return job_position

def GP_pair_S_test(data, tree_S):
    # data[0] = np.array([data[0] for i in range(len(data[3]))])
    # data[1] = np.array([data[1] for i in range(len(data[3]))])
    # data[2] = np.array([data[2] for i in range(len(data[3]))])
    new_data = []
    new_data.append(np.array([data[0] for i in range(len(data[3]))]))
    new_data.append(np.array([data[1] for i in range(len(data[3]))]))
    new_data.append(np.array([data[2] for i in range(len(data[3]))]))
    for i in range(3, len(data)):
        new_data.append(data[i])
    individualvalue = treeNode_S_test(tree_S, 0, new_data)  # todo: actually, this should be used for sequencing rule
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return 0  # todo: need to check if this is right!!! by mengxu 2022.10.15
    job_position = individualvalue.argmin()
    return job_position


def treeNode_S_test(tree, index, data):
    if tree[index] == 'add':
        return treeNode_S_test(tree, index+1, data) + treeNode_S_test(tree, index+2, data)
    elif tree[index] == 'subtract':
        return treeNode_S_test(tree, index+1, data) - treeNode_S_test(tree, index+2, data)
    elif tree[index] == 'multiply':
        return treeNode_S_test(tree, index+1, data) * treeNode_S_test(tree, index+2, data)
    elif tree[index] == 'protected_div':
        return protected_div(treeNode_S_test(tree, index+1, data), treeNode_S_test(tree, index+2, data))
    elif tree[index] == 'maximum':
        return np.maximum(treeNode_S_test(tree, index+1, data), treeNode_S_test(tree, index+2, data))
    elif tree[index] == 'minimum':
        return np.minimum(treeNode_S_test(tree, index+1, data), treeNode_S_test(tree, index+2, data))
    elif tree[index] == 'lf': # add by mengxu 2022.11.08
        ref = treeNode_S_test(tree, index+1, data)
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
        return data[3]
    elif tree[index] == 'NPT':
        return data[4]
    elif tree[index] == 'OWT':
        return data[5]
    elif tree[index] == 'WKR':
        return data[6]
    elif tree[index] == 'NOR':
        return data[7]
    elif tree[index] == 'TIS':
        return data[8]
    elif tree[index] == 'SLACK':
        return data[9]

def GP_pair_S_ranks(data, tree_S):
    # data[0] = np.array([data[0] for i in range(len(data[3]))])
    # data[1] = np.array([data[1] for i in range(len(data[3]))])
    # data[2] = np.array([data[2] for i in range(len(data[3]))])
    new_data = []
    new_data.append(np.array([data[0] for i in range(len(data[3]))]))
    new_data.append(np.array([data[1] for i in range(len(data[3]))]))
    new_data.append(np.array([data[2] for i in range(len(data[3]))]))
    for i in range(3, len(data)):
        new_data.append(data[i])
    individualvalue = treeNode_S_test(tree_S, 0, new_data)  # todo: actually, this should be used for sequencing rule
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return [0]  # todo: need to check if this is right!!! by mengxu 2022.10.15
    ranks = [0 for i in range(len(individualvalue))]

    for i in range(len(individualvalue)):
        job_position = individualvalue.argmin()
        if job_position > len(ranks) - 1:
            print("Error!")
        ranks[job_position] = i
        # print("individualvalue[job_position]: " + str(individualvalue[job_position]))
        individualvalue[job_position] = 10000000
    return ranks  # todo: need to check by mengxu 2023.10.18

def GP_evolve_S_ranks(data, tree_S): # genetic programming evolved sequencing rule
    new_data = []
    new_data.append(np.array([data[0] for i in range(len(data[3]))]))
    new_data.append(np.array([data[1] for i in range(len(data[3]))]))
    new_data.append(np.array([data[2] for i in range(len(data[3]))]))
    for i in range(3,len(data)):
        new_data.append(data[i])
    # data[0] = np.array([data[0] for i in range(len(data[3]))])
    # data[1] = np.array([data[1] for i in range(len(data[3]))])
    # data[2] = np.array([data[2] for i in range(len(data[3]))])
    individualvalue = treeNode_S(tree_S, 0, new_data)
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return [0] #todo: need to check if this is right!!! by mengxu 2022.10.15
    ranks = [0 for i in range(len(individualvalue))]

    for i in range(len(individualvalue)):
        job_position = individualvalue.argmin()
        ranks[job_position] = i
        # print("individualvalue[job_position]: " + str(individualvalue[job_position]))
        individualvalue[job_position] = 10000000
    return ranks #todo: need to check by mengxu 2023.10.18

def GP_evolve_S(data, tree_S): # genetic programming evolved sequencing rule
    new_data = []
    new_data.append(np.array([data[0] for i in range(len(data[3]))]))
    new_data.append(np.array([data[1] for i in range(len(data[3]))]))
    new_data.append(np.array([data[2] for i in range(len(data[3]))]))
    for i in range(3, len(data)):
        new_data.append(data[i])
    individualvalue = treeNode_S(tree_S, 0, new_data)  # todo: actually, this should be used for sequencing rule
    if isinstance(individualvalue, (np.int64, np.float64, float, int)):
        return 0 #todo: need to check if this is right!!! by mengxu 2022.10.15
    job_position = individualvalue.argmin()
    return job_position

def treeNode_S(tree, index, data):
    if tree[index].arity == 2:
        if tree[index].name == 'add':
            return treeNode_S(tree, index+1, data) + treeNode_S(tree, index+2, data)
        elif tree[index].name == 'subtract':
            return treeNode_S(tree, index+1, data) - treeNode_S(tree, index+2, data)
        elif tree[index].name == 'multiply':
            return treeNode_S(tree, index+1, data) * treeNode_S(tree, index+2, data)
        elif tree[index].name == 'protected_div':
            return protected_div(treeNode_S(tree, index+1, data), treeNode_S(tree, index+2, data))
        elif tree[index].name == 'maximum':
            return np.maximum(treeNode_S(tree, index+1, data), treeNode_S(tree, index+2, data))
        elif tree[index].name == 'minimum':
            return np.minimum(treeNode_S(tree, index+1, data), treeNode_S(tree, index+2, data))
    elif tree[index].arity == 1:
        if tree[index].name == 'lf': # add by mengxu 2022.11.08
            ref = treeNode_S(tree, index + 1, data)
            if isinstance(ref, (np.int64, np.float64, float, int)):
                return 1 / (1 + np.exp(-ref))
            else:
                for i in range(len(ref)):
                    ref[i] = 1 / (1 + np.exp(-ref[i]))
                    # print(ref[i])
                return ref
    elif tree[index].arity == 0:
        if tree[index].name == 'NIQ':
            return data[0]
        elif tree[index].name == 'WIQ':
            return data[1]
        elif tree[index].name == 'MWT':
            return data[2]
        elif tree[index].name == 'PT':
            return data[3]
        elif tree[index].name == 'NPT':
            return data[4]
        elif tree[index].name == 'OWT':
            return data[5]
        elif tree[index].name == 'WKR':
            return data[6]
        elif tree[index].name == 'NOR':
            return data[7]
        elif tree[index].name == 'TIS':
            return data[8]
        elif tree[index].name == 'SLACK':
            return data[9]

        # return tree[index].value



# def treeNode_S(tree, index, data):
#     if tree[index].arity == 2:
#         if tree[index].name == 'add':
#             return treeNode_S(tree, index+1, data) + treeNode_S(tree, index+2, data)
#         elif tree[index].name == 'subtract':
#             return treeNode_S(tree, index+1, data) - treeNode_S(tree, index+2, data)
#         elif tree[index].name == 'multiply':
#             return treeNode_S(tree, index+1, data) * treeNode_S(tree, index+2, data)
#         elif tree[index].name == 'protected_div':
#             return protected_div(treeNode_S(tree, index+1, data), treeNode_S(tree, index+2, data))
#     elif tree[index].arity == 0:
#         if tree[index].name == 'current_pt':
#             return data[0]
#         elif tree[index].name == 'remaining_job_pt':
#             return data[1]
#         elif tree[index].name == 'remaining_job_pt':
#             return data[2]
#         elif tree[index].name == 'due_list':
#             return data[3]
#         elif tree[index].name == 'env_now':
#             return data[4]
#         elif tree[index].name == 'completion_rate':
#             return data[5]
#         elif tree[index].name == 'time_till_due':
#             return data[6]
#         elif tree[index].name == 'slack':
#             return data[7]
#         elif tree[index].name == 'winq':
#             return data[8]
#         elif tree[index].name == 'avlm':
#             return data[9]
#         elif tree[index].name == 'next_pt':
#             return data[10]
#         elif tree[index].name == 'remaining_no_op':
#             return data[11]
#         elif tree[index].name == 'waited_time':
#             return data[12]
#         elif tree[index].name == 'wc_idx':
#             return data[13]
#         elif tree[index].name == 'queue':
#             return data[14]
#         elif tree[index].name == 'm_idx':
#             return data[15]
#         elif tree[index].name == 'cumulative_pt':
#             return 1
#         elif tree[index].name == 'time_in_system':
#             return 1
#         elif tree[index].name == 'que_size':
#             return 1
#         elif tree[index].name == 'cumulative_run_time':
#             return 1
#         # return tree[index].value

def protected_div(left, right):
    with np.errstate(divide='ignore', invalid='ignore'):
        x = np.divide(left, right)
        if isinstance(x, np.ndarray):
            x[np.isinf(x)] = 1
            x[np.isnan(x)] = 1
        elif np.isinf(x) or np.isnan(x):
            x = 1
    return x