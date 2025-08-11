import math
import random

import torch


def positional_encoding(seq_len, embed_dim, device):
    position = torch.arange(seq_len).unsqueeze(1).to(device)
    div_term = torch.exp(
        torch.arange(0, embed_dim, 2) * -(math.log(10000.0) / embed_dim)
    ).to(device)
    pe = torch.zeros(seq_len, embed_dim).to(device)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe


def list_net_loss(scores, labels, margin=0.0, lambda_var=0.1):
    """
    scores: 模型预测分数 [batch_size]
    labels: 样本标签 [batch_size]
    margin: Ranking Loss的间隔参数
    lambda_var: 组内方差正则化系数
    """
    loss = 0.0
    scores = -scores
    n = scores.shape[0]

    # 计算组内方差正则化
    unique_labels = torch.unique(labels)
    var_loss = 0.0
    for l in unique_labels:
        group_mask = labels == l
        group_scores = scores[group_mask]
        if len(group_scores) > 1:
            var_loss += torch.var(group_scores)
    var_loss *= lambda_var

    # 计算Pairwise对比损失
    for i in range(n):
        for j in range(i + 1, n):
            if labels[i] == labels[j]:
                # 相同标签：强制分数接近
                loss += scores[i] - scores[j]
            else:
                # 不同标签：使用Margin Ranking Loss
                sign = 1.0 if labels[i] > labels[j] else -1.0
                diff = (scores[i] - scores[j]) * sign
                loss += torch.relu(diff)

    return loss / (n * (n - 1) / 2) + var_loss


def varAnd(
    population,
    toolbox,
    cxpb,
    mutpb,
    reppb,
    config,
):
    offspring = [toolbox.clone(ind) for ind in population]
    new_cxpb = cxpb / (cxpb + mutpb + reppb)
    new_mutpb = mutpb / (cxpb + mutpb + reppb) + new_cxpb
    i = 0
    while i < len(offspring):
        randomValue = random.random()
        if randomValue < new_cxpb:  # crossover
            if random.random() < config.exploration_ratio or i == len(offspring) - 1:
                if i < len(offspring) - 1 and offspring[i] == offspring[i + 1]:
                    (offspring[i],) = toolbox.score_mutate(offspring[i])
                    (offspring[i + 1],) = toolbox.score_mutate(offspring[i + 1])
                    i += 2
                else:
                    (offspring[i], _) = toolbox.score_mate(
                        # offspring[i], population[min_indices[i]]
                        offspring[i],
                        population[(i + 1) % len(offspring)],
                    )
                    del offspring[i].fitness.values
                    del offspring[i].l_min
                    del offspring[i].l_max
                    del offspring[i].r_min
                    del offspring[i].r_max
                    i += 1
            else:
                if offspring[i] == offspring[(i + 1) % 40]:
                    (offspring[i],) = toolbox.mutate(offspring[i])
                    (offspring[i + 1],) = toolbox.mutate(offspring[i + 1])
                else:
                    offspring[i], offspring[i + 1] = toolbox.mate(
                        offspring[i], offspring[i + 1]
                    )
                    del offspring[i].fitness.values
                    del offspring[i].l_min
                    del offspring[i].l_max
                    del offspring[i].r_min
                    del offspring[i].r_max
                    del offspring[i + 1].fitness.values
                    del offspring[i + 1].l_min
                    del offspring[i + 1].l_max
                    del offspring[i + 1].r_min
                    del offspring[i + 1].r_max

                i += 2
        elif new_cxpb <= randomValue < new_mutpb:  # mutation
            if random.random() < config.exploration_ratio:
                (offspring[i],) = toolbox.score_mutate(offspring[i])
                del offspring[i].fitness.values
                del offspring[i].l_min
                del offspring[i].l_max
                del offspring[i].r_min
                del offspring[i].r_max
            else:
                (offspring[i],) = toolbox.mutate(offspring[i])
                del offspring[i].fitness.values
                del offspring[i].l_min
                del offspring[i].l_max
                del offspring[i].r_min
                del offspring[i].r_max
            i = i + 1
        else:
            i += 1
    return offspring
