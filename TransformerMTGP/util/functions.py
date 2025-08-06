import math

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


def phyno_hash_individual(ind):
    return hash(str(ind.decision_vector))


def hash_individual(ind):
    return hash(str(ind[0]) + str(ind[1]))


def phyno_remove_duplicates(population):
    unique_pop = []
    seen = set()

    for i, ind in enumerate(population):
        h = phyno_hash_individual(ind)
        if h not in seen:
            seen.add(h)
            unique_pop.append(ind)

    return unique_pop


def remove_duplicates(population):
    unique_pop = []
    seen = set()

    for ind in population:
        h = hash_individual(ind)
        if h not in seen:
            seen.add(h)
            unique_pop.append(ind)

    return unique_pop


def calculate_ranking_accuracy(data):
    concordant_pairs = 0
    discordant_pairs = 0

    n = len(data)

    for i in range(n):
        for j in range(i + 1, n):
            item_i = data[i]
            item_j = data[j]

            fitness_diff = item_i.fitness.values[0] - item_j.fitness.values[0]
            score_diff = item_i.score - item_j.score

            if fitness_diff != 0 and fitness_diff * score_diff < 0:
                concordant_pairs += 1
            elif fitness_diff == 0 and score_diff == 0:
                concordant_pairs += 1
            else:
                discordant_pairs += 1
    accuracy = concordant_pairs / (concordant_pairs + discordant_pairs)
    return accuracy


def calculate_score_based_ind_proportion(
    sorted_pop_intermediate_by_fitness, sorted_pop_intermediate_by_score
):
    total_individuals = len(sorted_pop_intermediate_by_fitness)
    concordant_pairs = 0

    for ind in sorted_pop_intermediate_by_score:
        if ind in sorted_pop_intermediate_by_fitness:
            concordant_pairs += 1

    return concordant_pairs / total_individuals
