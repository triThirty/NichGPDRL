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
                loss += (scores[i] - scores[j]) ** 2
            else:
                # 不同标签：使用Margin Ranking Loss
                sign = 1.0 if labels[i] > labels[j] else -1.0
                diff = (scores[i] - scores[j]) * sign
                loss += torch.relu(margin - diff)

    return loss / (n * (n - 1) / 2) + var_loss


# 保存的 checkpoint
def save_checkpoint(model, optimizer, epoch, loss, filename="checkpoint.pth"):
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "loss": loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved at epoch {epoch}")


def load_checkpoint(model, optimizer, filename="checkpoint.pth"):
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint["model_state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    epoch = checkpoint["epoch"]
    loss = checkpoint["loss"]
    print(f"Checkpoint loaded: Resuming from epoch {epoch} with loss {loss}")
    return epoch, loss


def lr_lambda(epoch):
    if epoch < warmup_epochs:
        return epoch / warmup_epochs
    else:
        return 0.5 * (
            1
            + math.cos((epoch - warmup_epochs) / (num_epochs - warmup_epochs) * math.pi)
        )
