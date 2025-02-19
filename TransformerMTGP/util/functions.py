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


def list_net_loss(output, target):
    scores = output.view(-1)
    sorted_indices = torch.argsort(target.view(-1), descending=True, dim=-1)
    sorted_scores = torch.gather(scores, dim=-1, index=sorted_indices)
    loss = -torch.sum(
        sorted_scores - torch.logcumsumexp(sorted_scores, dim=-1), dim=-1
    ) / scores.size(0)
    return loss


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
