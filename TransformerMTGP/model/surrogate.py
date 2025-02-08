import torch
from torch_geometric.data import Data

# from sklearn.model_selection import train_test_split
from torch_geometric.loader import DataLoader
from torch.optim import Adam

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from Summer.src.classes.individual import Individual
from Summer.util.functions import positional_encoding, list_net_loss


lr_deduction = 1
epoch = 20
train_batch_size = 20

embedding_layer = torch.nn.Embedding(53, 64, padding_idx=0)
embedding_layer.load_state_dict(torch.load("./TransformerMTGP/model/embedding.pth"))
# embedding_layer.eval()


def surrogate_train(population, model, optimizer):
    model.train()
    ind_list = []
    for id, ind in enumerate(population):
        i = Individual(str(ind[1]), str(ind[0]), id)
        i.true_fitness = torch.tensor(ind.fitness.values[0]).to(torch.float32)
        graph2 = Data(x=i.sequence_data, y=i.true_fitness, edge_index=i.sequence_edge)
        graph1 = Data(x=i.route_data, y=i.true_fitness, edge_index=i.route_edge)

        combined_x = torch.cat([graph1.x, graph2.x], dim=0)
        indices = torch.nonzero(combined_x[:, 1:] == 1, as_tuple=True)[1]

        x_embedding = embedding_layer(indices)
        position_embedding = positional_encoding(combined_x.shape[0], 64)

        x_pos_embedding = (x_embedding + position_embedding).detach()

        combined_edge_index = torch.cat(
            [graph1.edge_index, graph2.edge_index + graph1.x.size(0)], dim=1
        )
        ind_list.append(
            Data(x=x_pos_embedding, edge_index=combined_edge_index, y=graph1.y)
        )
    # training_dataset, validation_dataset = train_test_split(ind_list, test_size=0.0)
    training_loader = DataLoader(ind_list, batch_size=train_batch_size, shuffle=True)
    training_loss, validation_loss = model.mytraining(
        list_net_loss,
        optimizer,
        training_loader,
        training_loader,
        epoch=epoch,
        lr_deduction=lr_deduction,
    )


def surrogate_evaluate(population, model):
    model.eval()
    for ind in population:
        i = Individual(str(ind[1]), str(ind[0]))
        graph2 = Data(x=i.sequence_data, edge_index=i.sequence_edge)
        graph1 = Data(x=i.route_data, edge_index=i.route_edge)

        combined_x = torch.cat([graph1.x, graph2.x], dim=0)
        indices = torch.nonzero(combined_x[:, 1:] == 1, as_tuple=True)[1]

        x_embedding = embedding_layer(indices)
        position_embedding = positional_encoding(combined_x.shape[0], 64)

        x_pos_embedding = (x_embedding + position_embedding).detach()

        combined_edge_index = torch.cat(
            [graph1.edge_index, graph2.edge_index + graph1.x.size(0)], dim=1
        )
        ind_data = Data(x=x_pos_embedding, edge_index=combined_edge_index, y=graph1.y)
        output = model(ind_data, is_batch=False)
        ind.score = output.detach().item()
