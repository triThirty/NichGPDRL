# import networkx as nx
# import matplotlib.pyplot as plt
# from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import torch
from torch_geometric.data import Data
from torch.nn.utils.rnn import pad_sequence

from torch.utils.data import TensorDataset, DataLoader, Subset

from TransformerMTGP.src.classes.individual import Individual
from TransformerMTGP.util.functions import positional_encoding, list_net_loss
from TransformerMTGP.model.model import mytraining

lr_deduction = 0.9
epoch = 100
train_batch_size = 40


def construct_tensor_dataset(population, device):
    x_list = []
    y_list = []
    segment_list = []
    for id, ind in enumerate(population):
        i = Individual(str(ind[1]), str(ind[0]), id)
        i.true_fitness = torch.tensor(ind.fitness.values[0]).to(torch.float32)
        combined_x = torch.cat([i.route_data, i.sequence_data], dim=0)
        indices = torch.nonzero(combined_x[:, 1:] == 1, as_tuple=True)[1] + 1
        segement_ids = torch.tensor(
            [1] * i.route_data.size(0) + [2] * i.sequence_data.size(0), dtype=torch.long
        ).to(device)
        x_list.append(indices)
        y_list.append(i.true_fitness)
        segment_list.append(segement_ids)

    padded_x = pad_sequence(x_list, batch_first=True)
    padded_segment = pad_sequence(segment_list, batch_first=True)

    dataset = TensorDataset(padded_x, torch.tensor(y_list), padded_segment)
    return dataset


def new_surrogate_train(
    training_dataset,
    validation_dataset,
    model,
    optimizer,
    device=None,
):
    model.to(device)
    model.train()
    train_dataset = construct_tensor_dataset(training_dataset, device)
    validation_dataset = construct_tensor_dataset(validation_dataset, device)

    training_loader = DataLoader(
        train_dataset, batch_size=train_batch_size, shuffle=True
    )
    validation_loader = DataLoader(
        validation_dataset, batch_size=train_batch_size, shuffle=True
    )
    training_loss, validation_loss = mytraining(
        model,
        list_net_loss,
        optimizer,
        training_loader,
        validation_loader,
        epoch=epoch,
    )


def surrogate_evaluate(population, model, device):
    model.to(device)
    model.eval()
    for ind in population:
        i = Individual(str(ind[1]), str(ind[0]))
        combined_x = torch.cat([i.route_data, i.sequence_data], dim=0)
        indices = torch.nonzero(combined_x[:, 1:] == 1, as_tuple=True)[1] + 1
        segement_ids = torch.tensor(
            [1] * i.route_data.size(0) + [2] * i.sequence_data.size(0), dtype=torch.long
        ).to(device)

        output, score_vector = model(indices, segement_ids, is_batch=False)
        ind.score = output.clone().detach().item()
        ind.score_vector = score_vector.clone().detach().cpu().numpy()

        mask1 = segement_ids == 1
        mask2 = segement_ids == 2
        l_score_vector = ind.score_vector[mask1.clone().detach().cpu().numpy()]
        r_score_vector = ind.score_vector[mask2.clone().detach().cpu().numpy()]

        ind.l_min = np.argmin(l_score_vector)
        ind.l_max = np.argmax(l_score_vector)
        ind.r_min = np.argmin(r_score_vector)
        ind.r_max = np.argmax(r_score_vector)

        # G = to_networkx(ind_data, to_undirected=False)
        # node_colors = ["skyblue" for i in G.nodes]
        # node_colors[ind.l_min] = "red"
        # node_colors[ind.l_max] = "green"
        # node_colors[ind.r_min + graph1.x.size(0)] = "red"
        # node_colors[ind.r_max + graph1.x.size(0)] = "green"
        # # node_colors[ind.max_score_node_index] = "green"
        # pos = graphviz_layout(G, prog="dot")
        # plt.figure(figsize=(8, 6))
        # nx.draw(
        #     G,
        #     pos,
        #     with_labels=True,
        #     # labels=node_labels,
        #     node_color=node_colors,
        #     node_size=200,
        # )
        # plt.title("GNN Input Graph")
        # plt.show()
