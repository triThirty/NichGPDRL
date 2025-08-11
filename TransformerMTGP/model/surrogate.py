import networkx as nx
import matplotlib.pyplot as plt
from networkx.drawing.nx_agraph import graphviz_layout
import numpy as np
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_networkx
from torch_geometric.loader import DataLoader

from TransformerMTGP.src.classes.individual import Individual
from TransformerMTGP.util.functions import positional_encoding, list_net_loss

lr_deduction = 0.9
epoch = 100
train_batch_size = 20


def surrogate_train(
    population,
    model,
    optimizer,
    device=None,
):
    embedding_layer = torch.nn.Embedding(17, 64, padding_idx=0).to(device)
    # embedding_layer.load_state_dict(torch.load("./TransformerMTGP/model/embedding.pth"))

    model.to(device)
    model.train()
    ind_list = []
    for id, ind in enumerate(population):
        i = Individual(str(ind[1]), str(ind[0]), id)
        i.true_fitness = torch.tensor(ind.fitness.values[0]).to(torch.float32)
        graph2 = Data(
            x=i.sequence_data, y=i.true_fitness, edge_index=i.sequence_edge
        ).to(device)
        graph1 = Data(x=i.route_data, y=i.true_fitness, edge_index=i.route_edge).to(
            device
        )

        combined_x = torch.cat([graph1.x, graph2.x], dim=0)
        indices = torch.nonzero(combined_x[:, 1:] == 1, as_tuple=True)[1]

        x_embedding = embedding_layer(indices + 1)
        position_embedding = positional_encoding(combined_x.shape[0], 64, device)

        x_pos_embedding = (x_embedding + position_embedding).clone().detach()

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
    )


def new_surrogate_train(
    population,
    model,
    optimizer,
    device=None,
):
    model.to(device)
    model.train()
    ind_list = []
    for id, ind in enumerate(population):
        i = Individual(str(ind[1]), str(ind[0]), id)
        i.true_fitness = torch.tensor(ind.fitness.values[0]).to(torch.float32)
        graph2 = Data(
            x=i.sequence_data, y=i.true_fitness, edge_index=i.sequence_edge
        ).to(device)
        graph1 = Data(x=i.route_data, y=i.true_fitness, edge_index=i.route_edge).to(
            device
        )

        combined_x = torch.cat([graph1.x, graph2.x], dim=0)
        segement_ids = torch.tensor(
            [1] * graph1.x.size(0) + [2] * graph2.x.size(0), dtype=torch.long
        ).to(device)
        indices = torch.nonzero(combined_x[:, 1:] == 1, as_tuple=True)[1] + 1

        combined_edge_index = torch.cat(
            [graph1.edge_index, graph2.edge_index + graph1.x.size(0)], dim=1
        )
        ind_list.append(
            Data(
                x=indices,
                edge_index=combined_edge_index,
                y=graph1.y,
                segement_ids=segement_ids,
            )
        )
    training_loader = DataLoader(ind_list, batch_size=train_batch_size, shuffle=True)
    training_loss, validation_loss = model.mytraining(
        list_net_loss,
        optimizer,
        training_loader,
        training_loader,
        epoch=epoch,
    )


def surrogate_evaluate(population, model, device):
    model.to(device)
    model.eval()
    for ind in population:
        i = Individual(str(ind[1]), str(ind[0]))
        graph2 = Data(x=i.sequence_data, edge_index=i.sequence_edge).to(device)
        graph1 = Data(x=i.route_data, edge_index=i.route_edge).to(device)

        combined_x = torch.cat([graph1.x, graph2.x], dim=0)
        segement_ids = torch.tensor(
            [1] * graph1.x.size(0) + [2] * graph2.x.size(0), dtype=torch.long
        ).to(device)
        indices = torch.nonzero(combined_x[:, 1:] == 1, as_tuple=True)[1] + 1

        combined_edge_index = torch.cat(
            [graph1.edge_index, graph2.edge_index + graph1.x.size(0)], dim=1
        )
        ind_data = Data(
            x=indices,
            edge_index=combined_edge_index,
            y=graph1.y,
            segement_ids=segement_ids,
        )
        output, score_vector = model(ind_data, is_batch=False)
        ind.score = output.clone().detach().item()
        ind.score_vector = score_vector.clone().detach().cpu().numpy()

        mask1 = ind_data.segement_ids == 1
        mask2 = ind_data.segement_ids == 2
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
