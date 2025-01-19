import sys
import MTGP.LoadIndividual as mtload
import pygraphviz as pgv
from deap import gp
import pickle5 as pickle
import matplotlib.pyplot as plt
import networkx as nx
from ann_visualizer.visualize import ann_viz
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

from graphviz import Digraph
import keras
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Dropout, Flatten
import json


'''
Neural Networks
'''
class build_network_small(nn.Module):
    def __init__(self, input_size, output_size):
        super(build_network_small, self).__init__()
        # size of layers
        layer_1 = 16
        layer_2 = 16
        layer_3 = 16
        layer_4 = 8
        layer_5 = 8
        # FCNN
        self.fc1 = nn.Linear(input_size, layer_1)
        self.fc2 = nn.Linear(layer_1, layer_2)
        self.fc3 = nn.Linear(layer_2, layer_3)
        self.fc4 = nn.Linear(layer_3, layer_4)
        self.fc5 = nn.Linear(layer_4, layer_5)
        self.fc6 = nn.Linear(layer_5, output_size)
        # activation functions
        self.tanh = nn.Tanh()
        self.instancenorm = nn.InstanceNorm1d(input_size)
        self.flatten = nn.Flatten()
        # Huber loss function
        self.loss_func = F.smooth_l1_loss

    def forward(self, x, *args):
        #print('original',x)
        x = self.instancenorm(x)
        #print('normalized',x)
        x = self.flatten(x)
        #print('flattened',x)
        x = self.fc1(x)
        x = self.tanh(x)
        x = self.fc2(x)
        x = self.tanh(x)
        x = self.fc3(x)
        x = self.tanh(x)
        x = self.fc4(x)
        x = self.tanh(x)
        x = self.fc5(x)
        x = self.tanh(x)
        x = self.fc6(x)
        return x

'''
Neural Networks
'''
class network_validated(nn.Module):
    def __init__(self, input_size, output_size):
        super(network_validated, self).__init__()
        self.lr = 0.001
        self.input_size = input_size
        self.output_size = output_size
        # for slicing the data
        self.no_size = 3
        self.pt_size = 6
        self.remaining_pt_size = 11
        self.ttd_slack_size = 16
        # FCNN parameters
        layer_1 = 48
        layer_2 = 36
        layer_3 = 36
        layer_4 = 24
        layer_5 = 24
        layer_6 = 12
        # normalization modules
        self.normlayer_no = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_pt = nn.Sequential(
                                nn.InstanceNorm1d(3),
                                nn.Flatten()
                                )
        self.normlayer_remaining_pt = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        self.normlayer_ttd_slack = nn.Sequential(
                                nn.InstanceNorm1d(5),
                                nn.Flatten()
                                )
        # shared layers of machines
        self.subsequent_module = nn.Sequential(
                                nn.Linear(self.input_size, layer_1),
                                nn.Tanh(),
                                nn.Linear(layer_1, layer_2),
                                nn.Tanh(),
                                nn.Linear(layer_2, layer_3),
                                nn.Tanh(),
                                nn.Linear(layer_3, layer_4),
                                nn.Tanh(),
                                nn.Linear(layer_4, layer_5),
                                nn.Tanh(),
                                nn.Linear(layer_5, layer_6),
                                nn.Tanh(),
                                nn.Linear(layer_6, output_size)
                                )
        # Huber loss function
        self.loss_func = F.smooth_l1_loss
        # the universal network for all scheudling agents
        self.network = nn.ModuleList([self.normlayer_no, self.normlayer_pt, self.normlayer_remaining_pt, self.normlayer_ttd_slack, self.subsequent_module])
        # accompanied by optimizer
        self.optimizer = optim.SGD(self.network.parameters(), lr=self.lr, momentum = 0.9)

def ann_viz_meng_sequencing(input_size, output_size, parameters, view=True, filename="network.gv", title="My Neural Network"):
    """Vizualizez a Sequential model.

    # Arguments
        model: A Keras model instance.

        view: whether to display the model after generation.

        filename: where to save the vizualization. (a .gv file)

        title: A title for the graph
    """

    input_layer = input_size
    hidden_layers_nr = 6
    layer_types = []
    hidden_layers = [48,36,36,24,24,12]
    weights_all = []
    output_layer = output_size
    limit_nodes = 4

    for l in range(len(parameters[0]["params"])):
        if l%2 == 0:
            weights = []
            data = parameters[0]["params"][l].data
            # print()
            # print('-------layer ' + str(l) + '-------')
            for i in range(len(data)):
                weights.append(data[i])
                # print('-------data ' + str(i) + '-------')
                # print(data[i])
            weights_all.append(weights)

    last_layer_nodes = input_layer
    nodes_up = input_layer
    # g = nx.Graph()
    g = Digraph('g', filename=filename)
    n = 0
    g.graph_attr.update(splines="true", nodesep='0.1', ranksep='1', rankdir="LR")
    # g.graph_attr.update(splines="false", nodesep='1', ranksep='2')
    #Input Layer
    with g.subgraph(name='cluster_input') as c:
        the_label = ''
        c.attr(color='white')
        # If hidden_layers[i] > 10, dont include all
        if input_layer > limit_nodes:
            the_label += " (+" + str(input_layer - limit_nodes) + ")"
            input_layer = limit_nodes
            last_layer_nodes = input_layer
            nodes_up = input_layer
        for i in range(0, input_layer):
            n += 1
            c.node(str(n))
            c.attr(label=the_label, fontsize="30")
            c.attr(rank='same')
            c.node_attr.update(color="#800080", style="filled", fontcolor="#800080", shape="circle")
        c.attr(label='Input Layer', labelloc="bottom", fontsize="30")


    for i in range(0, hidden_layers_nr):
        with g.subgraph(name="cluster_"+str(i+1)) as c:
            c.attr(color='white')
            c.attr(rank='same')
            #If hidden_layers[i] > 10, dont include all
            the_label = ""
            if hidden_layers[i] > limit_nodes:
                the_label += " (+"+str(hidden_layers[i] - limit_nodes)+")"
                hidden_layers[i] = limit_nodes
            c.attr(labeljust="right", labelloc="b", label=the_label, fontsize="30")
            for j in range(0, hidden_layers[i]):
                n += 1
                c.node(str(n), shape="circle", style="filled", color="#3498db", fontcolor="#3498db")
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    pos_x = j
                    pos_y = h - (nodes_up - last_layer_nodes + 1)
                    w = round(float(weights_all[i][pos_x][pos_y]),2)
                    if w > 0:
                        g.edge(str(h), str(n), label=str(w), labelangle="0", labeldistance='0.2', fontsize="30",
                               arrowhead="none", color="#FFA500")
                        # g.edge(str(h), str(n), label=str(w), fontsize="20", arrowhead="none", color="#FFA500", labelangle="0", labeldistance='0.2', labelfloat='true')
                    else:
                        g.edge(str(h), str(n), label=str(w), labelangle="0", labeldistance='0.8', fontsize="30",
                               style="dashed", arrowhead="none",
                               color="#008000")
                    # g.edge(str(h), str(n), weight=0.1, color="#3498db")
            last_layer_nodes = hidden_layers[i]
            nodes_up += hidden_layers[i]


    with g.subgraph(name='cluster_output') as c:
        c.attr(color='white')
        c.attr(rank='same')
        c.attr(labeljust="1")
        for i in range(1, output_layer+1):
            n += 1
            c.node(str(n), shape="circle", style="filled", color="#e74c3c", fontcolor="#e74c3c")
            for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                pos_x = j
                pos_y = h - (nodes_up - last_layer_nodes + 1)
                w = round(float(weights_all[i][pos_x][pos_y]), 2)
                if w > 0:
                    g.edge(str(h), str(n), label=str(w), labelangle="0", labeldistance='0.2', fontsize="30",
                           arrowhead="none", color="#FFA500")
                    # g.edge(str(h), str(n), label=str(w), fontsize="20", arrowhead="none", color="#FFA500", labelangle="0", labeldistance='0.2', labelfloat='true')
                else:
                    g.edge(str(h), str(n), label=str(w), labelangle="-45", labeldistance='0.8', fontsize="30",
                           style="dashed", arrowhead="none",
                           color="#008000")
        c.attr(label='Output Layer', labelloc="bottom", fontsize="30")
        c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle")

    g.attr(arrowShape="none")
    # g.edge_attr.update(arrowhead="none", color="#707070")
    if view == True:
        g.view()

def ann_viz_meng_routing(input_size, output_size, parameters, view=True, filename="network.gv", title="My Neural Network"):
    """Vizualizez a Sequential model.

    # Arguments
        model: A Keras model instance.

        view: whether to display the model after generation.

        filename: where to save the vizualization. (a .gv file)

        title: A title for the graph
    """

    input_layer = input_size
    hidden_layers_nr = 5
    layer_types = []
    hidden_layers = [16,16,16,8,8]
    weights_all = []
    output_layer = output_size
    limit_nodes = 4

    weights_all = parameters


    last_layer_nodes = input_layer
    nodes_up = input_layer
    # g = nx.Graph()
    g = Digraph('g', filename=filename)
    n = 0
    g.graph_attr.update(splines="true", nodesep='0.1', ranksep='1', rankdir="LR")
    # g.graph_attr.update(splines="true", nodesep='0.3', ranksep='1.5', rankdir="LR")
    # g.graph_attr.update(splines="false", nodesep='1', ranksep='2')
    #Input Layer
    with g.subgraph(name='cluster_input') as c:
        the_label = ''
        # the_label = title+'\n\n\n\nInput Layer'
        c.attr(color='white')
        # If hidden_layers[i] > 10, dont include all
        # the_label = ""
        if input_layer > limit_nodes:
            the_label += " (+" + str(input_layer - limit_nodes) + ")"
            input_layer = limit_nodes
            last_layer_nodes = input_layer
            nodes_up = input_layer
        for i in range(0, input_layer):
            n += 1
            c.node(str(n))
            c.attr(label=the_label, fontsize="30", labelloc="bottom")
            c.attr(rank='same')
            c.node_attr.update(color="#800080", style="filled", fontcolor="#800080", shape="circle")
        c.attr(label='Input Layer', labelloc="bottom", fontsize="30")


    for i in range(0, hidden_layers_nr):
        with g.subgraph(name="cluster_"+str(i+1)) as c:
            c.attr(color='white')
            c.attr(rank='same')
            #If hidden_layers[i] > 10, dont include all
            the_label = ""
            if hidden_layers[i] > limit_nodes:
                the_label += " (+"+str(hidden_layers[i] - limit_nodes)+")"
                hidden_layers[i] = limit_nodes
            c.attr(labeljust="right", labelloc="b", label=the_label, fontsize="30")
            for j in range(0, hidden_layers[i]):
                n += 1
                c.node(str(n), shape="circle", style="filled", color="#3498db", fontcolor="#3498db")
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    pos_x = j
                    pos_y = h - (nodes_up - last_layer_nodes + 1)
                    w = round(float(weights_all[i][pos_x][pos_y]),2)
                    if w > 0:
                        # linewidth = "setlinewidth(" + str(w) + ")"
                        # g.edge(str(h), str(n), label=str(w), style="setlinewidth(2)", labelangle="0", labeldistance='0.2', fontsize="30", arrowhead="none", color="#FFA500")
                        g.edge(str(h), str(n), label=str(w), labelangle="0",
                               labeldistance='0.2', fontsize="30", arrowhead="none", color="#FFA500")
                        # g.edge(str(h), str(n), label=str(w), fontsize="20", arrowhead="none", color="#FFA500", labelangle="0", labeldistance='0.2', labelfloat='true')
                    else:
                        g.edge(str(h), str(n), label=str(w), labelangle="0", labeldistance='0.8', fontsize="30", style="dashed", arrowhead="none",
                               color="#008000")
                        # g.edge(str(h), str(n), label=str(w), fontsize="20", style="dashed", arrowhead="none", color="#008000", labelangle="0", labeldistance='0.8', labelfloat='true')
                    # g.edge(str(h), str(n), weight=0.1, color="#3498db")
            last_layer_nodes = hidden_layers[i]
            nodes_up += hidden_layers[i]


    with g.subgraph(name='cluster_output') as c:
        c.attr(color='white')
        c.attr(rank='same')
        c.attr(labeljust="1")
        for i in range(1, output_layer+1):
            n += 1
            c.node(str(n), shape="circle", style="filled", color="#e74c3c", fontcolor="#e74c3c")
            for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                pos_x = j
                pos_y = h - (nodes_up - last_layer_nodes + 1)
                w = round(float(weights_all[i][pos_x][pos_y]),2)
                if w > 0:
                    g.edge(str(h), str(n), label=str(w), labelangle="0", labeldistance='0.2', fontsize="30", arrowhead="none", color="#FFA500")
                    # g.edge(str(h), str(n), label=str(w), fontsize="20", arrowhead="none", color="#FFA500", labelangle="0", labeldistance='0.2', labelfloat='true')
                else:
                    g.edge(str(h), str(n), label=str(w), labelangle="-45", labeldistance='0.8', fontsize="30", style="dashed", arrowhead="none",
                           color="#008000")
                    # g.edge(str(h), str(n), label=str(w), fontsize="20", style="dashed", arrowhead="none", color="#008000", labelangle="0", labeldistance='0.8', labelfloat='true')
        c.attr(label='Output Layer', labelloc="bottom", fontsize="30")
        c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle")

    g.attr(arrowShape="none")
    # g.edge_attr.update(arrowhead="none", color="#707070")
    if view == True:
        g.view()

def ann_viz_meng(input_size, output_size, parameters, view=True, filename="network.gv", title="My Neural Network"):
    """Vizualizez a Sequential model.

    # Arguments
        model: A Keras model instance.

        view: whether to display the model after generation.

        filename: where to save the vizualization. (a .gv file)

        title: A title for the graph
    """

    input_layer = input_size
    hidden_layers_nr = 6
    layer_types = []
    hidden_layers = [48,36,36,24,24,12]
    weights_all = []
    output_layer = output_size

    for l in range(len(parameters[0]["params"])):
        if l%2 == 0:
            weights = []
            data = parameters[0]["params"][l].data
            # print()
            # print('-------layer ' + str(l) + '-------')
            for i in range(len(data)):
                weights.append(data[i])
                # print('-------data ' + str(i) + '-------')
                # print(data[i])
            weights_all.append(weights)

    last_layer_nodes = input_layer
    nodes_up = input_layer
    # g = nx.Graph()
    g = Digraph('g', filename=filename)
    n = 0
    g.graph_attr.update(splines="true", nodesep='1', ranksep='2')
    # g.graph_attr.update(splines="false", nodesep='1', ranksep='2')
    #Input Layer
    with g.subgraph(name='cluster_input') as c:
        the_label = title+'\n\n\n\nInput Layer'
        c.attr(color='white')
        for i in range(0, input_layer):
            n += 1
            c.node(str(n))
            c.attr(label=the_label)
            c.attr(rank='same')
            c.node_attr.update(color="#800080", style="filled", fontcolor="#800080", shape="circle")


    for i in range(0, hidden_layers_nr):
        with g.subgraph(name="cluster_"+str(i+1)) as c:
            c.attr(color='white')
            c.attr(rank='same')
            #If hidden_layers[i] > 10, dont include all
            the_label = ""
            # if (int(str(model.layers[i].output_shape).split(",")[1][1:-1]) > 10):
            #     the_label += " (+"+str(int(str(model.layers[i].output_shape).split(",")[1][1:-1]) - 10)+")"
            #     hidden_layers[i] = 10
            c.attr(labeljust="right", labelloc="b", label=the_label)
            for j in range(0, hidden_layers[i]):
                n += 1
                c.node(str(n), shape="circle", style="filled", color="#3498db", fontcolor="#3498db")
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    pos_x = j
                    pos_y = h - (nodes_up - last_layer_nodes + 1)
                    w = round(float(weights_all[i][pos_x][pos_y]),2)
                    if w > 0:
                        g.edge(str(h), str(n), label=str(w), fontname="Microsoft Yahei", arrowhead="none", color="#FFA500")
                    else:
                        g.edge(str(h), str(n), label=str(w), style="dashed", arrowhead="none", color="#008000")
                    # g.edge(str(h), str(n), weight=0.1, color="#3498db")
            last_layer_nodes = hidden_layers[i]
            nodes_up += hidden_layers[i]


    with g.subgraph(name='cluster_output') as c:
        c.attr(color='white')
        c.attr(rank='same')
        c.attr(labeljust="1")
        for i in range(1, output_layer+1):
            n += 1
            c.node(str(n), shape="circle", style="filled", color="#e74c3c", fontcolor="#e74c3c")
            for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                pos_x = j
                pos_y = h - (nodes_up - last_layer_nodes + 1)
                w = round(float(weights_all[i][pos_x][pos_y]),2)
                if w > 0:
                    g.edge(str(h), str(n), label=str(w), arrowhead="none", color="#FFA500")
                else:
                    g.edge(str(h), str(n), label=str(w), style="dashed", arrowhead="none", color="#008000")
        c.attr(label='Output Layer', labelloc="bottom")
        c.node_attr.update(color="#2ecc71", style="filled", fontcolor="#2ecc71", shape="circle")

    g.attr(arrowShape="none")
    # g.edge_attr.update(arrowhead="none", color="#707070")
    if view == True:
        g.view()


def ann_viz_meng_v1(input_size, output_size, model, view=True, filename="network.gv", title="My Neural Network"):

    input_layer = input_size
    hidden_layers_nr = 8
    layer_types = []
    hidden_layers = [25,48,36,36,24,24,12,4]
    output_layer = output_size

    # for l in range(len(parameters[0]["params"])):
    #     data = parameters[0]["params"][l].data
    #     print()
    #     print('-------layer ' + str(l) + '-------')
    #     for i in range(len(data)):
    #         print('-------data ' + str(i) + '-------')
    #         print(data[i])

    last_layer_nodes = input_layer
    nodes_up = input_layer
    g = nx.Graph()
    n = 0
    for i in range(0, hidden_layers_nr):
        if i == 0:
            the_label = title + '\n\n\n\nInput Layer'
            for j in range(0, input_layer):
                n += 1
                g.add_node(n, pos=(i,j*2), size = 1)
        elif i == 7:
            for j in range(1, output_layer + 1):
                n += 1
                g.add_node(n, pos=(i,j*2), size = 1)
                for h in range(nodes_up - last_layer_nodes + 1, nodes_up + 1):
                    g.add_edge(h, n, weight=1, capacity=15)
        else:
            for j in range(0, hidden_layers[i]):
                n += 1
                g.add_node(n, pos=(i,j*2), size = 1)
                for h in range(nodes_up - last_layer_nodes + 1 , nodes_up + 1):
                    g.add_edge(h, n, weight=7, capacity=15)
                    # g.edge(str(h), str(n), weight=0.1, color="#3498db")
            last_layer_nodes = hidden_layers[i]
            nodes_up += hidden_layers[i]

    pos = nx.get_node_attributes(g, 'pos')
    nx.draw(g, pos, with_labels=False)
    plt.show()


if __name__ == "__main__":

    # sequencing network
    # input_size = 25
    # output_size = 4
    # address_seed = "{}/sequencing_models/scenario_HH/run_0_MC_rwd1.pt"  # modified by mengxu
    # sys.path[0] = "/home/xume/IdeaProjects/Deep-reinforcement-learning-for-dynamic-scheduling-of-a-flexible-job-shop-master"
    # network = network_validated(input_size, output_size)
    # network.network.load_state_dict(torch.load(address_seed.format(sys.path[0])))
    # network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
    # parameters = network.optimizer.param_groups  #this store the weights of connections
    # ann_viz_meng_sequencing(25,4,parameters,filename="/home/xume/IdeaProjects/Deep-reinforcement-learning-for-dynamic-scheduling-of-a-flexible-job-shop-master/paintingAnalysis/sequencingnetwork.gv", title="Sequencing network")


    # routing network
    input_size = 9
    output_size = 2
    address_seed = "{}/routing_models/scenario_HH/run_0_small_state_dict3wc6m.pt"  # modified by mengxu
    sys.path[0] = "/home/xume/IdeaProjects/Deep-reinforcement-learning-for-dynamic-scheduling-of-a-flexible-job-shop-master"
    network = build_network_small(input_size, output_size)
    network.load_state_dict(torch.load(address_seed.format(sys.path[0])))
    network.eval()  # must have this if you're loading a model, unnecessray for loading state_dict
    parameters = [] #this store the weights of connections
    parameters.append(network.fc1.weight.data)
    parameters.append(network.fc2.weight.data)
    parameters.append(network.fc3.weight.data)
    parameters.append(network.fc4.weight.data)
    parameters.append(network.fc5.weight.data)
    parameters.append(network.fc6.weight.data)
    ann_viz_meng_routing(9,2,parameters,filename="/home/xume/IdeaProjects/Deep-reinforcement-learning-for-dynamic-scheduling-of-a-flexible-job-shop-master/paintingAnalysis/routingnetwork.gv", title="Routing network")


