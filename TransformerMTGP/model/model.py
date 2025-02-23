import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import torch.nn.utils.parametrizations as P

from torch_geometric.nn import (
    global_add_pool,
    GATConv,
)


class MyNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_heads, num_layers, lr):
        super(MyNN, self).__init__()

        self.lr = lr

        self.feature_dim_size = input_size
        self.ff_hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_size = output_size

        self.lst_gnn = torch.nn.ModuleList()

        self.ugformer_layers = torch.nn.ModuleList()

        for _layer in range(1):
            encoder_layers = TransformerEncoderLayer(
                d_model=self.feature_dim_size,
                nhead=self.num_heads,
                norm_first=True,
                dim_feedforward=self.ff_hidden_size,
                dropout=0.3,
                batch_first=True,
                # bias=False,
            )
            self.ugformer_layers.append(
                TransformerEncoder(encoder_layers, 1, enable_nested_tensor=True)
            )
        for _ in range(self.num_layers):
            self.lst_gnn.append(
                GATConv(
                    in_channels=self.feature_dim_size,
                    out_channels=self.feature_dim_size,
                    heads=self.num_heads,
                    concat=False,
                    dropout=0.3,
                )
            )

        self.predictions = torch.nn.ModuleList()
        self.predictions.append(
            P.spectral_norm(nn.Linear(self.feature_dim_size, self.feature_dim_size))
        )
        self.predictions.append(nn.BatchNorm1d(self.feature_dim_size))
        for _ in range(3):
            self.predictions.append(nn.LeakyReLU())
            self.predictions.append(
                P.spectral_norm(nn.Linear(self.feature_dim_size, self.feature_dim_size))
            )
            self.predictions.append(nn.BatchNorm1d(self.feature_dim_size))
        self.predictions.append(nn.LeakyReLU())
        self.final = torch.nn.ModuleList()
        self.final.append(
            P.spectral_norm(nn.Linear(self.feature_dim_size, self.output_size))
        )

    def forward(self, batch_data, is_batch=True):
        if is_batch:
            num_nodes_per_graph = batch_data.batch.bincount().tolist()
            split_x = torch.split(batch_data.x, num_nodes_per_graph)
        else:
            split_x = [batch_data.x]
        padded_dataset = pad_sequence(split_x, batch_first=True)
        self.src_key_padding_mask = self.src_mask(padded_dataset)
        for layer in self.ugformer_layers:
            x = layer(padded_dataset, src_key_padding_mask=self.src_key_padding_mask)

        filtered_x = x[~self.src_key_padding_mask]
        batch_data.x = filtered_x

        x = batch_data.x
        for layer in self.lst_gnn:
            x = x + layer(x, batch_data.edge_index)

        x = global_add_pool(x, batch_data.batch)

        for layer in self.predictions:
            x = layer(x)

        x = self.final[0](x)

        return x

    def src_mask(self, x):
        padding_mask = torch.all(x == 0, dim=-1)
        return padding_mask

    def mytraining(
        self,
        cost_func,
        optimizer,
        training_batch,
        validation_batch,
        epoch,
        lr_deduction=0.9,
        toolbox=None,
        population=None,
        rd=None,
        reduce_scheduler=None,
    ):
        times = 0
        early_stopping_times = 0
        epoch_times = epoch
        validation_loss = []
        training_loss = []
        for param_group in optimizer.param_groups:
            param_group["lr"] = self.lr
        while times <= epoch_times:
            cumulation_training_loss = 0.0
            self.train()
            for batch_data in training_batch:
                optimizer.zero_grad()
                output_data = self.forward(batch_data)
                target_data = batch_data.y.view(-1, 1)
                training_loss_value = cost_func(output_data, target_data)

                training_loss_value.backward()
                optimizer.step()
                cumulation_training_loss = (
                    cumulation_training_loss + training_loss_value.item()
                )
            training_loss.append(cumulation_training_loss / len(training_batch))

            times += 1
            if times % 5 == 0:
                print("The training loss value is:", training_loss_value.item())
            if epoch_times > epoch:
                reduce_scheduler.step(training_loss[-1])
            if epoch_times > 200:
                continue
            elif times > epoch_times and sum(training_loss[-5:]) / 5 > 1.5:
                epoch_times += 20
                print(
                    f"The average loss value of latest 5 epochs is {sum(training_loss[-5:]) / 5}. Add 10 more epochs to {epoch_times}"
                )
                # if epoch_times > 100:
                #     rd["seed"] = np.random.randint(2000000000)
                #     fitnesses = toolbox.multiProcess(toolbox.evaluate, population, rd)
                #     for ind, fit in zip(population, fitnesses):
                #         ind.fitness.values = fit[0]
                #         ind.num_calculation = fit[1]

        return training_loss, validation_loss
