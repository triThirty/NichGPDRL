import torch
import torch.nn as nn
import numpy as np
from torch import Tensor

import torch.nn.utils.parametrizations as P
from typing import Any, Callable, Optional, Union
import torch.nn.functional as F
from torch.nn.modules.module import Module
from torch.nn.modules.activation import MultiheadAttention
from torch.nn.modules.linear import Linear
from torch.nn.modules.dropout import Dropout
from torch.nn.modules.normalization import LayerNorm
from torch.nn.modules.container import ModuleList
import copy

from torch_geometric.nn import global_add_pool, GATConv


class MyNN(nn.Module):
    def __init__(
        self, input_size, hidden_size, output_size, num_heads, num_layers, shared_emb
    ):
        super(MyNN, self).__init__()

        self.embedding_layer = shared_emb

        self.feature_dim_size = input_size
        self.ff_hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.output_size = output_size

        self.lst_gnn = torch.nn.ModuleList()

        self.ugformer_layers = torch.nn.ModuleList()

        for _layer in range(1):
            encoder_layers = nn.TransformerEncoderLayer(
                d_model=self.feature_dim_size,
                nhead=self.num_heads,
                norm_first=True,
                dim_feedforward=self.ff_hidden_size,
                dropout=0.5,
                batch_first=True,
                # bias=False,
            )
            self.ugformer_layers.append(
                nn.TransformerEncoder(encoder_layers, 1, enable_nested_tensor=False)
            )
        for _ in range(self.num_layers):
            self.lst_gnn.append(
                GATConv(
                    in_channels=self.feature_dim_size,
                    out_channels=self.feature_dim_size,
                    heads=self.num_heads,
                    concat=False,
                    dropout=0.5,
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

    # split_x: the primitive set index
    def forward(self, x, segment, is_batch=True):
        if is_batch:
            batch = self.get_batch(x)
            post_processed_data = self.embedding_layer(x, segment)
            src_key_padding_mask = self.src_mask(x)
        else:
            post_processed_data = self.embedding_layer(x, segment, is_batch=is_batch)
            src_key_padding_mask = None
            batch = None
        for layer in self.ugformer_layers:
            if torch.is_grad_enabled():
                x, score_vector = layer(
                    post_processed_data, src_key_padding_mask=src_key_padding_mask
                )
            elif not torch.is_grad_enabled():
                x = layer(
                    post_processed_data, src_key_padding_mask=src_key_padding_mask
                )
        del post_processed_data

        if is_batch:
            valid_mask = ~src_key_padding_mask
            filtered_x = x[valid_mask]
        else:
            filtered_x = x.squeeze(0)
        x = filtered_x

        x = global_add_pool(x, batch)

        for layer in self.predictions:
            x = layer(x)

        x = self.final[0](x)

        if is_batch:
            return x
        else:
            return x, score_vector

    def src_mask(self, x):
        padding_mask = x == 0
        return padding_mask

    def get_batch(self, x):
        batch_list = []
        for i, t in enumerate(x):
            non_zero_dim = torch.count_nonzero(t).item()
            batch_tensor = torch.full((non_zero_dim,), fill_value=i, dtype=torch.long)
            batch_list.append(batch_tensor)
        batch = torch.cat(batch_list)
        return batch


class SharedEmbeddings(nn.Module):
    def __init__(self):
        super().__init__()
        self.token_emb = nn.Embedding(17, 64, padding_idx=0)
        self.pos_emb = nn.Embedding(5000, 64)
        self.seg_emb = nn.Embedding(3, 64, padding_idx=0)
        # self.LayerNorm = nn.LayerNorm(64)
        # self.dropout = nn.Dropout(0.3)

    def forward(self, input_ids, segment_ids, is_batch=True):
        if is_batch:
            seq_len = input_ids.size(1)
        else:
            seq_len = input_ids.size(0)
        position_ids = torch.arange(
            seq_len, dtype=torch.long, device=input_ids.device
        ).unsqueeze(0)

        token_emb = self.token_emb(input_ids)
        pos_emb = self.pos_emb(position_ids)
        seg_emb = self.seg_emb(segment_ids)

        embeddings = token_emb + pos_emb + seg_emb
        # return self.dropout(self.LayerNorm(embeddings))
        return embeddings


if __name__ == "__main__":
    loaded_checkpoint = torch.load(
        "data/046cd36bd4f_single_model_0.8/checkpoint_30.pth"
        # "data/ab7e39e061a_single_model_nesi_0.5/checkpoint_30.pth"
    )
    shared_emb = SharedEmbeddings()
    shared_emb.load_state_dict(loaded_checkpoint["embedding_state_dict"])
    transformer_model = MyNN(64, 1024, 1, 8, 3, shared_emb)
    transformer_model.load_state_dict(loaded_checkpoint["model_state_dict"])
    transformer_model.eval()

