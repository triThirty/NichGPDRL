import torch
import torch.nn as nn
import numpy as np

import torch.nn.utils.parametrizations as P

from torch_geometric.nn import (
    global_add_pool,
    GATConv,
)

# import sys
# from pathlib import Path

# sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from TransformerMTGP.model.transformer import (
    TransformerEncoder,
    TransformerEncoderLayer,
)


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
            encoder_layers = TransformerEncoderLayer(
                d_model=self.feature_dim_size,
                nhead=self.num_heads,
                norm_first=True,
                dim_feedforward=self.ff_hidden_size,
                dropout=0.5,
                batch_first=True,
                # bias=False,
            )
            self.ugformer_layers.append(
                TransformerEncoder(encoder_layers, 1, enable_nested_tensor=False)
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

        self.attention_weights = []

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
                self.attention_weights = layer.attention_weights
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
            return x, score_vector, self.attention_weights

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


def mytraining(
    model,
    cost_func,
    optimizer,
    training_batch,
    validation_batch,
    epoch,
):
    times = 0
    epoch_times = epoch
    validation_loss = []
    training_loss = []
    early_stopping_times = 0
    while times <= epoch_times:
        cumulation_training_loss = 0.0
        model.train()
        for x, y, segment in training_batch:
            if len(x) == 1:
                break
            optimizer.zero_grad()
            output_data = model.forward(x, segment)
            target_data = y.view(-1, 1)
            training_loss_value = cost_func(output_data, target_data)

            training_loss_value.backward()
            optimizer.step()
            cumulation_training_loss = (
                cumulation_training_loss + training_loss_value.item()
            )
        # training_loss.append(cumulation_training_loss / len(training_batch))

        cumulation_validation_loss = 0.0
        model.eval()
        with torch.no_grad():
            for x, y, segment in validation_batch:
                if len(x) == 1:
                    break
                validation_outputs = model.forward(x, segment)
                validation_loss_value = cost_func(validation_outputs, y.view(-1, 1))
                cumulation_validation_loss += validation_loss_value.item()
        avg_validation_loss = cumulation_validation_loss / (
            len(validation_batch) * len(validation_batch.dataset)
        )
        avg_training_loss = cumulation_training_loss / (
            len(training_batch) * len(training_batch.dataset)
        )
        validation_loss.append(avg_validation_loss)
        diffs = np.diff(validation_loss)

        if len(diffs) > 3 and np.all(diffs[-3:] > 0):
            print(f"Early Stop, the validation loss value is: {avg_validation_loss}")
            break

        # times += 1
        # if times % 5 == 0:
        #     print("The training loss value is:", avg_training_loss)
        #     print("The validation loss value is:", avg_validation_loss)
    return training_loss, validation_loss


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
