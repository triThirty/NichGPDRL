import torch
import torch.nn as nn
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from bertviz import head_view


# ============================
# 1. 自定义 EncoderLayer
# ============================
class CustomTransformerEncoderLayer(nn.TransformerEncoderLayer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, src, src_mask=None, src_key_padding_mask=None, need_weights=True):
        # 直接调用 MultiheadAttention, 强制返回 attn_weights
        src2, attn_weights = self.self_attn(
            src,
            src,
            src,
            attn_mask=src_mask,
            key_padding_mask=src_key_padding_mask,
            need_weights=True,
            average_attn_weights=False,  # 保持 [batch, num_heads, seq_len, seq_len]
        )
        src = src + self.dropout1(src2)
        src = self.norm1(src)
        src2 = self.linear2(self.dropout(self.activation(self.linear1(src))))
        src = src + self.dropout2(src2)
        src = self.norm2(src)
        return src, attn_weights


# ============================
# 2. 自定义 Encoder，收集所有层的 attn
# ============================
class CustomTransformerEncoder(nn.Module):
    def __init__(self, encoder_layer, num_layers):
        super().__init__()
        self.layers = nn.ModuleList([encoder_layer for _ in range(num_layers)])
        self.num_layers = num_layers

    def forward(self, src, mask=None, src_key_padding_mask=None):
        attn_list = []
        output = src
        for layer in self.layers:
            output, attn = layer(
                output, src_mask=mask, src_key_padding_mask=src_key_padding_mask
            )
            attn_list.append(attn.detach().cpu())
        return output, attn_list


# ============================
# 3. 示例输入
# ============================
d_model = 16
nhead = 2
num_layers = 2
seq_len = 6
batch_size = 1

encoder_layer = CustomTransformerEncoderLayer(
    d_model=d_model, nhead=nhead, dim_feedforward=64
)
transformer_encoder = CustomTransformerEncoder(encoder_layer, num_layers=num_layers)

# 模拟输入
x = torch.rand(seq_len, batch_size, d_model)  # [seq_len, batch, d_model]
output, attn_list = transformer_encoder(x)

print(f"Collected {len(attn_list)} layers of attention.")

# ============================
# 4. 可视化 with bertviz
# ============================
tokens = ["[CLS]", "I", "like", "PyTorch", "very", "much"]  # 示例 tokens
# bertviz 需要的输入: attention=[layer][batch, head, seq, seq]
# 我们这里只看第一层第一条样本
attentions = [attn_list[i][0].numpy() for i in range(len(attn_list))]

head_view(attentions, tokens)
