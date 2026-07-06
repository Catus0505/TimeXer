import torch
import torch.nn as nn
import torch.nn.functional as F

from layers.Embed import PositionalEmbedding


class ChannelIndependentSelfAttention(nn.Module):
    """Standard self-attention applied to channel-local token sequences."""

    def __init__(self, d_model, n_heads, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True,
        )

    def forward(self, x, attn_mask=None):
        output, _ = self.attention(
            x,
            x,
            x,
            attn_mask=attn_mask,
            need_weights=False,
        )
        return output


class ChannelIndependentPatchEmbedding(nn.Module):
    """Patch each channel independently and append one channel-local token."""

    def __init__(self, n_vars, d_model, patch_len, dropout):
        super().__init__()
        self.n_vars = n_vars
        self.patch_len = patch_len
        self.value_embedding = nn.Linear(patch_len, d_model, bias=False)
        self.global_token = nn.Parameter(torch.randn(1, n_vars, 1, d_model))
        self.position_embedding = PositionalEmbedding(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, channel, time]
        batch_size, n_vars, _ = x.shape
        if n_vars != self.n_vars:
            raise ValueError(
                f"Expected {self.n_vars} CI channels, but received {n_vars}."
            )

        x = x.unfold(
            dimension=-1, size=self.patch_len, step=self.patch_len
        )
        patch_num = x.shape[2]
        x = x.reshape(batch_size * n_vars, patch_num, self.patch_len)
        x = self.value_embedding(x) + self.position_embedding(x)
        x = x.reshape(batch_size, n_vars, patch_num, -1)

        global_token = self.global_token.expand(batch_size, -1, -1, -1)
        x = torch.cat([x, global_token], dim=2)
        x = x.reshape(batch_size * n_vars, patch_num + 1, -1)
        return self.dropout(x), n_vars


class CIEncoderLayer(nn.Module):
    """A channel-local Transformer layer with no cross-channel operation."""

    def __init__(
        self,
        self_attention,
        d_model,
        d_ff=None,
        dropout=0.1,
        activation="relu",
    ):
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.self_attention = self_attention
        self.conv1 = nn.Conv1d(d_model, d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(d_ff, d_model, kernel_size=1)
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        self.activation = F.relu if activation == "relu" else F.gelu

    def forward(self, x, attn_mask=None):
        attention_output = self.self_attention(x, attn_mask=attn_mask)
        x = self.norm1(x + self.dropout(attention_output))

        y = self.dropout(
            self.activation(self.conv1(x.transpose(-1, 1)))
        )
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        return self.norm2(x + y)


class CIEncoder(nn.Module):
    def __init__(self, layers, norm_layer=None):
        super().__init__()
        self.layers = nn.ModuleList(layers)
        self.norm = norm_layer

    def forward(self, x, attn_mask=None):
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)

        if self.norm is not None:
            x = self.norm(x)
        return x


class FlattenHead(nn.Module):
    """Map each channel's flattened CI representation to its forecast."""

    def __init__(self, n_features, target_window, dropout=0.0):
        super().__init__()
        self.flatten = nn.Flatten(start_dim=-2)
        self.linear = nn.Linear(n_features, target_window)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        # x: [batch, channel, d_model, patch_num + 1]
        return self.dropout(self.linear(self.flatten(x)))
