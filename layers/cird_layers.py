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


class NeedNet(nn.Module):
    """Build residual-need queries from channel-local CI information."""

    def __init__(
        self,
        seq_len,
        ci_feature_dim,
        pred_len,
        need_dim,
        q_dim,
        num_need_slots,
        need_eps=1e-6,
    ):
        super().__init__()
        self.num_need_slots = num_need_slots
        self.q_dim = q_dim
        self.need_eps = need_eps
        self.x_encoder = nn.Sequential(
            nn.Linear(seq_len, need_dim),
            nn.GELU(),
        )
        self.h_encoder = nn.Sequential(
            nn.Flatten(start_dim=2),
            nn.Linear(ci_feature_dim, need_dim),
            nn.GELU(),
        )
        self.y_encoder = nn.Sequential(
            nn.Linear(pred_len, need_dim),
            nn.GELU(),
        )
        self.fusion = nn.Sequential(
            nn.Linear(3 * need_dim, need_dim),
            nn.GELU(),
            nn.Linear(need_dim, num_need_slots * q_dim),
        )
        self.variance_head = nn.Linear(q_dim, 1)

    def forward(self, x_history, ci_features, y_ci):
        # Every projection operates independently on each channel.
        x_summary = self.x_encoder(x_history)
        h_summary = self.h_encoder(ci_features.detach())
        y_summary = self.y_encoder(
            y_ci.detach().permute(0, 2, 1)
        )

        fused_summary = torch.cat(
            [x_summary, h_summary, y_summary],
            dim=-1,
        )
        q_need = self.fusion(fused_summary).reshape(
            fused_summary.shape[0],
            fused_summary.shape[1],
            self.num_need_slots,
            self.q_dim,
        )
        raw_sigma2 = self.variance_head(q_need).squeeze(-1)
        need_variance = F.softplus(raw_sigma2) + self.need_eps
        return q_need, need_variance


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
