import math

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


class CDSourceEncoder(nn.Module):
    """Encode every raw source channel with CD-owned local parameters."""

    def __init__(
        self,
        seq_len,
        patch_len,
        cd_dim,
        n_heads,
        dropout=0.1,
    ):
        super().__init__()
        if seq_len <= 0:
            raise ValueError("seq_len must be positive for CD source.")
        if patch_len <= 0:
            raise ValueError(
                "patch_len must be positive for CD source."
            )
        if patch_len > seq_len:
            raise ValueError(
                "patch_len must not exceed seq_len for CD source."
            )
        if cd_dim <= 0:
            raise ValueError("cd_dim must be positive for CD source.")
        if n_heads <= 0:
            raise ValueError("n_heads must be positive for CD source.")
        if cd_dim % n_heads != 0:
            raise ValueError(
                "cd_dim must be divisible by n_heads for CD source."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1) for CD source."
            )

        self.seq_len = seq_len
        self.patch_len = patch_len
        self.patch_num = seq_len // patch_len
        self.cd_dim = cd_dim
        self.value_embedding = nn.Linear(
            patch_len,
            cd_dim,
            bias=False,
        )
        self.position_embedding = PositionalEmbedding(cd_dim)
        self.summary_token = nn.Parameter(
            torch.randn(1, 1, cd_dim)
        )
        self.dropout = nn.Dropout(dropout)
        self.encoder = CIEncoder(
            [
                CIEncoderLayer(
                    ChannelIndependentSelfAttention(
                        cd_dim,
                        n_heads,
                        dropout=dropout,
                    ),
                    cd_dim,
                    d_ff=2 * cd_dim,
                    dropout=dropout,
                    activation="gelu",
                )
            ],
            norm_layer=nn.LayerNorm(cd_dim),
        )

    def forward(self, x_history):
        # x_history: [batch, channel, time]
        if x_history.ndim != 3:
            raise ValueError(
                "x_history must have shape [B, C, seq_len]."
            )
        batch_size, n_vars, seq_len = x_history.shape
        if seq_len != self.seq_len:
            raise ValueError(
                f"Expected CD source seq_len {self.seq_len}, "
                f"but received {seq_len}."
            )

        patches = x_history.unfold(
            dimension=-1,
            size=self.patch_len,
            step=self.patch_len,
        )
        patches = patches.reshape(
            batch_size * n_vars,
            self.patch_num,
            self.patch_len,
        )
        tokens = self.value_embedding(patches)
        tokens = tokens + self.position_embedding(tokens)

        summary_token = self.summary_token.expand(
            batch_size * n_vars,
            -1,
            -1,
        )
        tokens = torch.cat([tokens, summary_token], dim=1)
        tokens = self.encoder(self.dropout(tokens))
        source_features = tokens[:, -1, :]
        return source_features.reshape(
            batch_size,
            n_vars,
            self.cd_dim,
        )


class LightContext(nn.Module):
    """Compress source anchors into a small set of context factors."""

    def __init__(
        self,
        cd_dim,
        context_rank,
        dropout=0.1,
    ):
        super().__init__()
        if cd_dim <= 0:
            raise ValueError(
                "cd_dim must be positive for LightContext."
            )
        if context_rank <= 0:
            raise ValueError(
                "context_rank must be positive for LightContext."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1) for LightContext."
            )

        self.cd_dim = cd_dim
        self.context_rank = context_rank
        self.context_queries = nn.Parameter(
            torch.empty(context_rank, cd_dim)
        )
        nn.init.normal_(
            self.context_queries,
            std=cd_dim ** -0.5,
        )
        self.input_norm = nn.LayerNorm(cd_dim)
        self.key_projection = nn.Linear(
            cd_dim,
            cd_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            cd_dim,
            cd_dim,
            bias=False,
        )
        self.output_norm = nn.LayerNorm(cd_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, source_features):
        # source_features: [batch, channel, cd_dim]
        if source_features.ndim != 3:
            raise ValueError(
                "source_features must have shape [B, C, cd_dim]."
            )
        if source_features.shape[-1] != self.cd_dim:
            raise ValueError(
                f"Expected source cd_dim {self.cd_dim}, but received "
                f"{source_features.shape[-1]}."
            )

        normalized_source = self.input_norm(source_features)
        context_key = self.key_projection(normalized_source)
        context_value = self.value_projection(normalized_source)
        context_score = torch.einsum(
            "rd,bcd->brc",
            self.context_queries,
            context_key,
        ) / math.sqrt(self.cd_dim)
        context_attention = torch.softmax(
            context_score,
            dim=-1,
        )
        context_tokens = torch.einsum(
            "brc,bcd->brd",
            context_attention,
            context_value,
        )
        context_tokens = self.output_norm(
            self.dropout(context_tokens)
        )
        return context_tokens, context_attention


class SourceContextRead(nn.Module):
    """Read shared context while keeping each source anchor dominant."""

    def __init__(self, cd_dim, dropout=0.1):
        super().__init__()
        if cd_dim <= 0:
            raise ValueError(
                "cd_dim must be positive for source context read."
            )
        if not 0.0 <= dropout < 1.0:
            raise ValueError(
                "dropout must be in [0, 1) for source context read."
            )

        self.cd_dim = cd_dim
        self.source_norm = nn.LayerNorm(cd_dim)
        self.query_projection = nn.Linear(
            cd_dim,
            cd_dim,
            bias=False,
        )
        self.key_projection = nn.Linear(
            cd_dim,
            cd_dim,
            bias=False,
        )
        self.value_projection = nn.Linear(
            cd_dim,
            cd_dim,
            bias=False,
        )
        self.context_gate = nn.Linear(cd_dim, cd_dim)
        self.source_delta = nn.Sequential(
            nn.Linear(cd_dim, cd_dim),
            nn.GELU(),
            nn.Linear(cd_dim, cd_dim),
        )
        self.dropout = nn.Dropout(dropout)
        self.eta = nn.Parameter(torch.tensor(0.1))

    def forward(self, source_features, context_tokens):
        if source_features.ndim != 3:
            raise ValueError(
                "source_features must have shape [B, C, cd_dim]."
            )
        if context_tokens.ndim != 3:
            raise ValueError(
                "context_tokens must have shape [B, R, cd_dim]."
            )
        if (
            source_features.shape[0] != context_tokens.shape[0]
            or source_features.shape[-1] != self.cd_dim
            or context_tokens.shape[-1] != self.cd_dim
        ):
            raise ValueError(
                "Source and context shapes are incompatible."
            )

        read_query = self.query_projection(
            self.source_norm(source_features)
        )
        read_key = self.key_projection(context_tokens)
        read_value = self.value_projection(context_tokens)
        read_score = torch.einsum(
            "bcd,brd->bcr",
            read_query,
            read_key,
        ) / math.sqrt(self.cd_dim)
        read_attention = torch.softmax(read_score, dim=-1)
        source_context = torch.einsum(
            "bcr,brd->bcd",
            read_attention,
            read_value,
        )

        context_gate = torch.sigmoid(
            self.context_gate(source_context)
        )
        source_delta = self.source_delta(source_features)
        contextual_source = source_features + (
            self.eta
            * context_gate
            * self.dropout(source_delta)
        )
        return contextual_source, read_attention


class ResidualCDBranch(nn.Module):
    """Produce a cross-channel residual without reading CI features."""

    def __init__(
        self,
        seq_len,
        patch_len,
        pred_len,
        num_need_slots,
        q_dim,
        cd_dim,
        n_heads,
        context_rank,
        dropout=0.1,
        attention_temperature=1.0,
        null_supply=True,
        value_adapter_rank=16,
        cd_init_scale=0.1,
        need_eps=1e-6,
    ):
        super().__init__()
        if pred_len <= 0:
            raise ValueError(
                "pred_len must be positive for Residual-CD."
            )
        if num_need_slots <= 0:
            raise ValueError(
                "num_need_slots must be positive for Residual-CD."
            )
        if pred_len % num_need_slots != 0:
            raise ValueError(
                "pred_len must be divisible by num_need_slots "
                "for Residual-CD."
            )
        if q_dim <= 0:
            raise ValueError(
                "q_dim must be positive for Residual-CD."
            )
        if cd_dim <= 0:
            raise ValueError(
                "cd_dim must be positive for Residual-CD."
            )
        if n_heads <= 0:
            raise ValueError(
                "n_heads must be positive for Residual-CD."
            )
        if cd_dim % n_heads != 0:
            raise ValueError(
                "cd_dim must be divisible by n_heads for "
                "Residual-CD."
            )
        if context_rank <= 0:
            raise ValueError(
                "context_rank must be positive for Residual-CD."
            )
        if attention_temperature <= 0:
            raise ValueError(
                "attention_temperature must be positive for "
                "Residual-CD."
            )
        if value_adapter_rank <= 0:
            raise ValueError(
                "value_adapter_rank must be positive for "
                "Residual-CD."
            )
        if cd_init_scale < 0:
            raise ValueError(
                "cd_init_scale must be non-negative for Residual-CD."
            )
        if need_eps < 0:
            raise ValueError(
                "need_eps must be non-negative for Residual-CD."
            )
        if null_supply not in {False, True, 0, 1}:
            raise ValueError(
                "null_supply must be boolean for Residual-CD."
            )

        self.pred_len = pred_len
        self.num_need_slots = num_need_slots
        self.q_dim = q_dim
        self.cd_dim = cd_dim
        self.attention_temperature = attention_temperature
        self.null_supply = bool(null_supply)
        self.cd_init_scale = cd_init_scale
        self.need_eps = need_eps
        self.block_len = pred_len // num_need_slots

        self.source_encoder = CDSourceEncoder(
            seq_len=seq_len,
            patch_len=patch_len,
            cd_dim=cd_dim,
            n_heads=n_heads,
            dropout=dropout,
        )
        self.light_context = LightContext(
            cd_dim=cd_dim,
            context_rank=context_rank,
            dropout=dropout,
        )
        self.source_read = SourceContextRead(
            cd_dim=cd_dim,
            dropout=dropout,
        )
        self.q_match = nn.Linear(q_dim, q_dim, bias=False)
        self.supply_key = nn.Linear(cd_dim, q_dim, bias=False)

        if self.null_supply:
            self.null_keys = nn.Parameter(
                torch.empty(num_need_slots, q_dim)
            )
            nn.init.normal_(
                self.null_keys,
                std=q_dim ** -0.5,
            )
        else:
            self.register_parameter("null_keys", None)

        self.value_base = nn.Linear(
            cd_dim,
            cd_dim,
            bias=False,
        )
        self.value_adapter_down = nn.ModuleList(
            [
                nn.Linear(
                    cd_dim,
                    value_adapter_rank,
                    bias=False,
                )
                for _ in range(num_need_slots)
            ]
        )
        self.value_adapter_up = nn.ModuleList(
            [
                nn.Linear(
                    value_adapter_rank,
                    cd_dim,
                    bias=False,
                )
                for _ in range(num_need_slots)
            ]
        )
        for adapter_up in self.value_adapter_up:
            nn.init.normal_(adapter_up.weight, std=1e-3)

        self.gate_scale = nn.Parameter(
            torch.ones(num_need_slots)
        )
        self.gate_bias = nn.Parameter(
            torch.zeros(num_need_slots)
        )
        self.decoders = nn.ModuleList(
            [
                nn.Linear(
                    cd_dim,
                    self.block_len,
                    bias=False,
                )
                for _ in range(num_need_slots)
            ]
        )

    def _validate_inputs(
        self,
        x_history,
        q_need,
        need_variance,
    ):
        if x_history.ndim != 3:
            raise ValueError(
                "x_history must have shape [B, C, seq_len]."
            )
        if q_need.ndim != 4:
            raise ValueError(
                "q_need must have shape [B, C, M, Dq]."
            )
        if need_variance.ndim != 3:
            raise ValueError(
                "need_variance must have shape [B, C, M]."
            )

        batch_size, n_vars, _ = x_history.shape
        expected_q_shape = (
            batch_size,
            n_vars,
            self.num_need_slots,
            self.q_dim,
        )
        if tuple(q_need.shape) != expected_q_shape:
            raise ValueError(
                f"Expected q_need shape {expected_q_shape}, "
                f"but received {tuple(q_need.shape)}."
            )

        expected_variance_shape = (
            batch_size,
            n_vars,
            self.num_need_slots,
        )
        if tuple(need_variance.shape) != expected_variance_shape:
            raise ValueError(
                "Expected need_variance shape "
                f"{expected_variance_shape}, but received "
                f"{tuple(need_variance.shape)}."
            )
        if n_vars == 1 and not self.null_supply:
            raise ValueError(
                "Residual-CD without null supply requires at least "
                "two channels."
            )

    def _build_slot_values(self, contextual_source):
        value_base = self.value_base(contextual_source)
        slot_values = []
        for adapter_down, adapter_up in zip(
            self.value_adapter_down,
            self.value_adapter_up,
        ):
            adapted_value = value_base + adapter_up(
                adapter_down(value_base)
            )
            slot_values.append(
                F.normalize(adapted_value, dim=-1)
            )
        return torch.stack(slot_values, dim=2)

    def forward(self, x_history, q_need, need_variance):
        self._validate_inputs(
            x_history,
            q_need,
            need_variance,
        )
        source_features = self.source_encoder(x_history)
        context_tokens, context_attention = self.light_context(
            source_features
        )
        contextual_source, read_attention = self.source_read(
            source_features,
            context_tokens,
        )

        q_match = F.normalize(
            self.q_match(q_need),
            dim=-1,
        )
        k_supply = F.normalize(
            self.supply_key(contextual_source),
            dim=-1,
        )
        score = torch.einsum(
            "bimd,bjd->bimj",
            q_match,
            k_supply,
        ) / self.attention_temperature

        n_vars = x_history.shape[1]
        self_mask = torch.eye(
            n_vars,
            device=score.device,
            dtype=torch.bool,
        ).view(1, n_vars, 1, n_vars)
        score = score.masked_fill(self_mask, float("-inf"))

        if self.null_supply:
            normalized_null_keys = F.normalize(
                self.null_keys,
                dim=-1,
            )
            null_score = torch.einsum(
                "bimd,md->bim",
                q_match,
                normalized_null_keys,
            ) / self.attention_temperature
            attention_with_null = torch.softmax(
                torch.cat(
                    [score, null_score.unsqueeze(-1)],
                    dim=-1,
                ),
                dim=-1,
            )
            cd_attention = attention_with_null[..., :n_vars]
            null_attention = attention_with_null[..., n_vars]
        else:
            cd_attention = torch.softmax(score, dim=-1)
            null_attention = None

        slot_values = self._build_slot_values(
            contextual_source
        )
        r_cd = torch.einsum(
            "bimj,bjmd->bimd",
            cd_attention,
            slot_values,
        )

        variance_floor = max(
            self.need_eps,
            torch.finfo(need_variance.dtype).tiny,
        )
        log_need = torch.log(
            need_variance.detach().clamp_min(variance_floor)
        )
        gate = torch.sigmoid(
            log_need
            * self.gate_scale.view(1, 1, -1)
            + self.gate_bias.view(1, 1, -1)
        ).unsqueeze(-1)

        delta_blocks = [
            gate[:, :, slot, :]
            * decoder(r_cd[:, :, slot, :])
            for slot, decoder in enumerate(self.decoders)
        ]
        delta_cd = torch.cat(delta_blocks, dim=-1)
        delta_cd = self.cd_init_scale * delta_cd
        delta_cd = delta_cd.permute(0, 2, 1).contiguous()

        return delta_cd, {
            "source_features": source_features,
            "context_tokens": context_tokens,
            "context_attention": context_attention,
            "read_attention": read_attention,
            "k_supply": k_supply,
            "cd_attention": cd_attention,
            "null_attention": null_attention,
            "r_cd": r_cd,
        }


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
