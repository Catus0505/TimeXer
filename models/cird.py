import torch
import torch.nn as nn

from layers.cird_layers import (
    ChannelIndependentPatchEmbedding,
    ChannelIndependentSelfAttention,
    CIEncoder,
    CIEncoderLayer,
    FlattenHead,
    NeedNet,
)


class Model(nn.Module):
    """Channel-independent forecasting baseline for CIRD."""

    def __init__(self, configs):
        super().__init__()
        self.task_name = configs.task_name
        self.features = configs.features
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.use_norm = configs.use_norm
        self.patch_len = configs.patch_len
        self.num_need_slots = getattr(configs, "num_need_slots", 4)
        self.need_eps = getattr(configs, "need_eps", 1e-6)
        self.need_dim = (
            getattr(configs, "need_dim", None) or configs.d_model
        )
        self.q_dim = (
            getattr(configs, "q_dim", None) or configs.d_model
        )

        if self.patch_len > self.seq_len:
            raise ValueError(
                "patch_len must not be greater than seq_len for cird."
            )
        if self.num_need_slots <= 0:
            raise ValueError("num_need_slots must be positive for cird.")
        if self.need_eps < 0:
            raise ValueError("need_eps must be non-negative for cird.")
        if self.need_dim <= 0:
            raise ValueError("need_dim must be positive for cird.")
        if self.q_dim <= 0:
            raise ValueError("q_dim must be positive for cird.")
        if self.pred_len % self.num_need_slots != 0:
            raise ValueError(
                "pred_len must be divisible by num_need_slots for cird."
            )

        self.patch_num = self.seq_len // self.patch_len
        self.n_vars = (
            1 if self.features in {"S", "MS"} else configs.enc_in
        )

        self.ci_embedding = ChannelIndependentPatchEmbedding(
            self.n_vars,
            configs.d_model,
            self.patch_len,
            configs.dropout,
        )
        self.ci_encoder = CIEncoder(
            [
                CIEncoderLayer(
                    ChannelIndependentSelfAttention(
                        configs.d_model,
                        configs.n_heads,
                        dropout=configs.dropout,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for _ in range(configs.e_layers)
            ],
            norm_layer=nn.LayerNorm(configs.d_model),
        )
        self.ci_head = FlattenHead(
            configs.d_model * (self.patch_num + 1),
            self.pred_len,
            dropout=configs.dropout,
        )
        self.need_net = NeedNet(
            self.seq_len,
            configs.d_model * (self.patch_num + 1),
            self.pred_len,
            self.need_dim,
            self.q_dim,
            self.num_need_slots,
            need_eps=self.need_eps,
        )

    def _select_ci_input(self, x_enc):
        if self.features == "MS":
            return x_enc[:, :, -1:]
        return x_enc

    def forecast_ci(self, x_enc):
        if self.use_norm:
            means = x_enc.mean(dim=1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(
                    x_enc,
                    dim=1,
                    keepdim=True,
                    unbiased=False,
                )
                + 1e-5
            )
            x_enc = x_enc / stdev

        ci_input = self._select_ci_input(x_enc)
        ci_tokens, n_vars = self.ci_embedding(
            ci_input.permute(0, 2, 1)
        )
        ci_tokens = self.ci_encoder(ci_tokens)
        ci_tokens = ci_tokens.reshape(
            -1,
            n_vars,
            ci_tokens.shape[-2],
            ci_tokens.shape[-1],
        )
        ci_features = ci_tokens.permute(0, 1, 3, 2)

        y_ci = self.ci_head(ci_features).permute(0, 2, 1)

        if self.use_norm:
            if self.features == "MS":
                means = means[:, :, -1:]
                stdev = stdev[:, :, -1:]
            y_ci = y_ci * stdev + means

        return y_ci, ci_features

    def forward(
        self,
        x_enc,
        x_mark_enc,
        x_dec,
        x_mark_dec,
        mask=None,
        return_aux=False,
    ):
        if self.task_name not in {
            "long_term_forecast",
            "short_term_forecast",
        }:
            return None

        y_ci, ci_features = self.forecast_ci(x_enc)
        prediction = y_ci

        if return_aux:
            x_history = self._select_ci_input(x_enc).permute(
                0, 2, 1
            )
            q_need, need_variance = self.need_net(
                x_history,
                ci_features,
                y_ci,
            )
            return prediction, {
                "ci_prediction": y_ci,
                "q_need": q_need,
                "need_variance": need_variance,
            }
        return prediction
