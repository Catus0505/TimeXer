# Residual-CD Core-v1 Design

## Status

This document records the design checkpoint reached before
implementation. It replaces the earlier draft that reused CI features as
the CD source representation.

The normative Q/K/V semantics remain those in
`docs/注意力机制设计.md`.

## Invariants

The model has two forecasting paths with an explicit gradient boundary:

```python
y_ci, h_ci = forecast_ci(x)
q_need, need_variance = need_net(
    x,
    h_ci.detach(),
    y_ci.detach(),
)
delta_cd, cd_aux = cd_branch(
    x,
    q_need,
    need_variance,
)
prediction = y_ci.detach() + delta_cd
```

Numerically:

```python
prediction == y_ci + delta_cd
```

The detach only controls training gradients.

- CI parameters receive gradients only from the standalone CI forecast
  MSE.
- NeedNet and Residual-CD cannot update the CI encoder or CI head.
- `aux["ci_prediction"]` is the pure `y_ci`.
- `delta_cd` is an additive residual and never replaces `y_ci`.
- NeedNet retains the input semantics
  `NeedNet(X_i, sg(h_i_CI), sg(y_i_CI))`.
- Core-v1 introduces no CD-specific auxiliary loss. CD is trained by
  final forecast MSE.
- `need_variance` is detached before it gates the CD decoder.

## Losses and Gradient Ownership

Training uses three terms:

```python
loss_ci = mse(y_ci, batch_y)
loss_cd = mse(y_ci.detach() + delta_cd, batch_y)
loss_need = compute_need_loss(
    batch_y,
    y_ci.detach(),
    need_variance,
)
loss = loss_ci + loss_cd + need_loss_weight * loss_need
```

Gradient ownership is:

| Loss | CI | NeedNet | CD source/K/V/decoder |
| --- | --- | --- | --- |
| `loss_ci` | yes | no | no |
| `loss_cd` | no | through `q_need` | yes |
| `loss_need` | no | yes | no |

Validation and testing evaluate the final numerical prediction
`y_ci + delta_cd`.

## CD Source Encoder

The supply branch does not reuse `h_j_CI`. It has an independent,
channel-local encoder:

```text
X_j
  -> non-overlapping patching
  -> shared CD patch embedding
  -> append one shared summary token
  -> one channel-local Transformer encoder layer
  -> z_j
```

The encoder has independent parameters from CI and is trained only
through `loss_cd`. The same encoder weights are applied to every channel;
there is no cross-channel operation in this stage.

Core-v1 fixes the source encoder structure rather than selecting a
different encoder per dataset:

- use the same `patch_len` as the CI path;
- project each patch from `patch_len` to `cd_dim`;
- use one summary token shared by every channel, rather than one
  channel-specific token per variable;
- add positional encoding and `cd_dropout`;
- use exactly one Transformer encoder layer with model `n_heads`,
  feed-forward width `2 * cd_dim`, GELU, residual connections, and layer
  normalization;
- require `cd_dim` to be divisible by `n_heads`;
- take the final normalized summary token as `z_j`.

This encoder is applied by folding channels into the batch dimension.
Its local-attention cost therefore grows linearly with the number of
channels.

For all channels:

```python
Z = CDSourceEncoder(x_history)  # [B, C, cd_dim]
```

For target channel `i`, `z_j` is an auxiliary source anchor when
`j != i`. Channel `j` can simultaneously be a prediction target in its
own attention row. Self-supply is always masked.

## LightContext

LightContext uses learnable-query attention pooling, not mean/max
pooling.

```python
context_query = learned_parameter  # [R, cd_dim]
context_key = W_k_ctx(Z)           # [B, C, cd_dim]
context_value = W_v_ctx(Z)         # [B, C, cd_dim]

context_score = torch.einsum(
    "rd,bcd->brc",
    context_query,
    context_key,
) / sqrt(cd_dim)
context_weight = softmax(context_score, dim=-1)
H = torch.einsum(
    "brc,bcd->brd",
    context_weight,
    context_value,
)                                      # [B, R, cd_dim]
```

`R = context_rank` is small and shared across all datasets. This stage
costs `O(B * R * C * cd_dim)` and does not create a `C x C` context
matrix.

`Z` is layer-normalized before pooling, and the pooled `H` is
layer-normalized before Read. Attention scaling and dropout follow the
same configuration on every dataset.

## Source-conditioned Read

Read is attention over the `R` context tokens, not a single linear
layer:

```python
read_query = W_q_read(Z)  # [B, C, cd_dim]
read_key = W_k_read(H)    # [B, R, cd_dim]
read_value = W_v_read(H)  # [B, R, cd_dim]

read_score = torch.einsum(
    "bcd,brd->bcr",
    read_query,
    read_key,
) / sqrt(cd_dim)
read_weight = softmax(read_score, dim=-1)
c = torch.einsum(
    "bcr,brd->bcd",
    read_weight,
    read_value,
)                          # [B, C, cd_dim]
```

Each source `z_j` therefore selects its own mixture of the shared
context factors.

The source remains anchored during modulation:

```python
context_gate = sigmoid(W_gamma(c))
source_delta = W_phi_2(GELU(W_phi_1(Z)))
u_ctx = Z + eta * context_gate * source_delta
```

`eta` is learnable and initialized to `0.1`.

## K and Demand-Supply Score

```python
q_match = normalize(W_q_match(q_need))  # [B, C, M, Dq]
k_supply = normalize(W_k(u_ctx))        # [B, C, Dq]

score = torch.einsum(
    "bimd,bjd->bimj",
    q_match,
    k_supply,
) / attention_temperature              # [B, C, M, C]
```

`score[b, i, m, j]` measures whether auxiliary channel `j` supplies the
correction type requested by target channel `i` in residual slot `m`.
Every diagonal element `j == i` is masked before softmax.

## Null Supply

When enabled, each slot has one learned null key and an exactly zero
null value:

```python
null_score = dot(q_match, normalized_null_key[m])
attention_with_null = softmax(
    concat(masked_score, null_score),
    dim=-1,
)
cd_attention = attention_with_null[..., :C]
null_attention = attention_with_null[..., C]
```

Null and real sources participate in the same normalization. Therefore
real-source attention can sum to less than one when no source matches the
need.

## V and Slot Adapters

```python
v_base = W_v0(u_ctx)  # [B, C, cd_dim]
v_jm = normalize(
    v_base_j + A_m(B_m(v_base_j))
)                      # [B, C, M, cd_dim]
```

`B_m` and `A_m` are slot-specific low-rank linear adapters. Their output
projection is initialized at a small scale.

Real source values are aggregated as:

```python
r_cd = torch.einsum(
    "bimj,bjmd->bimd",
    cd_attention,
    v_slot,
)  # [B, C, M, cd_dim]
```

The null value is zero and does not enter this sum.

## Decoder

Each slot decodes exactly one horizon block:

```python
gate = sigmoid(
    gate_scale[m]
    * log(need_variance[:, :, m].detach() + need_eps)
    + gate_bias[m]
)
delta_block_m = gate * decoder_m(r_cd[:, :, m])
```

All blocks are concatenated in slot order:

```python
delta_cd = cat(delta_blocks, dim=-1)
delta_cd = cd_init_scale * delta_cd
delta_cd = delta_cd.permute(0, 2, 1)  # [B, pred_len, C]
```

## Unified Dense Attention

Core-v1 uses the same exact dense demand-supply attention for every
dataset, including Traffic:

```python
cd_attention.shape == [B, C, M, C]
```

There is no dataset-specific top-k, grouping, approximate attention, or
chunking in core-v1.

For `B=8`, `C=864`, and `M=4`, one FP32 attention tensor is about
91 MiB. The first Traffic server run must measure full model peak memory
and step time. If the agreed 6--12 GiB budget is exceeded, a later
checkpointed blockwise implementation may change the computation
strategy while preserving the exact same mathematical attention.

## Interface

`ResidualCDBranch` consumes:

- `x_history`: `[B, C, seq_len]`
- `q_need`: `[B, C, M, Dq]`
- `need_variance`: `[B, C, M]`

It returns:

- `delta_cd`: `[B, pred_len, C]`
- `cd_attention`: `[B, C, M, C]`
- `null_attention`: `[B, C, M]` when null supply is enabled

With auxiliary output enabled, CIRD returns:

```python
{
    "ci_prediction": y_ci,
    "q_need": q_need,
    "need_variance": need_variance,
    "cd_residual": delta_cd,
    "cd_attention": cd_attention,
    "null_attention": null_attention,
}
```

## Validation and Edge Cases

- `pred_len` must be divisible by `num_need_slots`.
- `context_rank`, `cd_dim`, and adapter rank must be positive.
- Attention temperature must be positive.
- Initial CD scale must be nonnegative.
- With null supply enabled, a single-channel input routes all attention
  to null and produces zero CD residual.
- Without null supply, at least two channels are required.
- Server tests must verify that CI parameters receive gradients from
  `loss_ci` but receive no gradients from isolated `loss_cd` or
  `loss_need` backward passes.
- Server tests must verify
  `prediction == aux["ci_prediction"] + aux["cd_residual"]`.
