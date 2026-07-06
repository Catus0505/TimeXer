# Residual-CD Stage 2 Integration Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Integrate the verified standalone Residual-CD branch into CIRD with strictly separated CI, CD, and need-loss gradients.

**Architecture:** CIRD computes the pure CI path first, builds NeedNet queries from detached CI state, and gives raw channel history to the independent CD source branch. Training combines a standalone CI MSE, final residual forecast MSE with detached `y_ci`, and the existing need loss.

**Tech Stack:** Python, PyTorch, argparse, standard-library `unittest`

## Global Constraints

- `prediction` is numerically `y_ci + delta_cd`.
- Final forecast MSE and need loss must not update CI parameters.
- CI parameters are updated only by standalone `MSE(y_ci, batch_y)`.
- The CD source branch reads raw channel history and never CI features.
- `aux["ci_prediction"]` remains the pure `y_ci`.
- Do not modify `models/TimeXer.py` or `layers/SelfAttention_Family.py`.
- Do not run local experiments; all Python execution occurs on the server.

---

### Task 1: Define integration and gradient tests

**Files:**
- Create: `tests/test_cird_model_integration.py`

**Interfaces:**
- Consumes: `models.cird.Model`, `compute_need_loss`
- Produces: shape, residual identity, gradient ownership, and combined-loss tests

- [ ] **Step 1: Specify complete auxiliary outputs**

For `B=2`, `C=3`, `seq_len=24`, `pred_len=12`, and `M=3`, assert final,
CI, need, residual, real attention, and null attention shapes. Assert:

```python
prediction == aux["ci_prediction"] + aux["cd_residual"]
```

- [ ] **Step 2: Specify final forecast gradient isolation**

Backpropagate only `MSE(prediction, target)`. Assert every CI embedding,
encoder, and head parameter has no gradient. Assert NeedNet fusion and CD
source/decoder parameters have finite nonzero gradients.

- [ ] **Step 3: Specify standalone CI gradient ownership**

Backpropagate only `MSE(aux["ci_prediction"], target)`. Assert CI
parameters receive finite nonzero gradients while all CD parameters have
no gradient.

- [ ] **Step 4: Specify combined training loss**

Calculate:

```python
loss_ci = mse(aux["ci_prediction"], target)
loss_cd = mse(prediction, target)
loss_need = compute_need_loss(
    target,
    aux["ci_prediction"],
    aux["need_variance"],
    num_need_slots=3,
    eps=1e-6,
)
loss = loss_ci + loss_cd + 0.1 * loss_need
```

Assert the total is finite and backward gives gradients to CI, NeedNet,
and CD parameters.

### Task 2: Integrate the CD branch into CIRD

**Files:**
- Modify: `models/cird.py`
- Test: `tests/test_cird_model_integration.py`

**Interfaces:**
- Produces: CIRD final prediction and documented auxiliary dictionary

- [ ] **Step 1: Construct `ResidualCDBranch`**

Resolve CD config defaults from `d_model` and initialize the branch with
raw-history length, patch length, horizon, need dimensions, model heads,
context rank, dropout, temperature, null flag, adapter rank, initial
scale, and need epsilon.

- [ ] **Step 2: Execute NeedNet and CD unconditionally**

After `forecast_ci`, calculate NeedNet from raw selected history and
detached CI state, then pass the same raw history, `q_need`, and
`need_variance` to the CD branch.

- [ ] **Step 3: Block final forecast gradients from CI**

Return:

```python
prediction = y_ci.detach() + delta_cd
```

This remains numerically equal to `y_ci + delta_cd`.

- [ ] **Step 4: Preserve pure CI auxiliary output**

When requested, return exactly the documented CI, need, residual, real
attention, and null attention tensors. Otherwise return only the final
prediction.

### Task 3: Separate trainer losses

**Files:**
- Modify: `exp/exp_long_term_forecasting.py`
- Test: `tests/test_cird_model_integration.py`

**Interfaces:**
- Consumes: final prediction and CIRD auxiliary output
- Produces: `loss_ci + loss_cd + need_loss_weight * loss_need`

- [ ] **Step 1: Always request CIRD auxiliary output during training**

Because CIRD is the only current model exposing `return_aux`, set
training `request_aux` from `self.model_supports_aux`, independently of
need-loss weight.

- [ ] **Step 2: Add standalone CI MSE**

Slice `aux["ci_prediction"]` with the same horizon/feature rules as the
final output and add `MSE(ci_prediction, batch_y)` to final forecast
loss. Do not detach this CI prediction.

- [ ] **Step 3: Preserve need loss semantics**

Keep `compute_need_loss` unchanged so its residual remains:

```python
batch_y - ci_prediction.detach()
```

Then add the weighted need term after the standalone CI loss.

### Task 4: Add CLI configuration

**Files:**
- Modify: `run.py`

**Interfaces:**
- Produces: validated Residual-CD attributes on parsed args

- [ ] **Step 1: Add requested options**

Add `context_rank`, `cd_dim`, `cd_dropout`,
`attention_temperature`, `null_supply`, `value_adapter_rank`, and
`cd_init_scale` with the documented defaults.

- [ ] **Step 2: Resolve and validate**

Default `cd_dim` to `d_model`. Reject nonpositive dimensions/ranks or
temperature, negative scale, invalid dropout/null flag, and
`cd_dim % n_heads != 0` through `parser.error`.

### Task 5: Review and server handoff

**Files:**
- Review: `models/cird.py`
- Review: `exp/exp_long_term_forecasting.py`
- Review: `run.py`
- Review: `tests/test_cird_model_integration.py`

**Interfaces:**
- Produces: one pushed stage-2 checkpoint and server report

- [ ] **Step 1: Perform non-executing scope review**

Check detach placement, aux identity, feature slicing, AMP/non-AMP loss
parity, and absence of edits to protected files.

- [ ] **Step 2: Commit and push**

```bash
git add models/cird.py exp/exp_long_term_forecasting.py run.py \
  tests/test_cird_model_integration.py \
  docs/superpowers/plans/2026-07-06-residual-cd-stage2.md
git commit -m "feat: integrate residual cd forecasting"
git push origin cird-dev
```

- [ ] **Step 3: Stop for server verification**

The server runs compilation, all CIRD unit tests, exact gradient-isolation
checks, a small optimizer step, and a full-model Traffic memory test. No
real-dataset epoch begins until this report is reviewed.
