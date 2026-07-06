# Residual-CD Stage 1 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Implement a standalone, server-testable Residual-CD branch without changing CIRD model or trainer behavior.

**Architecture:** Encode every raw source channel with an independent-from-CI but channel-shared local patch encoder. Compress source anchors with learned-query LightContext, let each source read those context tokens, then perform exact dense need-supply matching, null-aware aggregation, slot-adapted values, and gated horizon decoding.

**Tech Stack:** Python, PyTorch, standard-library `unittest`

## Global Constraints

- Do not modify `models/cird.py`, `run.py`, or any trainer in stage 1.
- Do not modify `models/TimeXer.py` or `layers/SelfAttention_Family.py`.
- The CD source encoder consumes raw `x_history`, never CI features.
- Core-v1 uses exact dense attention for all datasets and no chunking.
- The decoder must detach `need_variance`.
- No local experiments are run; the user runs all verification on the server.

---

### Task 1: Define the standalone branch contract

**Files:**
- Create: `tests/test_cird_residual_cd_branch.py`
- Modify: `layers/cird_layers.py`

**Interfaces:**
- Consumes: `x_history [B,C,L]`, `q_need [B,C,M,Dq]`, and `need_variance [B,C,M]`
- Produces: `delta_cd [B,pred_len,C]` and a diagnostic dictionary containing source/context/read/K/attention/aggregation tensors

- [ ] **Step 1: Write source-encoder isolation tests**

Instantiate `CDSourceEncoder(seq_len=24, patch_len=6, cd_dim=8,
n_heads=2, dropout=0.0)`. Assert output shape `[2,3,8]`. In evaluation
mode, perturb only input channel zero and assert source outputs for
channels one and two remain exactly unchanged.

- [ ] **Step 2: Write branch shape and attention tests**

Instantiate a three-channel branch with three horizon slots. Assert:

```python
delta_cd.shape == (2, 12, 3)
aux["source_features"].shape == (2, 3, 8)
aux["context_tokens"].shape == (2, 2, 8)
aux["context_attention"].shape == (2, 2, 3)
aux["read_attention"].shape == (2, 3, 2)
aux["k_supply"].shape == (2, 3, 5)
aux["cd_attention"].shape == (2, 3, 3, 3)
aux["null_attention"].shape == (2, 3, 3)
aux["r_cd"].shape == (2, 3, 3, 8)
```

Assert source self-attention is exactly zero and real plus null weights
sum to one.

- [ ] **Step 3: Write null and gradient-boundary tests**

Assert one-channel inputs route all mass to null and produce zero
residual. Assert a no-null one-channel branch raises a clear error.
Backpropagate through `delta_cd` and assert CD parameters and `q_need`
receive gradients while the leaf `need_variance` receives no gradient.

- [ ] **Step 4: Defer test execution to the server**

Server command:

```bash
python -m unittest discover -s tests -p 'test_cird_residual_cd_branch.py' -v
```

Expected result after implementation: all tests pass.

### Task 2: Implement raw-channel supply encoding and context

**Files:**
- Modify: `layers/cird_layers.py`
- Test: `tests/test_cird_residual_cd_branch.py`

**Interfaces:**
- Produces: `CDSourceEncoder`, `LightContext`, and `SourceContextRead`

- [ ] **Step 1: Implement `CDSourceEncoder`**

Use non-overlapping patching, a shared linear patch projection,
positional encoding, one shared summary token, and one independent
channel-local `CIEncoderLayer`. Fold channels into the batch dimension,
then return the final normalized summary token as `[B,C,cd_dim]`.

- [ ] **Step 2: Implement learned-query `LightContext`**

Normalize source anchors, project context keys/values, calculate
`[B,R,C]` attention from learned `[R,D]` queries, and return normalized
`H [B,R,D]` plus the context attention.

- [ ] **Step 3: Implement source-conditioned Read and modulation**

Project Z queries and H keys/values, softmax over R, and produce
`c [B,C,D]`. Return:

```python
u_ctx = Z + eta * sigmoid(gamma(c)) * phi(Z)
```

with learnable scalar `eta` initialized to `0.1`.

### Task 3: Implement exact Residual-CD matching and decoding

**Files:**
- Modify: `layers/cird_layers.py`
- Test: `tests/test_cird_residual_cd_branch.py`

**Interfaces:**
- Produces: `ResidualCDBranch.forward(...) -> (delta_cd, cd_aux)`

- [ ] **Step 1: Implement exact need-supply matching**

Normalize projected queries and keys, calculate
`einsum("bimd,bjd->bimj") / temperature`, and mask every `j == i`
entry.

- [ ] **Step 2: Add slot null supply**

Normalize one learned null key per slot, concatenate its score before
the only source softmax, and split real and null attention afterward.
Keep the null value exactly zero.

- [ ] **Step 3: Add shared values and slot adapters**

Build one shared value base and one rank-reducing/rank-expanding linear
adapter per slot. Initialize expanding weights with standard deviation
`1e-3`, normalize values, and aggregate with:

```python
r_cd = torch.einsum("bimj,bjmd->bimd", cd_attention, v_slot)
```

- [ ] **Step 4: Add detached need gates and slot decoders**

Use per-slot learned gate scale/bias over detached log need variance.
Decode each slot to `pred_len // M`, concatenate blocks, multiply by
`cd_init_scale`, and permute to `[B,pred_len,C]`. Decoder linear layers
have no bias so a null-only input produces exactly zero residual.

### Task 4: Review and server handoff

**Files:**
- Review: `layers/cird_layers.py`
- Review: `tests/test_cird_residual_cd_branch.py`
- Review: `docs/superpowers/plans/2026-07-06-residual-cd-stage1.md`

**Interfaces:**
- Produces: one pushed stage-1 commit and server commands

- [ ] **Step 1: Perform static source review without execution**

Check tensor annotations, validation messages, self-mask indexing,
detach placement, and absence of changes outside stage-1 files.

- [ ] **Step 2: Commit and push**

```bash
git add layers/cird_layers.py tests/test_cird_residual_cd_branch.py \
  docs/superpowers/plans/2026-07-06-residual-cd-stage1.md
git commit -m "feat: add standalone residual cd branch"
git push origin cird-dev
```

- [ ] **Step 3: Stop for server verification**

The server runs unit tests, a small shape/backward smoke test, and a
Traffic-sized CUDA memory test. No model/trainer integration begins
until the user returns those results.
