# Kimi-Linear (hybrid linear-attention MoE) — requirements, gotchas, current status

`moonshotai/Kimi-Linear-48B-A3B-Instruct` is the first **linear-attention hybrid** the project has
ingested. It exercises code paths no dense/MoE model touches, and it fails in ways whose error
messages point somewhere else entirely. This documents every requirement so an install can be
reproduced without rediscovering them.

**Status (2026-08-09):** downloads ✅ · int4 shard-compile ✅ · load ✅ (25.4 GB, fully GPU-resident)
· **generation ❌ — blocked on linear-attention recurrent state (see "Why it cannot serve yet")**.

---

## What the architecture actually is

From `config.json` (`model_type: kimi_linear`, `KimiLinearForCausalLM`, 27 layers, hidden 2304,
vocab 163840):

| Property | Value | Consequence for InfiniteModel |
|---|---|---|
| **Hybrid attention** | 7 full-attn layers (`4,8,12,16,20,24,27`); the other **20 are KDA** (Kimi Delta Attention) | KDA layers keep **recurrent state**, not a KV cache |
| **MLA-style full layers** | `q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim=64`, `qk_nope_head_dim=128`, `head_dim=72` | rotary width is **`qk_rope_head_dim`**, NOT `head_dim` |
| **MoE** | 256 experts + shared experts, `moe_intermediate_size` 1024, **per-expert** (not fused-3D) layout | packer needs the meta skeleton to derive per-expert scope |
| **MTP head** | `num_nextn_predict_layers` | unused today (see #91 MTP self-spec) |
| **Remote code** | `auto_map` → `modeling_kimi.py` (trust_remote_code) | pinned to the transformers API of its release |

## Requirements

### 1. `fla-core` (flash-linear-attention) — mandatory

```bash
<venv>/bin/pip install -U fla-core        # installs the `fla` module (0.5.2 verified)
```

`modeling_kimi.py` imports `fla` for the KDA kernels and raises
`ImportError("Plese run pip install -U fla-core")` without it.

**Install it on every box that BUILDS the architecture**, which is more than people expect:

* the **controller** — `/compile_shards` builds the meta skeleton to derive the per-expert scope
* every **worker** that may hold a shard — `shard_build.from_stream` builds the real arch there

### 2. transformers compat shim — shipped in-repo

`modeling_kimi.py` does `from transformers.utils.generic import OutputRecorder, check_model_inputs`.
In transformers 5.x `check_model_inputs` is still there but **`OutputRecorder` moved to
`transformers.modeling_utils`**. `shards._tf_compat_shims()` re-exports it under the old name and is
called from **both** build paths (`shards._skeleton_from_cfg` for the controller,
`shard_build.from_stream` for the worker).

### 3. A model-level `rotary_emb` — shipped in-repo

The shared forward calls `model.model.rotary_emb`. `KimiLinearModel` has none, so the worker
synthesizes one. Two config quirks had to be handled:

* `KimiLinearConfig` has **no `max_position_embeddings`** (it uses `model_max_length`), which made
  `LlamaRotaryEmbedding.__init__` raise
* the rotary width is `qk_rope_head_dim` (64), **not** `head_dim` (72)

---

## Gotchas — errors that name the wrong thing

These cost real debugging time. The message you see is never the actual cause.

| You see | Actual cause |
|---|---|
| `per-expert MoE shard compile needs the model skeleton (it failed to build …)` | remote code failed to **import** — missing `fla`, or the moved `OutputRecorder` |
| `no room for the new model and resident model(s) are kept (auto-unload off)` | a **worker-side build ImportError**. Seen on a node with **0 GB used and 96 GB free** — it is not a capacity message |
| `'KimiLinearModel' object has no attribute 'rotary_emb'` (at first generate, load was fine) | rotary synthesis failed at BUILD time and was `contextlib.suppress`-ed. Now logs a build-time WARNING instead |
| `'_PreallocKVCache' object has no attribute 'conv_states'` | the real architectural gap — see below |

### Deploy gotcha

`POST /update` restarts the **controller only**, and **`shard_build.py` is not in its update set**.
A fix touching the worker build path needs the file deployed to the worker box *and*
`POST /restart_node?node=<host>` (or `/restart?workers=1`). Symptom of getting this wrong: the shim
is present in `shards.py` on disk, yet the worker still throws the old ImportError.

---

## Why it cannot serve yet

```
AttributeError: '_PreallocKVCache' object has no attribute 'conv_states'
```

20 of 27 layers are KDA. Linear attention does not keep a growing per-token KV cache — it keeps a
**fixed-size recurrent state** plus **short-convolution state** (`conv_states`), advanced per token.
InfiniteModel's `_PreallocKVCache` implements standard attention KV only.

Making Kimi-Linear serve needs a **hybrid cache**, which is a real feature, not a patch:

1. carry `conv_states` + recurrent state for KDA layers alongside KV for the 7 full-attn layers
2. make that state travel correctly across **pipeline stages** — each stage owns a layer range, and
   the recurrent state is per-request and strictly sequential (it cannot be recomputed from
   positions the way KV can)
3. define crop/rollback semantics for it, or explicitly disable the features that rely on rewinding
   — `#prefix-kv` reuse, `_crop`, and speculative decode all assume a rewindable KV cache

Until then the model downloads, compiles to int4, and loads fully GPU-resident — it just cannot
generate. The same applies to any other `fla`-based hybrid (GLA, Mamba-hybrids, RWKV-style).

---

## Verified numbers (om3nbox / gfx1151, 2026-08-09)

| Step | Result |
|---|---|
| bf16 source | ~96 GB on disk |
| int4 shard cache | **25.4 GB**, 29 units, compile ~24 min (box concurrently serving) |
| load | **25.36 GB**, `cpu_frac=0.0`, all 27 layers on one node |
| generation | fails — `conv_states` |
