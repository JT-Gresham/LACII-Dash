# Kimi-Linear (hybrid linear-attention MoE) — requirements, gotchas, current status

`moonshotai/Kimi-Linear-48B-A3B-Instruct` is the first **linear-attention hybrid** the project has
ingested. It exercises code paths no dense/MoE model touches, and it fails in ways whose error
messages point somewhere else entirely. This documents every requirement so an install can be
reproduced without rediscovering them.

**Status (2026-08-09): SERVING ✅** — downloads · int4 shard-compile · load · **generation**, all
validated end-to-end on om3nbox. Coherent output on `/api/generate` and `/v1/chat/completions`,
clean `stop`, **~6.5 tok/s** steady-state.

> **Two hard requirements or it will not generate:** `fla-core==**0.4.0**` (NOT `-U` — see below)
> and `triton` on the same box. Both compile and load succeed on the wrong `fla`, so *"it loads"*
> proves nothing.

> **Placement constraint:** the KDA layers' `fla` kernels (`chunk_kda`, `fused_recurrent_kda`,
> `fused_kda_gate`) are **Triton — GPU only, no CPU fallback**. A KDA layer that lands on CPU dies at
> the first forward. On this fleet that means Kimi runs on a box with one big GPU (om3nbox), not
> spread across the iM pool. The loader now logs a build-time WARNING naming the offending layers.

---

## What the architecture actually is

From `config.json` (`model_type: kimi_linear`, `KimiLinearForCausalLM`, 27 layers, hidden 2304,
vocab 163840):

| Property | Value | Consequence for InfiniteModel |
|---|---|---|
| **Hybrid attention** | 7 full-attn layers (`4,8,12,16,20,24,27`); the other **20 are KDA** (Kimi Delta Attention) | KDA layers keep **recurrent state**, not a KV cache |
| **MLA-style full layers** | `q_lora_rank`, `kv_lora_rank`, `qk_rope_head_dim=64`, `qk_nope_head_dim=128`, `head_dim=72` | rotary width is **`qk_rope_head_dim`**, NOT `head_dim` |
| **MoE** | 256 experts + shared experts, `moe_intermediate_size` 1024, **per-expert** (not fused-3D) layout | packer needs the meta skeleton to derive per-expert scope |
| **MTP head** | `num_nextn_predict_layers: 0` | off — `_has_mtp` actually gates on `mtp_num_hidden_layers`, which this config lacks entirely |
| **Remote code** | `auto_map` → `modeling_kimi.py` (trust_remote_code) | pinned to the transformers API of its release |

## Requirements

### 1. `fla-core` — **pin `==0.4.0`, and `-U` is actively wrong**

```bash
<venv>/bin/pip install 'fla-core==0.4.0'      # NOT -U: 0.4.1+ break this checkpoint
```

The model card says `pip install -U fla-core`. **Do not follow it.** `fla-core` changed the
`fused_kda_gate` API one release after Kimi-Linear shipped, and every later version fails at the
first decode step:

```
TypeError: fused_kda_gate() got an unexpected keyword argument 'g_bias'. Did you mean 'dt_bias'?
```

`modeling_kimi.py:560` calls `fused_kda_gate(g, self.A_log, self.head_dim, g_bias=self.dt_bias)`:

| `fla-core` | `fused_kda_gate` signature | Kimi |
|---|---|---|
| 0.3.2 | *(no `ops/kda/gate.py` at all)* | ✗ |
| **0.4.0** (2025-10-27) | `(g, A, head_k_dim, g_bias=None, beta=1.0, threshold=20.0)` | ✅ **the only match** |
| 0.4.1 → 0.5.2 | `(g, A_log, dt_bias=None, lower_bound=None, output_dtype=…)` | ✗ |

It is not just a rename — the **shape contract** changed too (0.4.0 takes `g` as
`[…, H*K]` plus an explicit `head_k_dim`; 0.4.1+ take `[…, H, K]`), and the softplus
parameterisation went from `beta`/`threshold` to `lower_bound`. So a compat shim is **not** provably
equivalent, and a wrong one yields plausible-but-wrong gates with no error. Pin the version.

**Safe for the rest of the fleet:** transformers' own `fla` consumers (`qwen3_next`, `qwen3_5`,
`qwen3_5_moe`, `olmo_hybrid`) import only `FusedRMSNormGated`, `chunk_gated_delta_rule` and
`fused_recurrent_gated_delta_rule`, which are signature-compatible in 0.4.0 — verified by importing
both transformers modules under it.

### 1b. Where each piece is needed — the split that costs time

| Box role | Needs | Why |
|---|---|---|
| **GPU worker holding KDA layers** | `fla-core==0.4.0` **+ `triton`** | runs the KDA kernels. This is the box that must have the *pinned* version |
| **Box that COMPILES shards** (`/compile_shards`) | `fla` **+ `triton`** | building the skeleton executes `modeling_kimi.py`'s module-level imports, and `fla.modules.convolution` imports `triton` **at import time** |
| **CPU-only controller** | *cannot build this arch at all* | no `triton` → `ModuleNotFoundError: No module named 'triton'` on **any** `fla` version. Compile Kimi on a GPU box (`#compile-picker` → one-node), not on the controller |

> ⚠️ The iM controller (VM, CPU-only) has `fla-core` installed but **no `triton`**, so it can never
> build the Kimi skeleton. That is why the int4 compile was run on om3nbox. Installing `fla` alone on
> a controller is *not* sufficient and gives a misleading sense that the box is ready.

⚠️ A KDA layer placed on **CPU** cannot run either — `chunk_kda` / `fused_recurrent_kda` /
`fused_kda_gate` are Triton kernels with no CPU fallback. The loader logs a build-time WARNING naming
the offending global layer indices.

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

There are **two disjoint `EXTRA_UPDATE_FILES` lists**, and confusing them costs an hour:

| List | Where | Contains |
|---|---|---|
| **controller** set | `server.py` | `engine_gen.py`, `engine_load.py`, `placement.py`, `model_store.py`, `shards.py`, … — **not** `shard_build.py` / `shard_forward.py` |
| **worker** set | `client.py` | `shard_build.py`, `shard_forward.py`, `state.py`, `worker_*.py`, … |

`POST /update` (the default, `workers=0`) restarts the **controller only**, so a worker-path fix
staged that way sits on disk while the worker keeps running the old modules in memory — it logs
`staged on disk (VERSION unchanged) - NOT restarting`, and you get the identical original error.

**The deploy that works, with no VERSION bump:**

```bash
curl -X POST "http://<controller>:21434/update?workers=1"
```

`/update?workers=1` sends `{"type":"restart","update":true}`; the worker fetches its own file list
and then `os._exit(42)` **unconditionally**, so it comes back on the new bytes.

Two traps to know: `/restart_node?node=<host>` fetches **no code** (it sends a bare restart and
relaunches whatever is already on disk — it is a step *after* staging, not an alternative); and
`raw.githubusercontent` lags a push by 1–5 minutes, so a "successful" update can silently restart
onto stale bytes. Verify both, always:

```bash
curl -s "http://<controller>:21434/code_manifest?grep=kimi-linear"
```

It reports per-file `sha1`/`mtime`/`grep_hit` and deliberately parses `client.py`'s worker list, so
worker files appear too. Controller-box copies of `shard_build.py`/`shard_forward.py` staying stale
is **expected and harmless** — the controller never runs them.

---

## What shipped: the hybrid cache

```
AttributeError: '_PreallocKVCache' object has no attribute 'conv_states'
```

20 of 27 layers are KDA. Linear attention keeps a **fixed-size recurrent state** plus a
**short-convolution state** (`conv_states`), advanced per token — not a growing per-token KV cache.

The model ships its own `KimiDynamicCache` (`modeling_kimi.py:118`), but **it is deliberately not
used**: its `isinstance` assert lives in `KimiLinearModel.forward`, which a shard never calls (a
shard drives the decoder layers directly), and its `update()` does a `torch.cat` per token — the
O(cache) memcpy `#kv-prealloc` exists to eliminate. Instead the existing prealloc cache gains the two
flat lists Kimi's KDA layers index by **global** layer index.

`#linattn-flat` marks archs that declare their linear layers via `linear_attn_config.kda_layers`
(**1-indexed** — `is_kda_layer` tests `layer_idx + 1`) instead of transformers' `layer_types`. It is
OR-ed into `_hybrid`, so every linear-attn safety gate that already existed for qwen3-next fires with
no new gate code. **0-based** split, for the record: full-attention = `{3,7,11,15,19,23,26}`,
KDA = the other 20.

### Worker

| Change | Why |
|---|---|
| `_PreallocKVLayer` sizes **V from V's own geometry** | it borrowed the *key's* last dim. MLA caches K at `qk_nope+qk_rope` (192) and V at `v_head_dim` (128) → `size of tensor a (192) must match tensor b (128)` on the first full-attn layer. Fixes DeepSeek-style MLA generally; byte-identical for symmetric models |
| `_make_linattn_kv` | prealloc cache + `conv_states`/`recurrent_states`, marked `im_no_crop`. Per-slot for free — `#kv-slots` already binds a cache per request |
| `layer.layer_type` stamped at build | Kimi carries `is_linear_attn`, not `layer_type`. Stamping lets the **existing** hybrid mask dispatch work unchanged, instead of editing `shard_forward`'s two layer loops (shared with Gemma-4 / Omni / VL — the highest-regression-risk code in the repo) |
| `_attn_implementation` restored to `sdpa` | `KimiLinearModel.__init__` **force-overwrites** it to `flash_attention_2` on the *same* config object, and MLA re-reads it every forward. `sdpa` not `eager`: eager would materialize a `[1,32,q,total]` score tensor on a prefill the hybrid path deliberately does not chunk (~8.6 GB at q=8192) |
| `Shard.crop` **drops** a non-rewindable cache | the old blanket `suppress(Exception)` turned "this cache has no `crop()`" into "the controller believes it was cropped" |

### Reservation — the correction that mattered most

The uniform `2 · n_kv · head_dim · 2` formula reads Kimi's `head_dim` as **72**, giving 9,216
B/token/layer. The truth is `32 · (192 + 128) · 2` = **20,480** — a **2.22× under-reserve** that
false-fits narrow stages and OOMs during decode. The existing `#gemma4-kv` per-layer override cannot
rescue it: it needs *both* `self_attn.head_dim` and `num_key_value_groups`, and Kimi's MLA sets the
second but not the first while KDA sets the first but not the second.

- `_kv_dims` folds the asymmetric K/V pair into an **effective head_dim** (160), so every existing
  caller stays a one-line formula.
- `_kv_layer_mask` masks the KDA layers out; their fixed **~1.09 MiB/layer** conv+recurrent state is
  funded explicitly rather than assumed to be zero.
- The planner mirrors both (`spec_from_config` → effective head_dim + `kv_layer_frac` 7/27), so it
  no longer plans 3.9× high on layer count.
- `kv_quant` / `kv_offload` sizing now follows what a hybrid shard **actually allocates** (plain
  bf16 — the hybrid branch precedes both in `shard_forward`). This also closed a latent ~4×
  under-reserve affecting qwen3-next.

### Controller — a recurrent state cannot be rewound

`_crop` truncates the KV layers only, leaving the linear layers at their **pre-crop** position:
silently divergent decode, not a crash. One `_linear_attn_arch(cfg)` sniff (conservative — any
failure means "not rewindable") gates every feature that assumes rewind. Kimi has **no
`layer_types`**, so before this it read as a plain dense model and *every one of these was ON*:

| Feature | Risk if left on |
|---|---|
| `#prefix-kv` cross-request resume | the top silent-corruption path — turn 2 sends `crop(L)` then a `reset=False` suffix while KDA layers still hold all of turn 1 |
| `#pipefill` chunked prefill | chunk-burst ordering across stages is unvalidated for a recurrent state |
| external-draft speculative decode | appends K+1 tokens then crops the rejected tail — irreversible here. **There was no arch gate at all** |
| MTP self-spec | already off via `mtp_num_hidden_layers`, now guarded on its own merits |
| `kv_slots > 1` | per-slot `#prefix-kv` LCP-resume machinery |

### Test

`scratch_kimi_hybrid_test.py` — 30 checks, no network, no model, no GPU. Bit-exact against a
`torch.cat` reference across the capacity-doubling boundary for **both** asymmetric and symmetric
K/V, the arch sniff, and the reservation/planner geometry against the real Kimi-48B numbers.

```bash
python scratch_kimi_hybrid_test.py
```

The same plumbing applies to any other `fla`-based hybrid (GLA, Mamba-hybrids, RWKV-style) that ships
the flat-list cache contract.

---

## Verified numbers (om3nbox / gfx1151, 2026-08-09)

| Step | Result |
|---|---|
| bf16 source | ~96 GB on disk |
| int4 shard cache | **25.4 GB**, 29 units, compile ~24 min (box concurrently serving) |
| load | **25.36 GB**, `cpu_frac=0.0`, all 27 layers on one node (`node=InferenceEngine`) |
| KV reserve @ ctx 8192 | **1.09 GB measured** — only the 7 full-attn layers grow KV. Pre-fix this was 27 layers at the wrong geometry |
| **generation** | ✅ coherent on `/api/generate` **and** `/v1/chat/completions`, clean `stop` |
| decode, steady | **~6.5 tok/s** — 80 tok in 12.2 / 12.3 / 12.7 s, *while co-resident qwen3-30b-a3b served live traffic* |
| decode, FIRST request after load | **~0.9 tok/s** — one-time Triton autotune of the KDA kernels. 6× faster from the 2nd request on. Do not benchmark the first call |
| prefill | 25-token chat prompt, no measurable stall |

### Why 6.5 tok/s and not ~28

`qwen3-30b-a3b` reaches ~28 tok/s on the same box with comparable *active* parameters. Kimi reports
`is_moe=False`: its 256 experts are **per-expert modules**, not the fused-3D layout, so it never
enters the fused-MoE path — the same shape that made MiniMax-M2 unusably slow. That is an **open
optimization, not a defect**, and it is unrelated to the linear-attention work.
