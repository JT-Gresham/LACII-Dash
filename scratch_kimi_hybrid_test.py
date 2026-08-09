"""#kimi-linear scratch validation — hybrid linear-attention cache + reservation. NO network, NO model.

Drives the REAL code paths that make Kimi-Linear (and any fla-style hybrid that declares its linear
layers via linear_attn_config rather than transformers' layer_types) generate on the pipeline:

  a. ASYMMETRIC MLA K/V through the REAL _PreallocKVLayer — Kimi caches K at
     qk_nope+qk_rope (192) and V at v_head_dim (128). The old buffer alloc took V's width from
     the KEY, which raised "size of tensor a (192) must match tensor b (128)" on the first
     full-attention layer. Bit-exact vs a torch.cat reference ACROSS the capacity-doubling
     boundary, both single-token decode and multi-token prefill, plus crop().
  b. SYMMETRIC regression — an ordinary model's K/V shape is unchanged, bit-for-bit.
  c. _linear_attn_arch (engine_gen) — the controller's one source of truth for "this cache is
     NOT rewindable": catches Kimi's linear_attn_config shape, qwen3-next's layer_types shape,
     and a bare model_type; says False for dense/VL/Gemma-style configs; True on garbage.
  d. Reservation geometry — _kv_dims folds MLA's asymmetric pair into an effective head_dim, and
     _kv_layer_mask marks ONLY the full-attention layers as KV-growing (the config lists are
     1-INDEXED). Checked against the real Kimi-48B numbers.
  e. _linattn_state_bytes — the fixed conv+recurrent state a KDA layer holds is funded, not 0.
  f. Planner mirror — model_store.spec_from_config's effective head_dim + kv_layer_frac, and
     ModelSpec.kv_bytes_per_layer, agree with the worker's per-layer figure.
  g. The linattn cache carries the flat conv_states/recurrent_states lists Kimi's KDA layers
     index by GLOBAL layer idx, and is marked non-croppable.

Run:  python scratch_kimi_hybrid_test.py    (prints PASS lines; non-zero exit on any failure)
"""
import sys
import types

import torch

FAILURES: list = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        FAILURES.append(name)
        print(f"  FAIL  {name}  {detail}")


# The real Kimi-Linear-48B-A3B-Instruct config fields this work reads.
KIMI_CFG = {
    "model_type": "kimi_linear",
    "num_hidden_layers": 27,
    "hidden_size": 2304,
    "num_attention_heads": 32,
    "num_key_value_heads": 32,
    "head_dim": 72,
    "qk_nope_head_dim": 128,
    "qk_rope_head_dim": 64,
    "v_head_dim": 128,
    "kv_lora_rank": 512,
    "intermediate_size": 9216,
    "vocab_size": 163840,
    "linear_attn_config": {
        "full_attn_layers": [4, 8, 12, 16, 20, 24, 27],          # 1-INDEXED
        "kda_layers": [1, 2, 3, 5, 6, 7, 9, 10, 11, 13, 14, 15,  # 1-INDEXED
                       17, 18, 19, 21, 22, 23, 25, 26],
        "head_dim": 128,
        "num_heads": 32,
        "short_conv_kernel_size": 4,
    },
}
# 0-based: is_kda_layer(i) tests (i+1) in kda_layers -> full-attn = {3,7,11,15,19,23,26}
KIMI_FULL_ATTN_0B = {3, 7, 11, 15, 19, 23, 26}


def _cfg_obj(d: dict):
    """A config-like object with attribute access + is_kda_layer, as AutoConfig would produce."""
    o = types.SimpleNamespace(**d)

    def is_kda_layer(layer_idx, _o=o):
        lac = getattr(_o, "linear_attn_config", None)
        return lac is not None and (layer_idx + 1) in lac["kda_layers"]

    o.is_kda_layer = is_kda_layer
    return o


# ---------------------------------------------------------------- a / b / g
def test_prealloc_asymmetric():
    print("a. asymmetric MLA K/V through the REAL _PreallocKVLayer")
    from shard_forward import _prealloc_kv_cache_cls
    cls = _prealloc_kv_cache_cls()

    # Kimi MLA geometry: 32 heads, K 192 wide, V 128 wide.
    for label, kd, vd in (("MLA asymmetric (K192/V128)", 192, 128),
                          ("symmetric (K128/V128)", 128, 128)):
        cache = cls()
        layer = cache.layer_class_to_replicate()
        ref_k, ref_v = [], []
        torch.manual_seed(0)
        # 3 tokens of prefill, then single-token decode past the 1024 doubling boundary.
        steps = [3] + [1] * 1100
        for q in steps:
            k = torch.randn(1, 32, q, kd, dtype=torch.bfloat16)
            v = torch.randn(1, 32, q, vd, dtype=torch.bfloat16)
            ref_k.append(k)
            ref_v.append(v)
            gk, gv = layer.update(k, v)
        ek, ev = torch.cat(ref_k, dim=2), torch.cat(ref_v, dim=2)
        check(f"{label}: K bit-exact vs torch.cat ({tuple(gk.shape)})",
              gk.shape == ek.shape and torch.equal(gk, ek))
        check(f"{label}: V bit-exact vs torch.cat ({tuple(gv.shape)})",
              gv.shape == ev.shape and torch.equal(gv, ev))
        check(f"{label}: V kept its OWN width (not the key's)", gv.shape[-1] == vd)
        n = int(gk.shape[2])
        layer.crop(n - 10)
        check(f"{label}: crop() narrows both views",
              layer.keys.shape[2] == n - 10 and layer.values.shape[2] == n - 10
              and torch.equal(layer.values, ev[:, :, :n - 10, :]))


def test_linattn_cache_shape():
    print("g. the linattn cache carries the flat state lists Kimi indexes by GLOBAL idx")
    import shard_forward

    sh = types.SimpleNamespace(cfg=_cfg_obj(KIMI_CFG), torch=torch)
    sh._make_prealloc_kv = types.MethodType(
        shard_forward.ShardForwardMixin._make_prealloc_kv, sh)
    kv = types.MethodType(shard_forward.ShardForwardMixin._make_linattn_kv, sh)()
    check("conv_states is a flat list sized num_hidden_layers",
          isinstance(kv.conv_states, list) and len(kv.conv_states) == 27
          and all(x is None for x in kv.conv_states))
    check("recurrent_states is a flat list sized num_hidden_layers",
          isinstance(kv.recurrent_states, list) and len(kv.recurrent_states) == 27)
    check("marked non-croppable (im_no_crop)", getattr(kv, "im_no_crop", False) is True)
    check("still the #kv-prealloc cache (index-write append, not torch.cat)",
          hasattr(kv, "layer_class_to_replicate")
          and hasattr(kv.layer_class_to_replicate, "_CHUNK"))
    # A mid/tail stage writes only its OWN global slots; the rest stay None.
    kv.recurrent_states[19] = torch.zeros(1)
    check("a global-index write leaves the other slots untouched",
          kv.recurrent_states[19] is not None and kv.recurrent_states[18] is None)


# ---------------------------------------------------------------------- c
def test_linear_attn_arch():
    print("c. _linear_attn_arch — the controller's not-rewindable sniff")
    from engine_gen import _linear_attn_arch
    check("Kimi (linear_attn_config, no layer_types)", _linear_attn_arch(KIMI_CFG) is True)
    check("qwen3-next style (layer_types)",
          _linear_attn_arch({"layer_types": ["linear_attention", "full_attention"]}) is True)
    check("nested under text_config",
          _linear_attn_arch({"text_config": {"layer_types": ["linear_attention"]}}) is True)
    check("model_type alone", _linear_attn_arch({"model_type": "kimi_linear"}) is True)
    check("dense qwen2 -> False",
          _linear_attn_arch({"model_type": "qwen2", "num_hidden_layers": 28}) is False)
    check("Gemma-4 per-type (sliding, NOT linear) -> False",
          _linear_attn_arch({"layer_types": ["sliding_attention", "full_attention"]}) is False)
    check("garbage -> True (conservative)", _linear_attn_arch(None) is True)


# ------------------------------------------------------------------ d / e
def _stub_shard(cfg_d, layer_start, layer_end):
    """A Shard-shaped stub carrying only what the reservation helpers read."""
    import shard_build
    cfg = _cfg_obj(cfg_d)
    sh = types.SimpleNamespace(cfg=cfg, layer_start=layer_start, kv_quant="none", kv_slots=1)
    lac = getattr(cfg, "linear_attn_config", None)
    sh._hybrid = bool(isinstance(lac, dict) and lac.get("kda_layers"))
    sh._linattn_flat = sh._hybrid
    # owned_layers as the real build produces them: modules that know their own type.
    sh.owned_layers = [types.SimpleNamespace(is_linear_attn=cfg.is_kda_layer(i))
                       for i in range(layer_start, layer_end)]
    for m in ("_kv_dims", "_kv_bf16_per_layer", "_kv_bytes_per_layer",
              "_kv_layer_mask", "_linattn_state_bytes"):
        setattr(sh, m, types.MethodType(getattr(shard_build.ShardBuildMixin, m), sh))
    return sh


def test_reservation_geometry():
    print("d. reservation geometry — effective head_dim + KV-layer mask")
    sh = _stub_shard(KIMI_CFG, 0, 27)
    nkv, hd = sh._kv_dims(1)
    check("MLA effective head_dim = ceil((192+128)/2) = 160", (nkv, hd) == (32, 160),
          f"got {(nkv, hd)}")
    per_tok = sh._kv_bf16_per_layer(1)
    truth = 32 * (192 + 128) * 2          # nkv * (K + V) * bf16
    check(f"per-layer bytes/token = {truth} (was 9216, a 2.22x under-reserve)",
          per_tok == truth, f"got {per_tok}")

    mask = sh._kv_layer_mask()
    got_full = {i for i, h in enumerate(mask) if h}
    check("only the 7 full-attention layers grow a KV (0-based, config is 1-INDEXED)",
          got_full == KIMI_FULL_ATTN_0B, f"got {sorted(got_full)}")

    # A narrow stage that owns exactly one full-attention layer — the case the uniform
    # formula under-funded by 2.22x and which no whole-model test can surface.
    narrow = _stub_shard(KIMI_CFG, 3, 4)
    check("narrow stage owning global layer 3 reserves it as full-attention",
          narrow._kv_layer_mask() == [True])
    check("narrow stage per-layer figure is the MLA truth",
          narrow._kv_bf16_per_layer(1) == truth)
    kda_only = _stub_shard(KIMI_CFG, 0, 3)
    check("an all-KDA stage (globals 0-2) reserves NO growing KV",
          kda_only._kv_layer_mask() == [False, False, False])

    print("e. the fixed KDA conv+recurrent state is funded, not treated as 0")
    st = sh._linattn_state_bytes()
    expect = 3 * (32 * 128) * 4 * 2 + 32 * 128 * 128 * 2
    check(f"KDA state = {expect} B/layer ({expect / 2**20:.2f} MiB)", st == expect,
          f"got {st}")
    dense = _stub_shard({"model_type": "qwen2", "num_hidden_layers": 4,
                         "num_attention_heads": 8, "num_key_value_heads": 8,
                         "head_dim": 64, "hidden_size": 512}, 0, 4)
    check("dense model: state bytes 0, mask all-True, head_dim untouched",
          dense._linattn_state_bytes() == 0 and dense._kv_layer_mask() == [True] * 4
          and dense._kv_dims(1) == (8, 64))


# ---------------------------------------------------------------------- f
def test_planner_mirror():
    print("f. planner mirror — spec_from_config + ModelSpec.kv_bytes_per_layer")
    import json
    import os
    import tempfile

    import model_store

    def _spec(name, cfg_d):
        d = tempfile.mkdtemp(prefix="imkimi_")
        with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
            json.dump(cfg_d, fh)
        return model_store._spec_from_config(d, name)

    spec = _spec("kimi-test", dict(KIMI_CFG))
    check("planner head_dim folded to 160", spec.head_dim == 160, f"got {spec.head_dim}")
    check("kv_layer_frac = 7/27", abs(spec.kv_layer_frac - 7 / 27) < 1e-9,
          f"got {spec.kv_layer_frac}")
    # Whole-model KV at ctx=1 must equal the worker's truth: 7 layers x 20480 B.
    whole = spec.num_layers * spec.kv_bytes_per_layer(1)
    truth = 7 * 32 * (192 + 128) * 2
    check(f"whole-model KV/token = {truth} B (worker-exact within rounding)",
          abs(whole - truth) <= spec.num_layers, f"got {whole}")
    dense = _spec("dense-test",
                  {"model_type": "qwen2", "num_hidden_layers": 28, "hidden_size": 3584,
                   "num_attention_heads": 28, "num_key_value_heads": 4, "head_dim": 128,
                   "intermediate_size": 18944, "vocab_size": 152064,
                   "architectures": ["Qwen2ForCausalLM"]})
    check("dense model: frac 1.0 and head_dim untouched",
          dense.kv_layer_frac == 1.0 and dense.head_dim == 128)
    check("dense kv_bytes_per_layer unchanged (2*4*128*ctx*2)",
          dense.kv_bytes_per_layer(1000) == 2 * 4 * 128 * 1000 * 2)


def main():
    print(__doc__.strip().splitlines()[0])
    print()
    test_prealloc_asymmetric()
    test_linattn_cache_shape()
    test_linear_attn_arch()
    test_reservation_geometry()
    test_planner_mirror()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
