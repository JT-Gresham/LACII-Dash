"""#nvfp4-tp test: does `_build_weight_tp_blob` slice an NVFP4 checkpoint on the SAME axis it
slices the equivalent plain-bf16 checkpoint on?

This is the failure mode the test exists for: a TP rank that got `W[:, off:off+len]` where it
should have got `W[off:off+len, :]` still has the right *shape* on every projection where the two
dims happen to be equal (q_proj and o_proj of any square-attention model), still passes
`load_state_dict`, still checksums, and still generates — garbage. Nothing downstream notices.
So "it loads" proves nothing and is not accepted as evidence anywhere below.

The argument the tests make, in order of strength:

  1. EQUIVALENCE (test_nvfp4_matches_bf16). Build the same logical weight matrix TWICE: once as an
     NVFP4 checkpoint (packed FP4 + fp8 block scale + f32 global scale), once as a plain-bf16
     checkpoint holding exactly `_dequant_nvfp4_to_bf16`'s output. Serve both through
     `_build_weight_tp_blob` at every rank and assert the two blobs are BIT-IDENTICAL, key for key.
     That covers axis, offsets, bias slicing/dropping, the `weight_packed`->`weight` rename, which
     tensors are replicated, and dtype — all at once, against a path that has been serving on this
     fleet for a long time.

  2. INDEPENDENT AXIS RECONSTRUCTION (test_axis_reconstruction). Equivalence would still hold if
     BOTH paths sliced the wrong axis, so this one does not use the shared code as its oracle:
     concatenate the per-rank slices along the axis the TP algebra requires — dim 0 for
     column-parallel (q/k/v/gate/up, each rank owns output rows), dim 1 for row-parallel (o/down,
     each rank owns input columns) — and assert the result equals the full dequantized matrix.

  3. REFUSALS (test_refuses_*). The three cases the implementation deliberately does NOT guess at:
     a quantized LM head (which `_is_tied` would silently read as tied embeddings and answer with
     the embedding matrix), a missing scale sidecar, and a non-2D packed tensor (fused 3D MoE).

  4. PURE (pure_checks). The torch-free half: name classification, sidecar-drop predicate, geometry.
     Runs anywhere, including a box with no torch.

STATUS ON MOBILE (2026-08-16): only `pure_checks()` has been RUN here — MOBILE has numpy but
neither torch nor safetensors installed, so tests 1-3 are UNRUN. They need a box with torch
(any build with `torch.float8_e4m3fn`, i.e. >= 2.1) plus `safetensors`; it is pure CPU tensor math,
no GPU and no model, so any fleet node qualifies — e.g. beast (192.168.15.38) or om3nbox.

Run:  python3 scratch_nvfp4_tp_test.py --pure     (torch-free checks only)
      python3 scratch_nvfp4_tp_test.py            (everything; needs torch + safetensors)
"""
import argparse
import json
import os
import sys
import tempfile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import shards                                                     # noqa: E402
from shards import (_is_fp8_meta_name, _nvfp4_global_scale_name, _nvfp4_group_size,  # noqa: E402
                    _nvfp4_scale_name, _tp_geo_for_rank, _tp_kind_and_slice)

# Small but NOT square where it matters, and no dim is a multiple of another: a transposed slice
# must change the shape on every tensor here, so a wrong axis cannot hide behind a lucky dimension.
NH, NKV, HD = 4, 2, 32          # q dim = 128, kv dim = 64
HIDDEN, INTER, VOCAB = 128, 96, 64
TP = 2
GROUP = 16                      # nvfp4 group size (compressed-tensors nvfp4-pack-quantized)

# (name, out_features, in_features, expected TP kind). in_features must be a multiple of GROUP.
PROJ = [("self_attn.q_proj",  NH * HD,  HIDDEN,  "col"),
        ("self_attn.k_proj",  NKV * HD, HIDDEN,  "col"),
        ("self_attn.v_proj",  NKV * HD, HIDDEN,  "col"),
        ("self_attn.o_proj",  HIDDEN,   NH * HD, "row"),
        ("mlp.gate_proj",     INTER,    HIDDEN,  "col"),
        ("mlp.up_proj",       INTER,    HIDDEN,  "col"),
        ("mlp.down_proj",     HIDDEN,   INTER,   "row")]


def _config(quant: bool, tied: bool = False) -> dict:
    cfg = {"model_type": "qwen3", "architectures": ["Qwen3ForCausalLM"],
           "num_hidden_layers": 1, "hidden_size": HIDDEN, "num_attention_heads": NH,
           "num_key_value_heads": NKV, "head_dim": HD, "intermediate_size": INTER,
           "vocab_size": VOCAB, "tie_word_embeddings": tied}
    if quant:
        # Exactly the shape `_nvfp4_group_size` probes for (unsloth/NVIDIA compressed-tensors).
        cfg["quantization_config"] = {
            "quant_method": "compressed-tensors", "format": "nvfp4-pack-quantized",
            "config_groups": {"group_0": {"weights": {"num_bits": 4, "type": "float",
                                                      "group_size": GROUP}}}}
    return cfg


# --------------------------------------------------------------------------------------------
# Pure (torch-free) checks — the name/axis/geometry contract, runnable on a box with no torch.
# --------------------------------------------------------------------------------------------
def pure_checks() -> None:
    d = tempfile.mkdtemp(prefix="nvfp4_pure_")
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(_config(quant=True), fh)
    assert _nvfp4_group_size(d) == GROUP, _nvfp4_group_size(d)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(_config(quant=False), fh)
    assert _nvfp4_group_size(d) is None, "a bf16 checkpoint must not be read as nvfp4"

    # Sidecar names + the drop predicate. `weight_packed` must SURVIVE selection (it is the weight);
    # all three scale tensors must be dropped, or they land in the blob as keys the worker has no
    # slot for.
    p = "model.layers.0.self_attn.q_proj.weight_packed"
    assert _nvfp4_scale_name(p) == "model.layers.0.self_attn.q_proj.weight_scale"
    assert _nvfp4_global_scale_name(p) == "model.layers.0.self_attn.q_proj.weight_global_scale"
    assert not _is_fp8_meta_name(p)
    for s in ("weight_scale", "weight_global_scale", "input_global_scale", "input_scale"):
        assert _is_fp8_meta_name(f"model.layers.0.self_attn.q_proj.{s}"), s

    # Classification is name-SUFFIX-blind: '...q_proj.weight_packed' and '...q_proj.weight' must get
    # the identical (kind, off, len). This is what makes it safe to dequantize first and rename
    # after — and it is why the implementation renames BEFORE calling _tp_kind_and_slice rather than
    # relying on that blindness.
    for rank in range(TP):
        geo = _tp_geo_for_rank(d, rank, TP, None)
        for name, out_f, in_f, want_kind in PROJ:
            base = f"model.layers.0.{name}."
            k_w, o_w, l_w = _tp_kind_and_slice(base + "weight", geo)
            k_p, o_p, l_p = _tp_kind_and_slice(base + "weight_packed", geo)
            assert (k_w, o_w, l_w) == (k_p, o_p, l_p), f"{name}: rename changed the slice"
            assert k_w == want_kind, f"{name}: kind {k_w} != {want_kind}"
            dim_len = out_f if want_kind == "col" else in_f     # col slices dim0, row slices dim1
            assert l_w == dim_len // TP, f"{name}: len {l_w} != {dim_len // TP}"
            assert o_w == rank * (dim_len // TP), f"{name}: off {o_w}"
    # Replicated set: nothing outside the projections may be sliced.
    geo = _tp_geo_for_rank(d, 0, TP, None)
    for n in ("model.embed_tokens.weight", "model.norm.weight", "lm_head.weight",
              "model.layers.0.input_layernorm.weight",
              "model.layers.0.self_attn.q_norm.weight"):
        assert _tp_kind_and_slice(n, geo)[0] is None, n
    print("pure_checks: OK (name classification, sidecar drop, TP geometry)")


# --------------------------------------------------------------------------------------------
# Checkpoint builders (need torch + safetensors)
# --------------------------------------------------------------------------------------------
def _build_pair(root: str, seed: int = 0):
    """Write TWO checkpoints of the SAME logical model into `root`:
      root/nvfp4/ — every projection stored as weight_packed + weight_scale + weight_global_scale
      root/bf16/  — every projection stored as a plain bf16 '.weight' equal to
                    `_dequant_nvfp4_to_bf16` of the nvfp4 one.
    Returns (nvfp4_dir, bf16_dir, {proj_name: full bf16 matrix}).

    The nvfp4 dir is SHARDED with the scale sidecars deliberately placed in a DIFFERENT file from
    their packed weights, to exercise `_companion`'s weight-map fallback — a real 27B checkpoint
    does split a weight from its scale across shard boundaries, and a same-file-only lookup would
    read None and silently fall through."""
    import torch
    from safetensors.torch import save_file
    g = torch.Generator().manual_seed(seed)
    nv, bf = os.path.join(root, "nvfp4"), os.path.join(root, "bf16")
    os.makedirs(nv, exist_ok=True)
    os.makedirs(bf, exist_ok=True)

    # fp8-e4m3 block scales: a handful of exactly-representable positive values (never 0 -> no
    # divide-by-zero surprises, and the reference stays exact).
    fp8_pool = torch.tensor([0.5, 0.75, 1.0, 1.5, 2.0, 3.0], dtype=torch.float32)
    # Shape [1], not 0-dim: real checkpoints ship it either way and `_dequant_nvfp4_to_bf16`
    # reshape(())s it regardless, but a 1-element tensor is the shape safetensors definitely round-
    # trips, and this test must not fail for a reason that has nothing to do with the TP slice.
    gscale = torch.tensor([5.0], dtype=torch.float32)         # per-tensor global scale

    packed_sd, scale_sd, plain_sd, full = {}, {}, {}, {}
    for name, out_f, in_f, _kind in PROJ:
        src = f"model.layers.0.{name}"
        packed = torch.randint(0, 256, (out_f, in_f // 2), generator=g, dtype=torch.uint8)
        idx = torch.randint(0, fp8_pool.numel(), (out_f, in_f // GROUP), generator=g)
        bscale = fp8_pool[idx].to(torch.float8_e4m3fn)
        w = shards._dequant_nvfp4_to_bf16(packed, bscale, gscale, GROUP, [out_f, in_f])
        assert w.shape == (out_f, in_f) and w.dtype == torch.bfloat16, (w.shape, w.dtype)
        packed_sd[src + ".weight_packed"] = packed
        scale_sd[src + ".weight_scale"] = bscale
        scale_sd[src + ".weight_global_scale"] = gscale.clone()
        # An unused W4A4 activation scale, present in real checkpoints; must be dropped, not served.
        scale_sd[src + ".input_global_scale"] = gscale.clone()
        plain_sd[src + ".weight"] = w
        full[name] = w

    def _rest():
        """Tensors both checkpoints share verbatim: embeddings, norms, head, and two biases chosen
        to exercise BOTH bias rules — a column-parallel bias (sliced with its weight) and a
        row-parallel bias (dropped, because the row reduction adds it exactly once)."""
        r = {"model.embed_tokens.weight": torch.randn(VOCAB, HIDDEN, generator=g).to(torch.bfloat16),
             "model.norm.weight": torch.randn(HIDDEN, generator=g).to(torch.bfloat16),
             "lm_head.weight": torch.randn(VOCAB, HIDDEN, generator=g).to(torch.bfloat16),
             "model.layers.0.input_layernorm.weight":
                 torch.randn(HIDDEN, generator=g).to(torch.bfloat16),
             "model.layers.0.post_attention_layernorm.weight":
                 torch.randn(HIDDEN, generator=g).to(torch.bfloat16),
             "model.layers.0.self_attn.q_proj.bias":
                 torch.randn(NH * HD, generator=g).to(torch.bfloat16),
             "model.layers.0.self_attn.o_proj.bias":
                 torch.randn(HIDDEN, generator=g).to(torch.bfloat16)}
        return r

    rest = _rest()
    # nvfp4: two shards, weights in one, scales in the other (cross-file sidecar lookup).
    save_file({**packed_sd, **rest}, os.path.join(nv, "model-00001-of-00002.safetensors"))
    save_file(scale_sd, os.path.join(nv, "model-00002-of-00002.safetensors"))
    wm = {k: "model-00001-of-00002.safetensors" for k in {**packed_sd, **rest}}
    wm.update({k: "model-00002-of-00002.safetensors" for k in scale_sd})
    with open(os.path.join(nv, "model.safetensors.index.json"), "w", encoding="utf-8") as fh:
        json.dump({"metadata": {}, "weight_map": wm}, fh)
    with open(os.path.join(nv, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(_config(quant=True), fh)

    # bf16: the same logical model, single file, no quantization_config.
    save_file({**plain_sd, **rest}, os.path.join(bf, "model.safetensors"))
    with open(os.path.join(bf, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(_config(quant=False), fh)
    return nv, bf, full


def _blob_sd(blob: bytes) -> dict:
    from safetensors.torch import load
    return load(blob)


# --------------------------------------------------------------------------------------------
def test_nvfp4_matches_bf16(root: str) -> None:
    """(1) Bit-identity against the plain-bf16 serve of the identical logical weights, per rank."""
    import torch
    nv, bf, _full = _build_pair(root)
    for rank in range(TP):
        a = _blob_sd(shards._build_weight_tp_blob(nv, 0, 1, True, True, rank, TP))
        b = _blob_sd(shards._build_weight_tp_blob(bf, 0, 1, True, True, rank, TP))
        assert set(a) == set(b), (f"rank {rank} key mismatch: nvfp4-only={sorted(set(a) - set(b))}, "
                                  f"bf16-only={sorted(set(b) - set(a))}")
        assert not [k for k in a if "scale" in k.rsplit(".", 1)[-1]], \
            f"scale sidecars leaked into the blob: {[k for k in a if 'scale' in k]}"
        assert not [k for k in a if k.endswith("_packed")], "'*_packed' key survived the rename"
        for k in sorted(a):
            assert a[k].dtype == b[k].dtype, f"{k}: {a[k].dtype} != {b[k].dtype}"
            assert a[k].shape == b[k].shape, f"{k}: {tuple(a[k].shape)} != {tuple(b[k].shape)}"
            # Bit-identical, not close: both sides are bf16 and slicing is exact, so any difference
            # at all is a real divergence, not rounding.
            assert torch.equal(a[k].view(torch.uint8), b[k].view(torch.uint8)), f"{k}: bytes differ"
        # The row-parallel bias must be gone and the column one sliced (the bf16 path is the oracle
        # for equality above; assert the ABSOLUTE rule here too so a shared regression is caught).
        assert "model.layers.0.self_attn.o_proj.bias" not in a, "row-parallel bias must be dropped"
        assert a["model.layers.0.self_attn.q_proj.bias"].shape[0] == NH * HD // TP
    print(f"test_nvfp4_matches_bf16: OK (ranks 0..{TP - 1} bit-identical to the bf16 serve)")


def test_axis_reconstruction(root: str) -> None:
    """(2) Oracle-free axis check: re-assemble the full matrix from the rank slices."""
    import torch
    nv, _bf, full = _build_pair(root, seed=1)
    blobs = [_blob_sd(shards._build_weight_tp_blob(nv, 0, 1, True, True, r, TP)) for r in range(TP)]
    for name, out_f, in_f, kind in PROJ:
        key = f"model.layers.0.{name}.weight"
        parts = [b[key] for b in blobs]
        dim = 0 if kind == "col" else 1
        # Shape check first: it is the assertion that fails LOUDEST on a swapped axis for the
        # non-square projections, before any value comparison.
        want = (out_f // TP, in_f) if kind == "col" else (out_f, in_f // TP)
        for r, p in enumerate(parts):
            assert tuple(p.shape) == want, f"{name} rank {r}: {tuple(p.shape)} != {want}"
        merged = torch.cat(parts, dim=dim)
        assert torch.equal(merged.view(torch.uint8), full[name].view(torch.uint8)), \
            f"{name}: cat(dim={dim}) of the rank slices != the full dequantized matrix"
    print("test_axis_reconstruction: OK (col=dim0, row=dim1, slices re-assemble exactly)")


def test_refuses_quantized_head(root: str) -> None:
    """(3a) A quantized LM head must RAISE, not be silently served as tied embeddings."""
    import torch
    from safetensors.torch import load_file, save_file
    nv, _bf, _full = _build_pair(root, seed=2)
    d = os.path.join(root, "qhead")
    os.makedirs(d, exist_ok=True)
    sd = {}
    for fn in sorted(os.listdir(nv)):
        if fn.endswith(".safetensors"):
            sd.update(load_file(os.path.join(nv, fn)))
    head = sd.pop("lm_head.weight")
    # Replace the bf16 head with a packed one: this is exactly the state that makes _is_tied()
    # conclude "no head tensor -> tied" and serve model.embed_tokens.weight as the head.
    sd["lm_head.weight_packed"] = torch.randint(0, 256, (head.shape[0], head.shape[1] // 2),
                                                dtype=torch.uint8)
    sd["lm_head.weight_scale"] = torch.ones(head.shape[0], head.shape[1] // GROUP,
                                            dtype=torch.float32).to(torch.float8_e4m3fn)
    sd["lm_head.weight_global_scale"] = torch.tensor([5.0])
    save_file(sd, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(_config(quant=True), fh)
    try:
        shards._build_weight_tp_blob(d, 0, 1, True, True, 0, TP)
    except NotImplementedError as e:
        assert "lm_head" in str(e) or "LM head" in str(e), str(e)
        print("test_refuses_quantized_head: OK (raised NotImplementedError)")
        return
    raise AssertionError("a quantized lm_head was served silently — this is the wrong-logits bug")


def test_refuses_missing_scale(root: str) -> None:
    """(3b) A packed weight whose scale sidecar is absent must RAISE, not be served packed."""
    from safetensors.torch import load_file, save_file
    nv, _bf, _full = _build_pair(root, seed=3)
    d = os.path.join(root, "noscale")
    os.makedirs(d, exist_ok=True)
    sd = {}
    for fn in sorted(os.listdir(nv)):
        if fn.endswith(".safetensors"):
            sd.update(load_file(os.path.join(nv, fn)))
    sd.pop("model.layers.0.mlp.down_proj.weight_global_scale")
    save_file(sd, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(_config(quant=True), fh)
    try:
        shards._build_weight_tp_blob(d, 0, 1, True, True, 0, TP)
    except KeyError as e:
        assert "down_proj" in str(e), str(e)
        print("test_refuses_missing_scale: OK (raised KeyError)")
        return
    raise AssertionError("a packed weight with no scale was served — the bytes are garbage as bf16")


def test_refuses_3d_packed(root: str) -> None:
    """(3c) A fused 3D MoE packed tensor must RAISE — the decoder only knows 2D [out, in//2]."""
    import torch
    from safetensors.torch import load_file, save_file
    nv, _bf, _full = _build_pair(root, seed=4)
    d = os.path.join(root, "moe3d")
    os.makedirs(d, exist_ok=True)
    sd = {}
    for fn in sorted(os.listdir(nv)):
        if fn.endswith(".safetensors"):
            sd.update(load_file(os.path.join(nv, fn)))
    sd["model.layers.0.mlp.experts.gate_up_proj.weight_packed"] = torch.randint(
        0, 256, (2, INTER, HIDDEN // 2), dtype=torch.uint8)
    save_file(sd, os.path.join(d, "model.safetensors"))
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(_config(quant=True), fh)
    try:
        shards._build_weight_tp_blob(d, 0, 1, True, True, 0, TP)
    except NotImplementedError as e:
        assert "ndim" in str(e) or "2D" in str(e), str(e)
        print("test_refuses_3d_packed: OK (raised NotImplementedError)")
        return
    raise AssertionError("a 3D packed tensor was unpacked as 2D — wrong shape, still loads")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pure", action="store_true",
                    help="run only the torch-free checks (MOBILE has no torch/safetensors)")
    args = ap.parse_args()
    pure_checks()
    if args.pure:
        return 0
    try:
        import torch                                             # noqa: F401
        import safetensors                                       # noqa: F401
    except Exception as e:
        print(f"SKIP (UNRUN): torch/safetensors unavailable here ({e}). "
              "Run on a node with torch >= 2.1 (float8_e4m3fn) + safetensors; CPU is enough.")
        return 2
    with tempfile.TemporaryDirectory(prefix="nvfp4_tp_") as root:
        test_nvfp4_matches_bf16(root)
        test_axis_reconstruction(root)
        test_refuses_quantized_head(root)
        test_refuses_missing_scale(root)
        test_refuses_3d_packed(root)
    print("ALL OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
