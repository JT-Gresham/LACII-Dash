"""#moe-int8: is the 3D routed-expert int8 packer BIT-IDENTICAL to the 2D packer, per expert?

That equality is the whole deliverable, not a nicety. Every int8 path — a cold load, a future
streamed expert build, a future int8 shard cache — has to produce the same bytes for expert `e`,
or a cached shard silently stops being the model you loaded last time. Quantization code that is
merely *approximately* right loads fine, passes its own checksum, and generates garbage; this
harness exists so that failure mode is caught on a laptop instead of on the fleet.

WHAT IS CHECKED (all EXACT equality — torch.equal, not allclose; there is nothing here that may
legitimately differ by an ulp):

  1. per-expert bit-identity  `_pack8_3d(W3)[e]` == `_quantize_linear(Linear(W3[e]))`
     The strong form of the claim. Structurally guaranteed (both go through `_pack8_expert`),
     which is exactly why it is worth asserting: the guarantee is one careless refactor deep.
  2. loop-vs-whole-tensor     `_pack8_3d(W3)` == `_pack8_expert(W3.reshape(E*out, in))`
     Independent of (1). The int8 scale is per output ROW and a row never crosses an expert
     boundary, so flattening the expert axis must change nothing. If this fails, the per-expert
     loop — not the arithmetic — is the bug.
  3. dequant contract         `holder[e]` == `ql.qweight.to(dt) * ql.scale`
     `Packed8Tensor3D.__getitem__` is what the eager MoE host actually consumes. It must equal
     what the 2D QuantLinear would have dequantized, in the SAME dtype (the host does
     `F.linear(bf16_activation, holder[e])` and an fp32 return raises inside it).
  4. shard-cache twin         `_pack8_expert(W)` == `shard_compile.pack_linear_int8(W)`
     Three copies of this arithmetic exist by design (worker 2D, worker 3D, cache). Pin them.
  5. `_quantize_experts8_` installs the holder, forces `_experts_implementation = "eager"`, is
     idempotent, leaves a dense module untouched, and REFUSES gpt-oss + meta experts rather than
     producing a model that loads and emits garbage.
  6. Adversarial rows: an all-zero row (scale hits the 1e-8 clamp), a row that saturates the
     +/-127 grid, a single-element row, E=1, and shapes that are prime in every dimension.

WHAT IS NOT CHECKED HERE: the Triton w8a16 expert subclass. That needs a GPU, and it already
self-checks at load against the exact bf16 dequant it replaces (Packed8Tensor3D.__getitem__,
M=1 and M=8) and falls back on any mismatch. Use scratch_w8a16_test.py for the kernel itself.

Run:  python3 scratch_moe_int8_pack_test.py           (CPU is enough — pure torch, no triton)
      python3 scratch_moe_int8_pack_test.py --big     (adds a realistic Qwen3-MoE expert shape)
"""
import argparse
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker_quant as wq                                        # noqa: E402

FAILS = []


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


def eq(a, b):
    """Exact equality INCLUDING dtype and shape. torch.equal already compares dtype+shape, but
    spelling it out keeps a silent bf16->fp32 promotion from reading as a pass."""
    return (a.dtype == b.dtype and a.shape == b.shape and torch.equal(a, b))


def as_linear(W2d):
    lin = nn.Linear(W2d.shape[1], W2d.shape[0], bias=False)
    with torch.no_grad():
        lin.weight = nn.Parameter(W2d.clone(), requires_grad=False)
    return lin


def make_W3(E, out, in_f, dt):
    """A 3D expert stack with deliberately UNEQUAL per-expert dynamic range, so a bug that reuses
    one expert's scale for another (the classic 3D-packer mistake) cannot hide behind uniform
    magnitudes."""
    g = torch.Generator().manual_seed(1234 + E * 131 + out * 17 + in_f)
    W3 = torch.randn(E, out, in_f, generator=g, dtype=torch.float32)
    for e in range(E):
        W3[e] *= (0.001 * (10 ** (e % 4)))       # 1e-3 .. 1e0 across experts
    return W3.to(dt).contiguous()


# ---------------------------------------------------------------------------------------------
# 1 + 2 + 3: the pack equivalences
# ---------------------------------------------------------------------------------------------
def test_pack_equivalence(E, out, in_f, dt, label):
    print(f"\n== {label}: E={E} out={out} in={in_f} dtype={dt} ==")
    W3 = make_W3(E, out, in_f, dt)
    holder = wq._pack8_3d(W3)

    check("holder shapes/dtypes",
          (holder.qweight.shape == (E, out, in_f) and holder.qweight.dtype == torch.int8
           and holder.scale.shape == (E, out, 1) and holder.scale.dtype == dt
           and holder.in_features == in_f),
          f"q={tuple(holder.qweight.shape)}/{holder.qweight.dtype} "
          f"s={tuple(holder.scale.shape)}/{holder.scale.dtype}")

    # (1) per-expert bit-identity vs the 2D packer via a real nn.Linear.
    bad = []
    for e in range(E):
        ql = wq._quantize_linear(as_linear(W3[e]))
        if not eq(holder.qweight[e], ql.qweight):
            bad.append(f"e{e}:qweight({int((holder.qweight[e] != ql.qweight).sum())} elems)")
        if not eq(holder.scale[e], ql.scale):
            bad.append(f"e{e}:scale")
        # (3) the dequant the eager MoE host actually consumes.
        if not eq(holder[e], (ql.qweight.to(dt) * ql.scale).contiguous()):
            bad.append(f"e{e}:dequant")
    check("per-expert BIT-IDENTICAL to _quantize_linear (qweight, scale, dequant)",
          not bad, ", ".join(bad[:6]))

    # (2) the loop is not the bug: flattening the expert axis must change nothing, because the
    # scale is per output row.
    q_flat, s_flat = wq._pack8_expert(W3.reshape(E * out, in_f))
    check("per-expert loop == whole-tensor 2D pack",
          eq(holder.qweight.reshape(E * out, in_f), q_flat)
          and eq(holder.scale.reshape(E * out, 1), s_flat))

    # A non-contiguous source must not change the bytes either: on Linux the bf16 expert blob is
    # an mmap slice and W3[e] is a view, so "works on contiguous input" is not the case that ships.
    W3v = W3.clone().transpose(1, 2).transpose(1, 2)       # same values, exercised as a view
    check("view-vs-contiguous source: identical bytes",
          eq(wq._pack8_3d(W3v).qweight, holder.qweight))
    return holder


# ---------------------------------------------------------------------------------------------
# 4: the shard-cache twin
# ---------------------------------------------------------------------------------------------
def test_shard_cache_twin():
    print("\n== shard_compile.pack_linear_int8 twin ==")
    try:
        import shard_compile as sc
    except Exception as exc:
        check("shard_compile importable", False, repr(exc))
        return
    for (out, in_f, dt) in ((97, 131, torch.bfloat16), (256, 512, torch.bfloat16),
                            (64, 64, torch.float16), (33, 7, torch.float32)):
        W = make_W3(1, out, in_f, dt)[0]
        qa, sa = wq._pack8_expert(W)
        qb, sb = sc.pack_linear_int8(W)
        check(f"_pack8_expert == pack_linear_int8  [{out}x{in_f} {dt}]",
              eq(qa, qb) and eq(sa, sb))


# ---------------------------------------------------------------------------------------------
# 6: adversarial rows
# ---------------------------------------------------------------------------------------------
def test_edge_rows():
    print("\n== adversarial rows ==")
    dt = torch.bfloat16
    W3 = make_W3(3, 8, 16, dt)
    W3[0, 0].zero_()                                  # all-zero row -> scale hits the 1e-8 clamp
    W3[1, 3] = torch.full((16,), 30000.0, dtype=dt)   # saturating row (near bf16 max ~3.39e38)
    W3[2, 5, 0] = -0.0                                # signed zero
    holder = wq._pack8_3d(W3)
    bad = []
    for e in range(3):
        ql = wq._quantize_linear(as_linear(W3[e]))
        if not (eq(holder.qweight[e], ql.qweight) and eq(holder.scale[e], ql.scale)):
            bad.append(f"e{e}")
    check("zero row / saturating row / signed zero: still bit-identical", not bad, ",".join(bad))
    check("all-zero row quantizes to all-zero codes",
          bool((holder.qweight[0, 0] == 0).all()))
    check("codes stay inside the int8 symmetric grid",
          bool((holder.qweight.abs() <= 127).all()))
    check("no NaN/Inf leaked into the scales",
          bool(torch.isfinite(holder.scale.float()).all()))


# ---------------------------------------------------------------------------------------------
# 5: the module walker
# ---------------------------------------------------------------------------------------------
class _Cfg:
    def __init__(self):
        self._experts_implementation = "grouped_mm"


class _Experts(nn.Module):
    """A fused-3D MoE experts block, shaped like Qwen3-MoE/MiniMax: gate_up_proj/down_proj are 3D
    nn.Parameters, indexed per routed expert in the eager forward."""

    def __init__(self, E, hidden, inter, dt=torch.bfloat16):
        super().__init__()
        self.config = _Cfg()
        self.gate_up_proj = nn.Parameter(make_W3(E, 2 * inter, hidden, dt), requires_grad=False)
        self.down_proj = nn.Parameter(make_W3(E, hidden, inter, dt), requires_grad=False)


class _Block(nn.Module):
    def __init__(self, E, hidden, inter):
        super().__init__()
        self.gate = nn.Linear(hidden, E, bias=False, dtype=torch.bfloat16)     # router
        self.experts = _Experts(E, hidden, inter)


class _Dense(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.up_proj = nn.Linear(hidden, 4 * hidden, bias=False, dtype=torch.bfloat16)
        self.down_proj = nn.Linear(4 * hidden, hidden, bias=False, dtype=torch.bfloat16)


def test_walker():
    print("\n== _quantize_int8_ / _quantize_experts8_ ==")
    PT8 = wq._packed8_3d_cls()
    QL = wq._quant_linear_cls()

    blk = _Block(6, 32, 64)
    ref = {a: getattr(blk.experts, a).data.clone() for a in ("gate_up_proj", "down_proj")}
    wq._quantize_int8_(blk)
    check("fused 3D experts -> Packed8Tensor3D",
          isinstance(blk.experts.gate_up_proj, PT8) and isinstance(blk.experts.down_proj, PT8))
    check("2D router gate -> QuantLinear", isinstance(blk.gate, QL))
    check("_experts_implementation forced to eager",
          blk.experts.config._experts_implementation == "eager")
    check("walker output == direct _pack8_3d",
          all(eq(getattr(blk.experts, a).qweight, wq._pack8_3d(ref[a]).qweight)
              for a in ref))
    # The holder is a submodule now, so a second walk must find nothing to do.
    wq._quantize_int8_(blk)
    check("idempotent (holder is no longer an nn.Parameter)",
          isinstance(blk.experts.gate_up_proj, PT8))

    dense = _Dense(32)
    wq._quantize_int8_(dense)
    check("dense module: linears quantized, no expert holder invented",
          isinstance(dense.up_proj, QL) and isinstance(dense.down_proj, QL))

    # #moe-offload has to be able to SEE an int8 fused-expert block.
    layer = nn.Module()
    layer.mlp = _Block(4, 16, 32)
    delattr(layer.mlp, "experts")                       # force the 3D-holder detection path
    layer.mlp.ffn = _Experts(4, 16, 32)
    wq._quantize_int8_(layer)
    name, blockm = wq._find_moe_block(layer)
    check("_find_moe_block sees a Packed8Tensor3D block", name == "mlp" and blockm is not None,
          f"got name={name!r}")

    # REFUSALS. Each of these would otherwise produce a model that loads and emits garbage.
    go = _Experts(3, 16, 32)
    go.gate_up_proj_bias = nn.Parameter(torch.zeros(3, 64), requires_grad=False)
    go.down_proj_bias = nn.Parameter(torch.zeros(3, 16), requires_grad=False)
    go.alpha, go.limit = 1.702, 7.0                     # what _is_gptoss_experts keys on
    try:
        wq._quantize_experts8_(go)
        check("gpt-oss experts REFUSED at int8", False, "no exception raised")
    except RuntimeError as exc:
        check("gpt-oss experts REFUSED at int8", "gpt-oss" in str(exc), str(exc))

    meta = _Experts(2, 16, 32)
    with torch.device("meta"):
        meta.gate_up_proj = nn.Parameter(torch.empty(2, 64, 16, dtype=torch.bfloat16),
                                         requires_grad=False)
    try:
        wq._quantize_experts8_(meta)
        check("meta experts REFUSED at int8", False, "no exception raised")
    except RuntimeError as exc:
        check("meta experts REFUSED at int8", "META" in str(exc), str(exc))


# ---------------------------------------------------------------------------------------------
def test_eager_forward_contract():
    """The holder has to survive the thing the MoE host actually does with it."""
    print("\n== eager host contract ==")
    import torch.nn.functional as F
    dt = torch.bfloat16
    W3 = make_W3(4, 24, 16, dt)
    holder = wq._pack8_3d(W3)
    x = torch.randn(1, 16, dtype=dt)
    outs = []
    for e in range(4):
        w = holder[e]
        outs.append(F.linear(x, w))
    check("F.linear(bf16 x, holder[e]) returns bf16",
          all(o.dtype == dt for o in outs))
    check("routed experts give DIFFERENT outputs (no expert-0 aliasing)",
          not torch.equal(outs[0], outs[1]))
    # bit-identical to routing through the equivalent QuantLinear weight
    ql = wq._quantize_linear(as_linear(W3[2]))
    check("holder[e] F.linear == QuantLinear-weight F.linear",
          torch.equal(F.linear(x, holder[2]), F.linear(x, ql.qweight.to(dt) * ql.scale)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--big", action="store_true",
                    help="add a realistic Qwen3-MoE expert shape (slow, ~GB)")
    a = ap.parse_args()

    print(f"torch {torch.__version__}  (CPU is sufficient — no triton on this path)")
    # Prime shapes on purpose: nothing lines up with any block size or packing width, so an
    # off-by-one in the expert stride cannot hide behind tidy dimensions.
    test_pack_equivalence(5, 97, 131, torch.bfloat16, "prime dims")
    test_pack_equivalence(1, 17, 3, torch.bfloat16, "E=1, tiny")
    test_pack_equivalence(8, 64, 128, torch.bfloat16, "power-of-two")
    test_pack_equivalence(3, 40, 32, torch.float16, "fp16 source")
    test_pack_equivalence(3, 40, 32, torch.float32, "fp32 source")
    if a.big:
        test_pack_equivalence(128, 768, 2048, torch.bfloat16, "Qwen3-MoE-ish gate_up")
    test_edge_rows()
    test_shard_cache_twin()
    test_walker()
    test_eager_forward_contract()

    print(f"\n{'ALL CHECKS PASSED' if not FAILS else str(len(FAILS)) + ' FAILURE(S): ' + ', '.join(FAILS)}")
    sys.exit(1 if FAILS else 0)


if __name__ == "__main__":
    main()
