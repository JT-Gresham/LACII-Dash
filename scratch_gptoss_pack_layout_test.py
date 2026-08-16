"""Bit-identity proof for the IN-major (gpt-oss) fused-expert pack: compile side vs worker cold path.

The shard cache's ENTIRE contract is that a cached shard equals what a cold load would have
quantized. For gpt-oss that equality has a layout in it: the checkpoint stores fused experts
[E, in, out] (it applies them as `y = x @ W`), so worker_quant._quantize_experts4_ packs
`p.data.transpose(1, 2).contiguous()` through `_pack4_3d`, while shard_compile.pack_linear_int4_3d
packs each expert as [out, in]. This test pins `pack_linear_int4_3d(..., in_major=True)` to the
worker's recipe BYTE FOR BYTE — not "close", not "same shape": torch.equal on the uint8 qweight and
on the bf16 scale/zero. A near-miss here is a model that loads, passes its own sha256, and generates
garbage, so an approximate check would be worth nothing.

Compares against the REAL worker functions (`worker_quant._pack4_3d`), never a local transcription
of them — a transcription would only prove this file agrees with itself.

Includes a NEGATIVE CONTROL (case 3): packing the same in-major tensor WITHOUT the transpose must
NOT match. It uses a SQUARE expert ([E, n, n], the gpt-oss-20b down_proj shape where in == out ==
2880), because that is precisely the case a shape check cannot see — the missed transpose produces
a correctly-shaped, wrong-valued cache. Without this control, cases 1-2 could pass vacuously.

    python3 scratch_gptoss_pack_layout_test.py        # needs torch; CPU is enough, no GPU

Runs anywhere torch + psutil are installed: om3nbox (~/imenv/bin/python), beast, amdcomp, or any
worker node. No GPU, no triton, no ROCm — every function under test is pure CPU tensor math.
"""
import sys

import torch

import shard_compile as SC
import worker_quant as WQ

G = 128            # INT4_GROUP; shards.INT4_GROUP == worker_quant._INT4_GROUP == 128


def _rand(*shape):
    """Deterministic bf16 source. Seeded per-call so every case sees the same well-conditioned
    values regardless of test order (bit-identity claims must not depend on execution order)."""
    torch.manual_seed(1234)
    return (torch.randn(*shape) * 0.05).to(torch.bfloat16)


def _worker_pack(W3_out_major):
    """The worker's cold-path result for an already-[E, out, in] tensor, as plain tensors."""
    h = WQ._pack4_3d(W3_out_major, G)          # Packed4Tensor3D (buffers registered)
    return h.qweight, h.scale, h.zero, h.in_features


def _same(tag, got, want):
    """Exact equality on the three packed tensors + in_features. torch.equal, never allclose."""
    (gq, gs, gz, gi), (wq, ws, wz, wi) = got, want
    for nm, a, b in (("qweight", gq, wq), ("scale", gs, ws), ("zero", gz, wz)):
        if a.shape != b.shape:
            return f"{tag}: {nm} shape {tuple(a.shape)} != {tuple(b.shape)}"
        if a.dtype != b.dtype:
            return f"{tag}: {nm} dtype {a.dtype} != {b.dtype}"
        if not torch.equal(a, b):
            n = int((a != b).sum())
            return f"{tag}: {nm} differs in {n}/{a.numel()} elements"
    if gi != wi:
        return f"{tag}: in_features {gi} != {wi}"
    return None


def case1_default_layout_unchanged():
    """Adding `in_major` must not have touched the default path. A plain [E, out, in] fused expert
    (Qwen3.6 / fused Mixtral) packs exactly as the worker's _pack4_3d does, as it did before."""
    W3 = _rand(4, 128, 320)                       # [E, out, in], in % G != 0 -> exercises F.pad
    q, s, z, in_f, ng = SC.pack_linear_int4_3d(W3, G)
    err = _same("case1 default", (q, s, z, in_f), _worker_pack(W3))
    assert ng == (320 + G - 1) // G, f"case1: ng {ng}"
    return err


def case2_in_major_matches_worker(name, E, in_f, out_f):
    """THE contract. An IN-major [E, in, out] expert packed with in_major=True must equal the
    worker's `p.data.transpose(1, 2).contiguous()` -> _pack4_3d, bit for bit."""
    W3 = _rand(E, in_f, out_f)                    # IN-major, as gpt-oss stores it
    got = SC.pack_linear_int4_3d(W3, G, in_major=True)
    q, s, z, gin_f, ng = got
    want = _worker_pack(W3.transpose(1, 2).contiguous())   # worker_quant._quantize_experts4_ recipe
    err = _same(f"case2 {name}", (q, s, z, gin_f), want)
    if err:
        return err
    if gin_f != in_f:
        return f"case2 {name}: in_features {gin_f} != {in_f} (transpose lost the axis)"
    if tuple(q.shape[:2]) != (E, out_f):
        return f"case2 {name}: qweight {tuple(q.shape)} is not [E={E}, out={out_f}, ...]"
    if ng != (in_f + G - 1) // G:
        return f"case2 {name}: ng {ng} != {(in_f + G - 1) // G}"
    return None


def case3_negative_control():
    """The control that makes cases 1-2 mean something. SQUARE expert (in == out, the gpt-oss-20b
    down_proj shape): packing it in-major-unaware yields the RIGHT SHAPE and the WRONG VALUES, which
    is the exact silent corruption the guard exists to prevent. If this assertion ever fails, the
    test above is vacuous — it would be passing without the transpose doing anything."""
    n = 256
    W3 = _rand(3, n, n)
    want = _worker_pack(W3.transpose(1, 2).contiguous())
    naive = SC.pack_linear_int4_3d(W3, G)                     # no in_major -> packs the wrong axis
    if _same("neg", (naive[0], naive[1], naive[2], naive[3]), want) is None:
        return ("case3 negative control: an in-major tensor packed WITHOUT the transpose matched "
                "the worker anyway — the test cannot detect a missing transpose (vacuous)")
    fixed = SC.pack_linear_int4_3d(W3, G, in_major=True)      # ...and with it, it matches
    return _same("case3 square in_major", (fixed[0], fixed[1], fixed[2], fixed[3]), want)


def case4_detector_and_guard():
    """`_in_major_expert_names` must flag a gpt-oss unit (fused 3D weight + per-expert `_bias`
    sibling) and must NOT flag a plain fused MoE — and `pack_unit_tensors` must REFUSE the former
    rather than write a transposed cache. Names mirror the real 'model.*' unit keys."""
    p = "model.layers.0.mlp.experts."
    go = {p + "gate_up_proj": _rand(4, 96, 128), p + "gate_up_proj_bias": _rand(4, 128),
          p + "down_proj": _rand(4, 64, 96), p + "down_proj_bias": _rand(4, 96)}
    plain = {p + "gate_up_proj": _rand(4, 128, 96), p + "down_proj": _rand(4, 96, 64)}
    hit = SC._in_major_expert_names(go)
    if hit != {p + "gate_up_proj", p + "down_proj"}:
        return f"case4: detector flagged {sorted(hit)}"
    if SC._in_major_expert_names(plain):
        return f"case4: false positive on a bias-free fused MoE ({sorted(SC._in_major_expert_names(plain))})"
    exp3d = {p + "gate_up_proj", p + "down_proj"}
    try:
        SC.pack_unit_tensors(dict(go), set(), exp3d, None, "int4", G)
    except ValueError as exc:
        if "IN-major" not in str(exc):
            return f"case4: refused, but not for the layout reason: {exc}"
    else:
        return "case4: pack_unit_tensors PACKED an in-major expert instead of refusing"
    try:                                            # the bias-free MoE must still pack fine
        SC.pack_unit_tensors(dict(plain), set(), exp3d, None, "int4", G)
    except Exception as exc:
        return f"case4: refused a legitimate fused MoE: {exc!r}"
    return None


def main() -> int:
    checks = [
        ("default [E,out,in] layout unchanged", case1_default_layout_unchanged),
        ("in-major gate_up [E,H,2I] == worker", lambda: case2_in_major_matches_worker("gate_up", 4, 96, 128)),
        ("in-major down [E,I,H] == worker", lambda: case2_in_major_matches_worker("down", 4, 64, 96)),
        ("in-major, in % G != 0 (pad path)", lambda: case2_in_major_matches_worker("padded", 3, 200, 96)),
        ("negative control: square expert", case3_negative_control),
        ("detector + pack_unit_tensors guard", case4_detector_and_guard),
    ]
    bad = 0
    for name, fn in checks:
        err = fn()
        print(f"{'FAIL' if err else 'ok  '}  {name}" + (f"\n        {err}" if err else ""))
        bad += bool(err)
    print(f"\n{len(checks) - bad}/{len(checks)} passed")
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
