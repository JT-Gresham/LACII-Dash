"""#w8a16 kernel test: is the Triton int8 GEMM/GEMV CORRECT, and is it actually FASTER?

Two questions, kept apart on purpose. "Reads fewer bytes" does not imply "faster" — torch's own
`_weight_int8pack_mm` measured 2-5x SLOWER than bf16 `F.linear` on CUDA, which is exactly why
`QuantLinear.prepare_fused` gates on a benchmark and not just a self-check. This harness is what
that gate's decision should be sanity-checked against before the kernel goes near the fleet.

CORRECTNESS is checked against `F.linear(x, qweight.bf16 * scale)` — the naive path the kernel
replaces — at several M, including the shapes where the two DIFFERENT kernels meet (M=1 takes the
split-K atomic GEMV, M>1 takes the tl.dot GEMM) and at sizes that are deliberately NOT multiples of
the block sizes, so masking bugs cannot hide behind tidy dimensions.

The reference is bf16-exact, not approximate: int8 values (|q| <= 127) sit exactly inside bf16's 8
mantissa bits, so `q.to(bf16)` is lossless and the only difference between kernel and reference is
accumulation order. Errors materially above fp32-accumulation noise mean a real bug.

Run:  python3 scratch_w8a16_test.py            (add --quick to skip the big head shapes)
"""
import argparse
import os
import sys

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker_quant as wq                                       # noqa: E402

# Real shapes. The head is the reason this kernel exists (#8): 152064 x 3584 is Qwen2.5-7B's
# lm_head, 1.02 GB in bf16 and 22% of everything read per decoded token.
SHAPES = [(152064, 3584, "lm_head Qwen2.5-7B"),
          (3584, 3584, "q/o_proj 7B"),
          (18944, 3584, "gate/up_proj 7B"),
          (3584, 18944, "down_proj 7B"),
          (152064, 1536, "lm_head Qwen2.5-1.5B")]
# Awkward on purpose: neither dim is a multiple of any BN/BK in the autotune space.
ODD = [(1000, 999, "unaligned N,K"), (17, 130, "tiny + unaligned")]


def quantize(W):
    """worker_quant._quantize_linear's math, without needing an nn.Linear."""
    scale = (W.abs().amax(dim=1, keepdim=True) / 127.0).clamp(min=1e-8)
    q = (W / scale).round().clamp(-127, 127).to(torch.int8).contiguous()
    return q, scale.to(torch.bfloat16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    a = ap.parse_args()
    dev = "cuda"
    print(f"{torch.cuda.get_device_name(0)}  torch={torch.__version__}  "
          f"triton={getattr(wq.triton, '__version__', None)}")
    op = wq._w8a16_triton_op()
    if op is None:
        sys.exit("w8a16 op failed to build")

    shapes = (ODD + SHAPES[1:3]) if a.quick else (ODD + SHAPES)

    print("\n== CORRECTNESS (vs F.linear(x, q.bf16 * scale) — the path being replaced) ==")
    print(f"{'shape':28s} {'M':>6s} {'rel err':>10s} {'max abs':>10s}  verdict")
    bad = 0
    for N, K, name in shapes:
        W = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02)
        q, s = quantize(W)
        ref_w = q.to(torch.bfloat16) * s
        for M in (1, 2, 4, 16, 129, 512):
            x = torch.randn(M, K, device=dev, dtype=torch.bfloat16)
            y = op(x, q, s).float()
            r = F.linear(x, ref_w).float()
            rel = float((y - r).abs().mean() / (r.abs().mean() + 1e-9))
            mx = float((y - r).abs().max())
            ok = rel < 0.02
            bad += (not ok)
            if M in (1, 512) or not ok:
                print(f"{name:28s} {M:6d} {rel:10.5f} {mx:10.4f}  {'ok' if ok else 'FAIL <<<'}")
        del W, q, s, ref_w
        torch.cuda.empty_cache()

    # Re-running M=1 twice in a row is not redundant: the GEMV atomic-adds into its output, so a
    # missing reset_to_zero shows up as the SECOND call being wrong while the first looks fine.
    print("\n== GEMV repeat-call stability (catches a missing reset_to_zero) ==")
    N, K = 4096, 3584
    W = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02)
    q, s = quantize(W)
    ref = F.linear(torch.ones(1, K, device=dev, dtype=torch.bfloat16),
                   q.to(torch.bfloat16) * s).float()
    x1 = torch.ones(1, K, device=dev, dtype=torch.bfloat16)
    for i in range(4):
        rel = float((op(x1, q, s).float() - ref).abs().mean() / (ref.abs().mean() + 1e-9))
        print(f"  call {i + 1}: rel={rel:.6f}  {'ok' if rel < 0.02 else 'FAIL <<<'}")
        bad += (rel >= 0.02)
    del W, q, s
    torch.cuda.empty_cache()

    print("\n== SPEED at decode (M=1) — the case the kernel exists for ==")
    print(f"{'shape':28s} {'int8 bytes':>11s} {'w8a16':>9s} {'dequant+F.linear':>17s} "
          f"{'bf16 F.linear':>14s}  verdict")
    for N, K, name in shapes[2:] if not a.quick else shapes[2:]:
        W = (torch.randn(N, K, device=dev, dtype=torch.bfloat16) * 0.02)
        q, s = quantize(W)
        x = torch.randn(1, K, device=dev, dtype=torch.bfloat16)
        t_f = wq._time_cuda(lambda: op(x, q, s), dev)
        t_n = wq._time_cuda(lambda: F.linear(x, q.to(torch.bfloat16) * s), dev)
        # The REAL alternative for an lm_head is not int8-dequant, it is just keeping bf16.
        t_b = wq._time_cuda(lambda: F.linear(x, W), dev)
        v = ("beats bf16 too" if t_f < t_b else
             "beats dequant, loses to bf16" if t_f < t_n else "LOSES")
        print(f"{name:28s} {q.numel() / 2**20:10.0f}M {t_f:8.3f}ms {t_n:16.3f}ms "
              f"{t_b:13.3f}ms  {v}")
        del W, q, s
        torch.cuda.empty_cache()

    print(f"\n{'ALL CHECKS PASSED' if bad == 0 else f'{bad} FAILURE(S)'}")
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
