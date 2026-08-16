"""#bare-linear-router: does the MoE ROUTER stay bf16 at int4, int8 AND int2?

Quantizing a router does not crash — it corrupts top-k EXPERT SELECTION, and the model keeps
serving plausible-but-wrong tokens. poolside/Laguna-XS-2.1 loaded clean at int4, answered "The
capital of France is Paris." once, then looped forever (4dc57e6). docs/ACCELERATION.md states the
bf16 router as an invariant across ALL tiers, and shard_compile._quant_scope enforces it
tier-agnostically for the shard cache.

WHY THIS FILE EXISTS: the invariant was implemented as a per-tier COPY of the exclusion, and the
int8 copy did not have it — `_quantize_int8_linears_` quantized routers while `_quantize_int4_`
and `_quantize_int2_` skipped them. It was latent only because routes_lifecycle downgraded
int8-on-MoE to int4 (#moe-int8, "a tier with no 3D routed-expert packer"); landing
`_quantize_experts8_` removed that downgrade and made the hole reachable. The fix routes all
three tiers through ONE walk (`_quantize_linears_`), and this harness is what stops a fourth tier
— or a refactor of the walk — from quietly reopening it.

WHAT IS CHECKED, per tier (int4 / int8 / int2):
  1. The four router SHAPES that occur in the wild all stay bf16 nn.Linear with byte-identical
     weights:
       * bare nn.Linear named `gate`   (Laguna `model.layers.N.mlp.gate` — no *Router ancestor)
       * bare nn.Linear named `router`
       * bare nn.Linear named `wg`
       * ANY Linear under a module whose CLASS name ends Router/Gate (gemma4 Gemma4TextRouter
         exposes `proj` as a plain nn.Linear), including one nested two levels down
  2. No OVER-exclusion: `gate_proj` / `gate_up_proj` / `down_proj` / attention projections are
     still quantized. A "fix" that skips every name containing "gate" would leave most of the
     MLP in bf16 and pass check (1) — that regression is the reason (2) is asserted separately.
  3. TIER PARITY: the set of quantized module paths is IDENTICAL across int4, int8 and int2.
     This is the assertion that would have caught the original defect; (1) states the rule, (3)
     states that no tier gets its own version of it.
  4. The newly-live path end to end: `_quantize_int8_` on a fused-3D MoE block packs the experts
     into Packed8Tensor3D AND leaves the block's bare `gate` router bf16 — the exact combination
     that was impossible to reach before the int8 expert packer shipped.
  5. The int8 3D expert walker (`_quantize_experts8_`) alone never touches a router, i.e. the
     hole was in the Linear walk and only there.

NOT CHECKED HERE: that a bf16 router actually routes better than a quantized one. That is a
generation-quality claim needing a real MoE checkpoint on the fleet; the evidence for it is the
Laguna repetition loop in 4dc57e6, not this file.

Run:  python3 scratch_router_quant_exclusion_test.py        (CPU torch is enough — no GPU, no triton)

STATUS 2026-08-16: UNRUN on MOBILE — this box has no torch (`ModuleNotFoundError: No module named
'torch'`; MOBILE is the Debian reinstall with no CUDA/torch install). Run it on any fleet box with
CPU torch: om3nbox (100.94.43.14), beast (192.168.15.38), amdcomp (192.168.15.39) or the dell CPU
worker (192.168.15.30) — `python3 scratch_router_quant_exclusion_test.py` from the repo root.
"""
import os
import sys

import torch
from torch import nn

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import worker_quant as wq                                        # noqa: E402

FAILS = []
DT = torch.bfloat16


def check(name, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {name}{('  — ' + detail) if detail else ''}")
    if not ok:
        FAILS.append(name)
    return ok


# --- module fixtures -------------------------------------------------------------------------
# Deliberately NOT a real checkpoint: the exclusion keys on attribute NAMES and CLASS names, so a
# hand-built tree exercises every shape the walk has to recognise, on a laptop, in milliseconds.

class _TextRouter(nn.Module):
    """gemma4 Gemma4TextRouter shape: a *Router*-classed module holding a plain nn.Linear. The
    inner name is `proj` — nothing the leaf-name set knows about — so ONLY the class-name rule
    can save it."""

    def __init__(self, hidden, E):
        super().__init__()
        self.proj = nn.Linear(hidden, E, bias=False, dtype=DT)


class _Inner(nn.Module):
    def __init__(self, hidden, E):
        super().__init__()
        self.proj = nn.Linear(hidden, E, bias=False, dtype=DT)


class _TopKGate(nn.Module):
    """*Gate*-classed, with the Linear TWO levels down: the walk must skip the whole subtree, not
    just its immediate children."""

    def __init__(self, hidden, E):
        super().__init__()
        self.inner = _Inner(hidden, E)


class _Attn(nn.Module):
    def __init__(self, hidden):
        super().__init__()
        self.q_proj = nn.Linear(hidden, hidden, bias=False, dtype=DT)
        self.k_proj = nn.Linear(hidden, hidden, bias=False, dtype=DT)
        self.v_proj = nn.Linear(hidden, hidden, bias=False, dtype=DT)
        self.o_proj = nn.Linear(hidden, hidden, bias=False, dtype=DT)


class _MoEMlp(nn.Module):
    def __init__(self, hidden, inter, E):
        super().__init__()
        self.gate = nn.Linear(hidden, E, bias=False, dtype=DT)          # ROUTER (#bare-linear-router)
        self.gate_proj = nn.Linear(hidden, inter, bias=False, dtype=DT)  # expert proj -> QUANTIZE
        self.up_proj = nn.Linear(hidden, inter, bias=False, dtype=DT)    # -> QUANTIZE
        self.down_proj = nn.Linear(inter, hidden, bias=False, dtype=DT)  # -> QUANTIZE


class _Layer(nn.Module):
    """One decoder layer carrying every router shape at once, plus enough ordinary Linears that an
    over-broad exclusion shows up as a missing quantization rather than as nothing."""

    def __init__(self, hidden=32, inter=64, E=8):
        super().__init__()
        self.self_attn = _Attn(hidden)
        self.mlp = _MoEMlp(hidden, inter, E)
        self.router = nn.Linear(hidden, E, bias=False, dtype=DT)        # ROUTER (leaf name)
        self.wg = nn.Linear(hidden, E, bias=False, dtype=DT)            # ROUTER (leaf name, Deepseek-ish)
        self.text_router = _TextRouter(hidden, E)                        # ROUTER (class name)
        self.topk_gate = _TopKGate(hidden, E)                            # ROUTER (class name, nested)
        # A shared expert with its own gate_up_proj: the fused-name form, still a real projection.
        self.shared_gate_up_proj = nn.Linear(hidden, 2 * inter, bias=False, dtype=DT)


ROUTER_PATHS = ("mlp.gate", "router", "wg", "text_router.proj", "topk_gate.inner.proj")
MUST_QUANTIZE = ("self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj",
                 "mlp.gate_proj", "mlp.up_proj", "mlp.down_proj", "shared_gate_up_proj")


def _get(mod, path):
    for part in path.split("."):
        mod = getattr(mod, part)
    return mod


def _linear_paths(root):
    """Every nn.Linear still present, by dotted path — i.e. everything the walk left alone."""
    return {n for n, m in root.named_modules() if isinstance(m, nn.Linear)}


# --- 1 + 2: per-tier exclusion + no over-exclusion --------------------------------------------
def test_tier(tier, quantize):
    print(f"\n== {tier}: router exclusion ==")
    lyr = _Layer()
    before = {p: _get(lyr, p).weight.data.clone() for p in ROUTER_PATHS}
    quantize(lyr)

    bad = []
    for p in ROUTER_PATHS:
        m = _get(lyr, p)
        if not isinstance(m, nn.Linear):
            bad.append(f"{p}: replaced by {type(m).__name__}")
            continue
        if m.weight.dtype != DT:
            bad.append(f"{p}: dtype {m.weight.dtype}")
        elif not torch.equal(m.weight.data, before[p]):
            bad.append(f"{p}: weights mutated")
    check(f"{tier}: all {len(ROUTER_PATHS)} router shapes still bf16 nn.Linear, bytes unchanged",
          not bad, "; ".join(bad))

    missed = [p for p in MUST_QUANTIZE if isinstance(_get(lyr, p), nn.Linear)]
    check(f"{tier}: no over-exclusion — every non-router Linear WAS quantized",
          not missed, "still bf16: " + ", ".join(missed))
    return _linear_paths(lyr)


# --- 3: tier parity --------------------------------------------------------------------------
def test_parity(skipped):
    print("\n== tier parity ==")
    tiers = sorted(skipped)
    base = skipped[tiers[0]]
    diffs = []
    for t in tiers[1:]:
        if skipped[t] != base:
            only_a = sorted(base - skipped[t])
            only_b = sorted(skipped[t] - base)
            diffs.append(f"{tiers[0]} vs {t}: +{only_a} -{only_b}")
    check("int4 / int8 / int2 skip the IDENTICAL set of Linears", not diffs, "; ".join(diffs))
    check("the skipped set is exactly the routers",
          base == set(ROUTER_PATHS), f"got {sorted(base)}")


# --- 4 + 5: the newly-live int8 fused-MoE path ------------------------------------------------
class _Cfg:
    def __init__(self):
        self._experts_implementation = "grouped_mm"


class _Experts(nn.Module):
    """Fused-3D MoE experts (Qwen3-MoE / MiniMax layout): gate_up_proj/down_proj are 3D
    nn.Parameters indexed per routed expert."""

    def __init__(self, E, hidden, inter):
        super().__init__()
        self.config = _Cfg()
        self.gate_up_proj = nn.Parameter(torch.randn(E, 2 * inter, hidden, dtype=DT),
                                         requires_grad=False)
        self.down_proj = nn.Parameter(torch.randn(E, hidden, inter, dtype=DT),
                                      requires_grad=False)


class _FusedBlock(nn.Module):
    def __init__(self, E=4, hidden=32, inter=64):
        super().__init__()
        self.gate = nn.Linear(hidden, E, bias=False, dtype=DT)   # the router, bare nn.Linear
        self.experts = _Experts(E, hidden, inter)


def test_int8_fused_moe():
    print("\n== int8 on a fused-3D MoE (the path routes_lifecycle just un-downgraded) ==")
    PT8 = wq._packed8_3d_cls()
    blk = _FusedBlock()
    gate_w = blk.gate.weight.data.clone()
    wq._quantize_int8_(blk)
    check("fused 3D experts packed to Packed8Tensor3D",
          isinstance(blk.experts.gate_up_proj, PT8) and isinstance(blk.experts.down_proj, PT8))
    check("router stayed a bf16 nn.Linear, bytes unchanged",
          isinstance(blk.gate, nn.Linear) and blk.gate.weight.dtype == DT
          and torch.equal(blk.gate.weight.data, gate_w),
          f"gate is {type(blk.gate).__name__}")

    # 5: the 3D walker in isolation. It targets 3D nn.Parameters named exactly gate_up_proj /
    # down_proj, so a 2D router weight cannot match — assert that rather than assume it.
    blk2 = _FusedBlock()
    gate_w2 = blk2.gate.weight.data.clone()
    wq._quantize_experts8_(blk2)
    check("_quantize_experts8_ alone leaves the router untouched",
          isinstance(blk2.gate, nn.Linear) and torch.equal(blk2.gate.weight.data, gate_w2))


# ---------------------------------------------------------------------------------------------
def main():
    print(f"torch {torch.__version__}  worker_quant from {wq.__file__}")
    check("_ROUTER_LEAF_NAMES is the expected set",
          wq._ROUTER_LEAF_NAMES == frozenset({"gate", "router", "wg"}),
          str(sorted(wq._ROUTER_LEAF_NAMES)))
    skipped = {
        "int4": test_tier("int4", wq._quantize_int4_),
        "int8": test_tier("int8", wq._quantize_int8_linears_),
        "int2": test_tier("int2", wq._quantize_int2_),
    }
    test_parity(skipped)
    test_int8_fused_moe()
    print(f"\n{'ALL PASS' if not FAILS else str(len(FAILS)) + ' FAILED: ' + ', '.join(FAILS)}")
    return 1 if FAILS else 0


if __name__ == "__main__":
    sys.exit(main())
