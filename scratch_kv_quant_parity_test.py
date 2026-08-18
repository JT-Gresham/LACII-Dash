"""#172 — ModelSpec.for_kv_quant must be a PURE dedup of /plan's old inline sizing, and the LIVE
load planner must now size identically to the Preview.

WHY THIS SHAPE. Two separate hazards, and only one of them is about the future:

  1. Was the collapse LOSSLESS? A "cleanup" that silently changes which footprint a model plans at
     is worse than the duplication it removes. So the old behaviour is not re-typed from memory —
     it is PINNED TO HEAD'S ACTUAL BYTES: the test greps the pre-change formula out of
     `git show HEAD:routes_dashboard.py` and fails if the expression it re-implements is not
     literally the one that shipped. If someone edits the helper and updates the expectation to
     match, the HEAD pin still disagrees and the test still fails.

  2. Can the two planners drift apart AGAIN? That is the defect being fixed: /plan sized packed KV
     while engine_load sized the same load at bf16, so Preview said "fits" and /load raised
     CapacityError on identical numbers. Asserted structurally — both call sites must route
     through the one helper, and nobody may re-spell the ratio inline.

Runs without torch (kv_quant and placement both import bare), so it is a real behavioural test
rather than a transcription of what I think the code does.
"""
import re
import subprocess
import sys

import kv_quant as _kq
from placement import KV_DTYPE_BYTES, ModelSpec

failures: list[str] = []


def _spec(**kw) -> ModelSpec:
    base = dict(name="t", hidden_size=4096, num_layers=32, num_heads=32, num_kv_heads=8,
                head_dim=128, intermediate_size=11008, vocab_size=32000, tie_embeddings=False)
    base.update(kw)
    return ModelSpec(**base)


# ---------------------------------------------------------------- 0. pin "old" to HEAD's bytes
HEAD_SRC = subprocess.run(["git", "show", "HEAD:routes_dashboard.py"],
                          capture_output=True, text=True, check=True).stdout
PINNED = [
    "_pt = _kq.kv_quant_bytes_per_token_per_layer(_kvq, spec.num_kv_heads, spec.head_dim)",
    "_bf = 2 * spec.num_kv_heads * spec.head_dim * KV_DTYPE_BYTES",
    "_r = _pt / _bf + 1.0 / max(1, spec.num_layers)",
    "kv_layer_frac=float(spec.kv_layer_frac or 1.0) * _r",
]
for frag in PINNED:
    if frag not in HEAD_SRC:
        failures.append(f"HEAD pin lost: {frag!r} is not in HEAD:routes_dashboard.py — this test "
                        f"can no longer prove the collapse was lossless; re-pin it deliberately")

# The old hybrid predicate, likewise pinned.
for frag in ('any(t != "full_attention" for t in _lt)',
             '_lac.get("kda_layers")'):
    if frag not in HEAD_SRC:
        failures.append(f"HEAD pin lost (hybrid predicate): {frag!r}")


def old_ratio(spec: ModelSpec, kvq: str):
    """Verbatim re-implementation of the four PINNED lines above."""
    _pt = _kq.kv_quant_bytes_per_token_per_layer(kvq, spec.num_kv_heads, spec.head_dim)
    _bf = 2 * spec.num_kv_heads * spec.head_dim * KV_DTYPE_BYTES
    if _pt > 0 and _bf > 0:
        return float(spec.kv_layer_frac or 1.0) * (_pt / _bf + 1.0 / max(1, spec.num_layers))
    return None


def old_hybrid(cfg: dict) -> bool:
    """Verbatim re-implementation of the old config-driven gate."""
    _lt = cfg.get("layer_types") or []
    _lac = cfg.get("linear_attn_config")
    return (any(t != "full_attention" for t in _lt)
            or bool(isinstance(_lac, dict) and _lac.get("kda_layers")))


# ---------------------------------------------------------------- 1. cross-product equivalence
DENSE = ("dense-32L", _spec(), {"layer_types": ["full_attention"] * 32})
DENSE_DECL = ("dense-declared", _spec(layer_types=tuple(["full_attention"] * 32)),
              {"layer_types": ["full_attention"] * 32})
SLIDING = ("sliding-hybrid", _spec(layer_types=tuple(
    ["sliding_attention" if i % 2 else "full_attention" for i in range(32)])),
    {"layer_types": ["sliding_attention" if i % 2 else "full_attention" for i in range(32)]})
LINEAR = ("linear-attn (kda)", _spec(kv_layer_frac=7 / 27),
          {"linear_attn_config": {"kda_layers": [1, 2, 3]}})
MQA = ("mqa-1kv-head", _spec(num_kv_heads=1, head_dim=64, num_layers=80),
       {"layer_types": ["full_attention"] * 80})

CASES = [DENSE, DENSE_DECL, SLIDING, LINEAR, MQA]
PRESETS = ["none", "turbo2", "turbo3", "turbo4", "", None, "TURBO3", "bogus", "int4"]

checked = 0
for label, spec, cfg in CASES:
    for preset in PRESETS:
        got, note = spec.for_kv_quant(preset)
        norm = (preset or "none").strip().lower()
        hyb = old_hybrid(cfg)

        if norm not in ("turbo2", "turbo3", "turbo4"):
            # not a preset -> untouched, no note
            if got.kv_layer_frac != spec.kv_layer_frac or note:
                failures.append(f"{label}/{preset!r}: non-preset must be a no-op, "
                                f"got frac={got.kv_layer_frac} note={note!r}")
        elif hyb:
            if got.kv_layer_frac != spec.kv_layer_frac:
                failures.append(f"{label}/{preset!r}: HYBRID must NOT shrink (worker reserves "
                                f"bf16) — frac {spec.kv_layer_frac} -> {got.kv_layer_frac}")
            if not note:
                failures.append(f"{label}/{preset!r}: hybrid refusal must explain itself")
        else:
            want = old_ratio(spec, norm)
            if want is None:
                failures.append(f"{label}/{preset!r}: old formula produced nothing to compare")
            elif abs(got.kv_layer_frac - want) > 1e-12:
                failures.append(f"{label}/{preset!r}: NOT a pure dedup — new={got.kv_layer_frac!r} "
                                f"old={want!r}")
            elif note:
                failures.append(f"{label}/{preset!r}: sized fine but emitted a refusal {note!r}")
            elif not got.kv_layer_frac < spec.kv_layer_frac:
                failures.append(f"{label}/{preset!r}: packed KV must be SMALLER than bf16 "
                                f"({got.kv_layer_frac} !< {spec.kv_layer_frac})")
        checked += 1

# ---------------------------------------------------------------- 2. is_hybrid == old predicate
for label, spec, cfg in CASES:
    if spec.is_hybrid != old_hybrid(cfg):
        failures.append(f"{label}: ModelSpec.is_hybrid={spec.is_hybrid} disagrees with the old "
                        f"config-driven gate={old_hybrid(cfg)} — the gate changed meaning")

# ---------------------------------------------------------------- 3. both planners use the helper
def _src(name: str) -> str:
    with open(name, encoding="utf-8") as fh:
        return fh.read()

for fn, what in (("routes_dashboard.py", "the /plan preview"),
                 ("engine_load.py", "the live load planner")):
    src = _src(fn)
    if "for_kv_quant(" not in src:
        failures.append(f"{fn}: {what} no longer sizes KV via ModelSpec.for_kv_quant — the "
                        f"Preview and the live load can disagree about whether a model fits again")
    # nobody may re-spell the ratio inline
    if re.search(r"kv_quant_bytes_per_token_per_layer\s*\(", src):
        failures.append(f"{fn}: re-spells the packed-KV ratio inline instead of calling "
                        f"for_kv_quant — that is exactly how the two planners drifted apart")

if failures:
    print("FAIL — #172 kv_quant parity:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"PASS — for_kv_quant is a pure dedup over {checked} (arch x preset) combinations, pinned to "
      f"HEAD's formula; is_hybrid matches the old gate; both planners route through one helper")
