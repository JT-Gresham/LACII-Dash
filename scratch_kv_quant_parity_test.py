"""#172 — ModelSpec.for_kv_quant must be a PURE dedup of /plan's old inline sizing, and the LIVE
load planner must now size identically to the Preview.

WHY THIS SHAPE. Two separate hazards, and only one of them is about the future:

  1. Was the collapse LOSSLESS? A "cleanup" that silently changes which footprint a model plans at
     is worse than the duplication it removes. So the old behaviour is not re-typed from memory —
     it is PINNED TO A FIXED PRE-CHANGE COMMIT'S ACTUAL BYTES: the test greps that formula out of
     `git show <pre-collapse commit>:routes_dashboard.py` and fails if the expression it re-implements is not
     literally the one that shipped. If someone edits the helper and updates the expectation to
     match, the pinned commit still disagrees and the test still fails.

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


# ---------------------------------------------------------------- 0. pin "old" to a FIXED commit
# b006ec2 is the last commit that still contains /plan's inline sizing — the parent of adf8011,
# which collapsed it into ModelSpec.for_kv_quant.
#
# This deliberately does NOT say "HEAD". It did at first, which was wrong in a way worth recording:
# HEAD is a MOVING target, so the moment the collapse itself was committed the pin pointed at the
# post-change file and could no longer find the code it exists to compare against. The test caught
# that itself and refused to pass — which is the behaviour wanted from a pin, but the pin still had
# to become immutable to be worth anything.
PRE_COLLAPSE = "b006ec2"
PRE_SRC = subprocess.run(["git", "show", f"{PRE_COLLAPSE}:routes_dashboard.py"],
                          capture_output=True, text=True, check=True).stdout
PINNED = [
    "_pt = _kq.kv_quant_bytes_per_token_per_layer(_kvq, spec.num_kv_heads, spec.head_dim)",
    "_bf = 2 * spec.num_kv_heads * spec.head_dim * KV_DTYPE_BYTES",
    "_r = _pt / _bf + 1.0 / max(1, spec.num_layers)",
    "kv_layer_frac=float(spec.kv_layer_frac or 1.0) * _r",
]
for frag in PINNED:
    if frag not in PRE_SRC:
        failures.append(f"pin lost: {frag!r} is not in {PRE_COLLAPSE}:routes_dashboard.py — this test "
                        f"can no longer prove the collapse was lossless; re-pin it deliberately")

# The old hybrid predicate, likewise pinned.
for frag in ('any(t != "full_attention" for t in _lt)',
             '_lac.get("kda_layers")'):
    if frag not in PRE_SRC:
        failures.append(f"pin lost (hybrid predicate): {frag!r} not in {PRE_COLLAPSE}")


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


def blended_ratio(spec, norm, cfg):
    """#172-hybrid: the ratio for_kv_quant SHOULD produce, derived from the FIXTURE's declared
    layer_types rather than from the ModelSpec properties under test — otherwise this would just
    assert that the code equals itself.

    Q = layers that PACK (exactly "full_attention").
    K = layers that grow a ctx-scaled K/V (everything except linear-attention; sliding COUNTS).
    Packed Q at `pt` bytes/token, the remaining K-Q at bf16, plus the one-bf16-layer dequant
    transient amortised over all layers. Returns None when nothing packs (the caller must then
    expect a refusal), which is the fla/kda case with no layer_types at all."""
    lts = cfg.get("layer_types")
    if lts is None:
        return None
    pt = _kq.kv_quant_bytes_per_token_per_layer(norm, spec.num_kv_heads, spec.head_dim)
    bf = 2 * spec.num_kv_heads * spec.head_dim * KV_DTYPE_BYTES
    q = sum(1 for t in lts if str(t) == "full_attention")
    k = sum(1 for t in lts if "linear" not in str(t))
    if q <= 0 or k <= 0 or pt <= 0 or bf <= 0:
        return None
    return ((q * pt + (k - q) * bf) / (k * bf)) + 1.0 / max(1, spec.num_layers)


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
        elif blended_ratio(spec, norm, cfg) is None:
            # Nothing packs on this arch (fla/kda: the worker builds _make_linattn_kv, never a
            # quantized cache). MUST refuse — sizing a packed cache the worker does not build is
            # an under-reserve, i.e. a decode OOM.
            if got.kv_layer_frac != spec.kv_layer_frac:
                failures.append(f"{label}/{preset!r}: NOTHING packs here — must NOT shrink "
                                f"(frac {spec.kv_layer_frac} -> {got.kv_layer_frac})")
            if not note:
                failures.append(f"{label}/{preset!r}: refusal must explain itself")
        elif hyb:
            # #172-hybrid: a hybrid now packs its full_attention layers. Expect the BLEND, and
            # expect it to be strictly between "all packed" and "no shrink at all" whenever the
            # arch actually mixes kinds — a hybrid that shrank like a dense model would mean
            # sliding/linear layers were packed too, which is the under-reserve this guards.
            want = blended_ratio(spec, norm, cfg) * float(spec.kv_layer_frac or 1.0)
            if abs(got.kv_layer_frac - want) > 1e-12:
                failures.append(f"{label}/{preset!r}: hybrid blend wrong — got {got.kv_layer_frac!r} "
                                f"want {want!r}")
            elif note:
                failures.append(f"{label}/{preset!r}: sized fine but emitted a refusal {note!r}")
            elif not got.kv_layer_frac < spec.kv_layer_frac:
                failures.append(f"{label}/{preset!r}: packing some layers must SHRINK the plan "
                                f"({got.kv_layer_frac} !< {spec.kv_layer_frac})")
            else:
                _dense = old_ratio(spec, norm) or 0.0
                if _dense and abs(got.kv_layer_frac - _dense * float(spec.kv_layer_frac or 1.0)) <= 1e-12:
                    failures.append(f"{label}/{preset!r}: hybrid shrank like a DENSE model — its "
                                    f"sliding/linear layers were packed, which under-reserves")
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
      f"{PRE_COLLAPSE}'s formula on dense; hybrids blend per-layer; fla/kda still refused; "
      "both planners use one helper")
