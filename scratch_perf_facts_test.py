"""#perf-facts — regression gate for arch facts that reach perf_profile.resolve() as DEAD inputs.

The bug this pins (found 2026-08-17): `perf_profile.resolve()` takes three arch facts —
`is_hybrid`, `is_multimodal`, `full_attention_layers` — and at BOTH of its call sites
(engine_load `_perf_auto_knobs`, routes_dashboard `/optimize_knobs`) all three were structurally
always their default:

  * `is_hybrid` / `is_multimodal` were read as `getattr(spec, "layer_types"/"is_multimodal", ...)`
    against a `ModelSpec` dataclass that declared NEITHER field. Nothing anywhere attached one.
  * `full_attention_layers` was simply never passed by any caller.

Consequence: the resolver's "multimodal/hybrid arch -> clamp kv_slots to 1" branch could never
fire, and `kv_slots` is an APPLIED knob — so a VLM could be auto-assigned C=2/3, reserving 2-3x
full-ctx KV to buy prefix reuse that `#prefix-kv` gates off for exactly those arches.

An inert input is invisible: the call site READS correct, the resolver READS correct, and a test
that exercises `resolve()` directly passes happily because it passes the flag itself. So the
tests here deliberately do NOT test `resolve()`. They test the WIRING:

  1. bug-class gate  — no `getattr(spec, X, default)` anywhere may name a non-field of ModelSpec.
  2. coverage gate   — every arch-fact parameter `resolve()` declares must be passed by EVERY
                       call site. This is what catches `full_attention_layers` (dead by OMISSION,
                       which no getattr scan can see).
  3. behaviour       — is_hybrid / full_attention_layers over real arch shapes.
  4. PARITY          — /plan's kv_quant hybrid gate and ModelSpec.is_hybrid must agree. They are
                       two implementations of "is this arch hybrid" and this asserts they answer
                       the same, which is the drift test with teeth.

Runs anywhere: no torch, no network, no fleet.
"""
from __future__ import annotations

import ast
import json
import os
import pathlib
import re
import sys
import tempfile

REPO = pathlib.Path(__file__).resolve().parent
sys.path.insert(0, str(REPO))

import model_store          # noqa: E402
from placement import ModelSpec   # noqa: E402

FAILURES: list[str] = []
CHECKS = 0


def check(label: str, ok: bool, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if ok:
        print(f"  PASS  {label}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"  FAIL  {label} — {detail}")


def _modelspec_names() -> set[str]:
    """Field + property + method names really declared on ModelSpec, parsed from the source."""
    tree = ast.parse((REPO / "placement.py").read_text(encoding="utf-8"))
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and node.name == "ModelSpec":
            out = set()
            for b in node.body:
                if isinstance(b, ast.AnnAssign) and isinstance(b.target, ast.Name):
                    out.add(b.target.id)
                elif isinstance(b, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    out.add(b.name)
            return out
    raise AssertionError("ModelSpec class not found in placement.py")


def _source_files() -> list[pathlib.Path]:
    return [p for p in sorted(REPO.glob("*.py"))
            if not p.name.startswith(("scratch_", "test_"))]


# --- 1. bug-class gate ----------------------------------------------------------------------

def test_no_dead_spec_reads() -> None:
    print("\n[1] no getattr(spec, X, default) may name a non-field of ModelSpec")
    names = _modelspec_names()
    pat = re.compile(r'getattr\(\s*(?:[\w.]*\bspec|lm\.spec|qspec)\s*,\s*["\'](\w+)["\']\s*,')
    dead = []
    for p in _source_files():
        for i, line in enumerate(p.read_text(encoding="utf-8").splitlines(), 1):
            for m in pat.finditer(line):
                if m.group(1) not in names:
                    dead.append(f"{p.name}:{i} spec.{m.group(1)}")
    check("zero always-default spec reads", not dead, "; ".join(dead))


# --- 2. coverage gate -----------------------------------------------------------------------

# Parameters of resolve() that describe the MODEL's architecture. A call site that omits one is
# silently planning against a default that may be wrong for the model in hand. (Placement/request
# facts are excluded: those legitimately vary per caller.)
ARCH_FACT_PARAMS = {"is_hybrid", "is_multimodal", "full_attention_layers"}

# file -> the attribute call we expect (both sites alias perf_profile as _pp)
RESOLVE_CALL_SITES = ("engine_load.py", "routes_dashboard.py")


def _resolve_kwargs(path: pathlib.Path) -> list[set[str]]:
    """Keyword names of every `<something>.resolve(...)` call in a file."""
    tree = ast.parse(path.read_text(encoding="utf-8"))
    out = []
    for node in ast.walk(tree):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)
                and node.func.attr == "resolve"):
            out.append({k.arg for k in node.keywords if k.arg})
    return out


def test_every_call_site_passes_every_arch_fact() -> None:
    print("\n[2] every resolve() call site passes every arch fact")
    declared = set()
    tree = ast.parse((REPO / "perf_profile.py").read_text(encoding="utf-8"))
    for node in tree.body:
        if isinstance(node, ast.FunctionDef) and node.name == "resolve":
            declared = {a.arg for a in node.args.kwonlyargs} | {a.arg for a in node.args.args}
    missing_decl = ARCH_FACT_PARAMS - declared
    check("resolve() still declares the arch facts", not missing_decl, f"gone: {missing_decl}")

    for fname in RESOLVE_CALL_SITES:
        calls = _resolve_kwargs(REPO / fname)
        check(f"{fname} has a resolve() call", bool(calls), "none found")
        for kw in calls:
            missing = ARCH_FACT_PARAMS - kw
            check(f"{fname} passes all arch facts", not missing, f"omits {sorted(missing)}")


# --- 3. behaviour ---------------------------------------------------------------------------

def _spec(**kw) -> ModelSpec:
    base = dict(hidden_size=4096, num_layers=24, num_heads=32, num_kv_heads=8, head_dim=128,
                intermediate_size=11008, vocab_size=32000, tie_embeddings=False)
    base.update(kw)
    return ModelSpec(name="t", **base)


def test_is_hybrid_and_full_attention_layers() -> None:
    print("\n[3] is_hybrid / full_attention_layers behaviour")

    dense = _spec()
    check("dense: not hybrid", dense.is_hybrid is False)
    check("dense: all layers full", dense.full_attention_layers == 24,
          f"got {dense.full_attention_layers}")

    # gpt-oss / Gemma-4 shape: alternating sliding + full.
    lt = tuple(("sliding_attention" if i % 2 == 0 else "full_attention") for i in range(24))
    slide = _spec(layer_types=lt)
    check("sliding: IS hybrid (no prefix reuse)", slide.is_hybrid is True)
    check("sliding layers still COUNT toward KV (bounded, not zero)",
          slide.full_attention_layers == 24, f"got {slide.full_attention_layers}")

    # qwen3-next shape: linear-attention layers hold fixed-size state.
    lt2 = tuple(("linear_attention" if i % 4 else "full_attention") for i in range(24))
    lin = _spec(layer_types=lt2)
    check("linear: IS hybrid", lin.is_hybrid is True)
    check("linear layers excluded from KV", lin.full_attention_layers == 6,
          f"got {lin.full_attention_layers} (expected 6 of 24)")

    # Kimi shape: no layer_types at all, arrives as kv_layer_frac.
    kimi = _spec(num_layers=27, kv_layer_frac=7 / 27)
    check("kda/frac: IS hybrid without layer_types", kimi.is_hybrid is True)
    check("kda/frac: full layers from frac", kimi.full_attention_layers == 7,
          f"got {kimi.full_attention_layers}")

    # An unknown layer kind must COUNT (conservative: over-reserve, never OOM at decode).
    weird = _spec(layer_types=tuple(["brand_new_attention"] * 24))
    check("unknown kind is hybrid", weird.is_hybrid is True)
    check("unknown kind still charged full KV", weird.full_attention_layers == 24,
          f"got {weird.full_attention_layers}")

    # A layer_types that does not describe THIS model's depth is not trusted.
    short = _spec(num_layers=24, layer_types=("full_attention", "linear_attention"))
    check("mismatched-length layer_types falls back to frac", short.full_attention_layers == 24,
          f"got {short.full_attention_layers}")


# --- 4. parity ------------------------------------------------------------------------------

# Transcribed from routes_dashboard.py's /plan kv_quant gate. It reads config.json directly
# rather than the spec, because it must stay CONSERVATIVE when the config is unreadable (a case
# where the spec falls back to the built-in dense table and would look non-hybrid). That is why
# it is not simply `spec.is_hybrid` — but on any config both CAN read, they must agree.
def _plan_gate_hybrid(cfg: dict) -> bool:
    cfg = cfg.get("text_config") or cfg
    lt = cfg.get("layer_types") or []
    lac = cfg.get("linear_attn_config")
    return (any(t != "full_attention" for t in lt)
            or bool(isinstance(lac, dict) and lac.get("kda_layers")))


_DENSE = dict(hidden_size=4096, num_hidden_layers=24, num_attention_heads=32,
              num_key_value_heads=8, head_dim=128, intermediate_size=11008,
              vocab_size=32000, model_type="llama", architectures=["LlamaForCausalLM"])


def _cfg(**kw) -> dict:
    c = dict(_DENSE)
    c.update(kw)
    return c


PARITY_CONFIGS = {
    "dense llama": _cfg(),
    "gemma4 sliding": _cfg(layer_types=["sliding_attention", "full_attention"] * 12),
    "qwen3-next linear": _cfg(layer_types=["linear_attention"] * 18 + ["full_attention"] * 6),
    "all-full layer_types": _cfg(layer_types=["full_attention"] * 24),
    "kimi kda": _cfg(num_hidden_layers=27, linear_attn_config={"kda_layers": list(range(1, 21))},
                     qk_nope_head_dim=128, qk_rope_head_dim=64, v_head_dim=128),
    "vlm nested text_config": _cfg(vision_config={"hidden_size": 1024},
                                   text_config=dict(_DENSE, layer_types=["sliding_attention",
                                                                         "full_attention"] * 12)),
}


def test_plan_gate_parity() -> None:
    print("\n[4] PARITY: /plan's kv_quant hybrid gate == ModelSpec.is_hybrid")
    with tempfile.TemporaryDirectory() as td:
        for label, cfg in PARITY_CONFIGS.items():
            d = os.path.join(td, re.sub(r"\W+", "_", label))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            spec = model_store._spec_from_config(d, label)
            if spec is None:
                check(f"{label}: spec built", False, "_spec_from_config returned None")
                continue
            want = _plan_gate_hybrid(cfg)
            check(f"{label}: gate({want}) == spec.is_hybrid({spec.is_hybrid})",
                  spec.is_hybrid == want,
                  f"/plan says {want}, ModelSpec says {spec.is_hybrid}")


def test_multimodal_detection() -> None:
    print("\n[5] is_multimodal detection")
    with tempfile.TemporaryDirectory() as td:
        cases = {
            "plain llama": (_cfg(), False),
            "nested text_config only": (_cfg(text_config=dict(_DENSE)), False),
            "vision tower": (_cfg(vision_config={"hidden_size": 1024}), True),
            "audio tower": (_cfg(audio_config={"hidden_size": 1024}), True),
            "omni thinker": (_cfg(thinker_config={"text_config": dict(_DENSE)}), True),
        }
        for label, (cfg, want) in cases.items():
            d = os.path.join(td, re.sub(r"\W+", "_", label))
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            spec = model_store._spec_from_config(d, label)
            if spec is None:
                check(f"{label}: spec built", False, "returned None")
                continue
            check(f"{label}: is_multimodal == {want}", spec.is_multimodal == want,
                  f"got {spec.is_multimodal}")


def test_kv_layer_frac_from_layer_types() -> None:
    """kv_layer_frac was derived ONLY from linear_attn_config.kda_layers — the layer_types branch
    of the same rule was missing, so every transformers-style hybrid planned KV over EVERY layer
    while the worker's _kv_layer_mask reserved only the layers that really grow one."""
    print("\n[6] kv_layer_frac derived from layer_types too (not just kda_layers)")
    with tempfile.TemporaryDirectory() as td:
        cases = {
            # (config, expected frac, expected full_attention_layers)
            "qwen3.6-35b-a3b shape (30 linear + 10 full of 40)": (
                _cfg(num_hidden_layers=40,
                     layer_types=["linear_attention"] * 30 + ["full_attention"] * 10),
                10 / 40, 10),
            "gemma-4 sliding (bounded KV -> layers still COUNT)": (
                _cfg(num_hidden_layers=24,
                     layer_types=["sliding_attention"] * 20 + ["full_attention"] * 4),
                1.0, 24),
            "dense (no layer_types)": (_cfg(num_hidden_layers=24), 1.0, 24),
            "unknown kind counts (over-reserve, never OOM)": (
                _cfg(num_hidden_layers=24, layer_types=["brand_new_attn"] * 24), 1.0, 24),
            "mamba is NOT kv-less to the worker -> must count": (
                _cfg(num_hidden_layers=24, layer_types=["mamba"] * 20 + ["full_attention"] * 4),
                1.0, 24),
            "linear_* variant IS kv-less (substring)": (
                _cfg(num_hidden_layers=24,
                     layer_types=["linear_deltanet"] * 18 + ["full_attention"] * 6),
                6 / 24, 6),
        }
        for label, (cfg, want_frac, want_full) in cases.items():
            d = os.path.join(td, re.sub(r"\W+", "_", label)[:40])
            os.makedirs(d, exist_ok=True)
            with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
                json.dump(cfg, fh)
            spec = model_store._spec_from_config(d, label)
            if spec is None:
                check(f"{label}: spec built", False, "returned None")
                continue
            check(f"{label}: kv_layer_frac == {want_frac:.4g}",
                  abs(spec.kv_layer_frac - want_frac) < 1e-9, f"got {spec.kv_layer_frac}")
            check(f"{label}: full_attention_layers == {want_full}",
                  spec.full_attention_layers == want_full, f"got {spec.full_attention_layers}")
            # the two must never disagree — one is derived from the other
            check(f"{label}: frac and layer count agree",
                  abs(spec.full_attention_layers
                      - round(spec.num_layers * spec.kv_layer_frac)) <= 0,
                  f"frac {spec.kv_layer_frac} vs count {spec.full_attention_layers}")


def test_builtin_table_carries_arch_facts() -> None:
    """A hard-coded MODEL_SPECS entry SHORT-CIRCUITS the config (resolve_spec returns it without
    reading config.json), so a fact missing from the table is missing for good."""
    print("\n[7] built-in MODEL_SPECS table carries the arch facts it needs")
    import placement
    q = placement.MODEL_SPECS.get("Qwen/Qwen3.6-35B-A3B")
    check("Qwen3.6-35B-A3B present in table", q is not None)
    if q is not None:
        # Read from the real checkpoint 2026-08-17: 30 linear_attention + 10 full_attention of 40,
        # plus a top-level vision_config.
        check("Qwen3.6-35B-A3B is hybrid", q.is_hybrid is True)
        check("Qwen3.6-35B-A3B is multimodal", q.is_multimodal is True)
        check("Qwen3.6-35B-A3B funds 10 of 40 layers", q.full_attention_layers == 10,
              f"got {q.full_attention_layers}")
        import dataclasses
        all_layers = dataclasses.replace(q, kv_layer_frac=1.0)
        check("Qwen3.6-35B-A3B reserves 1/4 the KV of the all-layer figure",
              q.kv_bytes_per_layer(8192) * 4 == all_layers.kv_bytes_per_layer(8192),
              f"{q.kv_bytes_per_layer(8192)} vs {all_layers.kv_bytes_per_layer(8192)}")

    # Every OTHER table entry is a dense text model: unchanged planning, bit-identical.
    for key, s in placement.MODEL_SPECS.items():
        if key == "Qwen/Qwen3.6-35B-A3B":
            continue
        check(f"{s.name}: still dense/non-hybrid",
              s.is_hybrid is False and s.is_multimodal is False
              and s.full_attention_layers == s.num_layers,
              f"hybrid={s.is_hybrid} mm={s.is_multimodal} full={s.full_attention_layers}")


def test_kvless_predicate_matches_the_worker() -> None:
    """THE test with teeth. The controller PLANS the KV reservation; the worker BUILDS it. If the
    controller thinks a layer kind is KV-less and the worker does not, the planner funds at zero
    what the worker reserves in full — a decode-time OOM.

    This caught a real defect during development: the first draft used a frozenset naming
    mamba/recurrent/gated_deltanet/short_conv/conv, six of which the worker's substring test does
    not recognise. Rather than assert a hand-copied expectation, this EXTRACTS the worker's
    function from shard_build.py and runs both over a shared vocabulary — so the two cannot drift
    without this failing, which is exactly what a per-branch invariant needs.
    """
    print("\n[8] PARITY: controller KV-less predicate == worker's shard_build predicate")
    src = (REPO / "shard_build.py").read_text(encoding="utf-8")
    tree = ast.parse(src)
    fn = next((n for n in tree.body
               if isinstance(n, ast.FunctionDef) and n.name == "_is_linear_attn_type"), None)
    check("worker predicate _is_linear_attn_type still exists", fn is not None,
          "renamed or removed — re-point this test before trusting it")
    if fn is None:
        return
    # Compile just that function (pure stdlib, no torch) and run it side by side.
    ns: dict = {}
    exec(compile(ast.Module(body=[fn], type_ignores=[]), "<worker>", "exec"), ns)
    worker = ns["_is_linear_attn_type"]

    from placement import _is_kvless_layer_type as ctrl

    vocab = ["full_attention", "sliding_attention", "linear_attention", "Linear_Attention",
             "linear_deltanet", "gated_deltanet", "mamba", "mamba2", "recurrent", "short_conv",
             "conv", "chunked_linear_attn", "brand_new_attention", "", "attention"]
    disagree = [t for t in vocab if bool(worker(t)) != bool(ctrl(t))]
    check(f"both agree on all {len(vocab)} layer-type spellings", not disagree,
          f"DIVERGENT: {disagree} — controller would plan a KV the worker does not, or vice versa")

    # And the asymmetry that makes divergence dangerous in one direction only.
    over = [t for t in vocab if ctrl(t) and not worker(t)]
    check("controller never calls KV-less what the worker funds", not over,
          f"UNDER-RESERVE risk on {over}")


def main() -> int:
    print("#perf-facts — dead arch facts reaching perf_profile.resolve()")
    test_no_dead_spec_reads()
    test_every_call_site_passes_every_arch_fact()
    test_is_hybrid_and_full_attention_layers()
    test_plan_gate_parity()
    test_multimodal_detection()
    test_kv_layer_frac_from_layer_types()
    test_builtin_table_carries_arch_facts()
    test_kvless_predicate_matches_the_worker()
    print(f"\n{CHECKS - len(FAILURES)}/{CHECKS} checks passed")
    if FAILURES:
        print("\nFAILURES:")
        for f in FAILURES:
            print(f"  - {f}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
