"""#perf-auto regression gate: no knob may be taken twice, and head_quant must never be int4.

Two separate failures are covered, because the bug that prompted this had both shapes:

  a. `take()` is LAST-WINS (`out[name] = value`, unconditional when the caller left the knob
     unset). So a second `take("x", ...)` anywhere later in resolve() silently erases the first,
     including all of its reasoning. Section 8 did exactly that to section 4b's head_quant and the
     feature shipped 100% inert. A duplicate take is therefore a BUG BY CONSTRUCTION — the second
     call is either dead or destructive, never both-correct. This test walks the AST so it catches
     a re-take in any future section without anyone having to think about ordering.

  b. head_quant must never resolve to "int4" on ANY device class. #8 measured the int4 head at
     +0.0389 nats / 84.7% top-1 — 92.5% of the damage the whole int4 body does — and client.py's
     validator refuses the value, so proposing it makes ⚡ suggest a load that fails.

Run:  python3 scratch_perf_profile_takeonce_test.py    (prints PASS lines; non-zero exit on failure)
"""
import ast
import sys

import perf_profile as pp

FAILED = []


def check(label, ok, detail=""):
    print(f"  {'PASS' if ok else 'FAIL'}  {label}" + (f"  -- {detail}" if detail and not ok else ""))
    if not ok:
        FAILED.append(label)


# ---------------------------------------------------------------- a. static: no duplicate take --
def test_no_duplicate_take():
    print("a. every knob is take()n at most once (take() is last-wins, so a 2nd call erases the 1st)")
    src = open(pp.__file__, encoding="utf-8").read()
    tree = ast.parse(src)
    fn = next(n for n in ast.walk(tree)
              if isinstance(n, ast.FunctionDef) and n.name == "resolve")
    seen: dict[str, list[int]] = {}
    for node in ast.walk(fn):
        if (isinstance(node, ast.Call) and isinstance(node.func, ast.Name)
                and node.func.id == "take" and node.args
                and isinstance(node.args[0], ast.Constant)):
            seen.setdefault(node.args[0].value, []).append(node.lineno)
    dupes = {k: v for k, v in seen.items() if len(v) > 1}
    # A knob taken once per branch of the SAME if/else is fine — those are mutually exclusive.
    # Detect that by checking whether the call sites share an enclosing If with distinct bodies.
    real = {}
    for knob, lines in dupes.items():
        exclusive = False
        for node in ast.walk(fn):
            if isinstance(node, ast.If):
                in_body = _lines_under(node.body)
                in_else = _lines_under(node.orelse)
                if all(l in in_body | in_else for l in lines) and \
                   any(l in in_body for l in lines) and any(l in in_else for l in lines):
                    exclusive = True
                    break
        if not exclusive:
            real[knob] = lines
    check(f"no unconditional re-take among {len(seen)} knobs", not real,
          f"re-taken: { {k: v for k, v in real.items()} }")


def _lines_under(nodes):
    out = set()
    for n in nodes:
        for sub in ast.walk(n):
            if hasattr(sub, "lineno"):
                out.add(sub.lineno)
    return out


# ------------------------------------------------------- b. behavioural: head_quant is sane -----
BASE = dict(params_b=7.6, num_layers=28, num_kv_heads=4, head_dim=128, vocab_size=152064,
            free_vram_gb=110.0, free_ram_gb=110.0, ctx=4096)


def test_head_quant_never_int4():
    print("b. head_quant never resolves to the REJECTED int4, on any device class")
    for dc in (pp.ROCM, pp.CUDA_MODERN, pp.CUDA_LEGACY, pp.UNIFIED, pp.CPU):
        for quant in ("int4", "int2", "bf16"):
            for tie in (False, True):
                out, why = pp.resolve(device_class=dc, requested={"quant": quant},
                                      tie_word_embeddings=tie, **BASE)
                hq = out.get("head_quant")
                check(f"{dc:12s} quant={quant:5s} tie={str(tie):5s} -> head_quant={hq!r}",
                      hq != "int4", f"got {hq!r} — client.py refuses this")


def test_int8_advice_survives():
    print("c. the int8 suggestion actually SURVIVES to the caller (it did not, before)")
    for dc in (pp.ROCM, pp.CUDA_MODERN):
        out, why = pp.resolve(device_class=dc, requested={"quant": "int4"},
                              tie_word_embeddings=False, **BASE)
        line = next((w for w in why if w.startswith("head_quant=")), "")
        check(f"{dc:12s} rationale mentions int8", "int8" in line, f"rationale was {line!r}")
        # section 4b takes "" (advice only, never applied) — the knob must not be forced
        check(f"{dc:12s} head_quant not auto-applied", out.get("head_quant") in ("", None),
              f"got {out.get('head_quant')!r} — this knob changes OUTPUT and must stay opt-in")


def test_explicit_still_wins():
    print("d. an explicit operator choice is still passed through untouched")
    out, _ = pp.resolve(device_class=pp.ROCM, requested={"quant": "int4", "head_quant": "int8"},
                        tie_word_embeddings=False, **BASE)
    check("explicit head_quant=int8 preserved", out.get("head_quant") == "int8",
          f"got {out.get('head_quant')!r}")


if __name__ == "__main__":
    test_no_duplicate_take()
    test_head_quant_never_int4()
    test_int8_advice_survives()
    test_explicit_still_wins()
    print()
    if FAILED:
        print(f"FAILED ({len(FAILED)}): " + "; ".join(FAILED))
        sys.exit(1)
    print("all checks passed")
