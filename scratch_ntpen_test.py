"""#ntpen equivalence: is the HEAD-side penalised reduction identical to the controller's legacy
full-row path?

The whole feature rests on one claim — moving repeat/presence/frequency penalties from the
controller onto the head changes the WIRE and nothing else. That claim is worth distrusting: the
two paths differ in dtype (model dtype vs the controller's .float()), in width (clipped slice vs
-inf-masked full row), in ORDER (penalise-then-clip vs clip-then-penalise) and in who does the
top-K. Each of those is a place a plausible-but-wrong token could come from, and a wrong token here
is invisible downstream — it is a perfectly well-formed sample from the wrong distribution.

So compare the two implementations directly, on adversarial rows:
  argmax  -> must be BIT-EXACT (same token id), including ties and beyond-clip maxima
  topk    -> the candidate SET must be the true top-K of the controller's penalised row, and the
             values must match it elementwise (the controller then samples over them, so a right
             set with wrong values is still wrong)

Run:  python3 scratch_ntpen_test.py
"""
import ast
import os
import random
import sys
import textwrap

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_forward import _diet_argmax, _diet_topk          # noqa: E402


def _lift(path, names):
    """Pull method sources straight out of engine_gen.py and exec them as plain functions.

    Deliberately NOT `import engine_gen`: that drags in the controller's whole module graph
    (registry, wire, the http layer) which does not exist on a worker box — and the point here is
    to compare the two implementations, not to boot a controller. Neither method touches `self`,
    so lifting them is faithful; if that ever stops being true this raises instead of silently
    testing a stale copy."""
    tree = ast.parse(open(path, encoding="utf-8").read())
    out, want = {}, set(names)
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)) and node.name in want:
            src = textwrap.dedent(ast.get_source_segment(
                open(path, encoding="utf-8").read(), node))
            g = {"__builtins__": __builtins__}
            exec(compile(ast.parse(src), path, "exec"), g)        # noqa: S102
            out[node.name] = g[node.name]
            want.discard(node.name)
    if want:
        sys.exit(f"could not lift {sorted(want)} from {path} — anchors drifted")
    return out


_HERE = os.path.dirname(os.path.abspath(__file__))
_L = _lift(os.path.join(_HERE, "engine_gen.py"), ["_penalized", "_pen_directive"])


class ENG:                       # the controller half, as lifted (self is always ignored)
    _penalized = staticmethod(lambda self, *a: _L["_penalized"](self, *a))
    _pen_directive = staticmethod(lambda self, *a: _L["_pen_directive"](self, *a))


def legacy_row(logits, clip, prompt_ids, hist, sp):
    """Exactly what _decode_plain does on a full-row reply: #21 tail mask, then _penalized."""
    row = logits[0, -1]
    if clip and clip < int(row.shape[-1]):
        row = row.clone()
        row[clip:] = float("-inf")
    return ENG._penalized(None, row, prompt_ids, hist, sp)


def case(rng, *, vocab, clip, dtype, mode):
    """One adversarial trial. Rows are deliberately nasty: a heavy -3..3 spread (so plenty of
    logits straddle 0, where repeat_penalty's divide/multiply branch flips), duplicated maxima
    (ties), and — when clip < vocab — a beyond-clip token planted as the GLOBAL max, which is the
    #21 case that made the clip mask necessary in the first place."""
    logits = (torch.randn(1, 1, vocab, generator=rng) * 3).to(dtype)
    # ties: repeat one value at two indices
    a, b = rng_ints(rng, clip, 2)
    logits[0, 0, b] = logits[0, 0, a]
    if clip < vocab:                       # a beyond-clip global max must never be selected
        logits[0, 0, rng_ints(rng, vocab - clip, 1)[0] + clip] = 99.0

    plen = int(torch.randint(1, 40, (1,), generator=rng))
    olen = int(torch.randint(0, 30, (1,), generator=rng))
    prompt_ids = rng_ints(rng, clip, plen)
    hist = rng_ints(rng, clip, olen)
    sp = {"repeat_penalty": float(torch.empty(1).uniform_(0.5, 1.8, generator=rng)),
          "presence_penalty": random.choice([0.0, 0.0, 0.7, -0.4]),
          "frequency_penalty": random.choice([0.0, 0.0, 0.5, -0.3]),
          "repeat_last_n": random.choice([64, 8, -1, 0, 1000])}

    pen = ENG._pen_directive(None, prompt_ids, hist, sp)
    ref = legacy_row(logits, clip, prompt_ids, hist, sp)

    if mode == "argmax":
        want = int(ref.float().argmax())
        got = int(_diet_argmax(logits, clip, pen).reshape(-1)[-1])
        return ("argmax", want, got, want == got, sp, pen)

    k = 8
    wv, wi = torch.topk(ref.float(), k)
    gv, gi = _diet_topk(logits, clip, k, pen)
    ok = torch.equal(wi, gi.long()) and torch.allclose(wv, gv.float(), rtol=0, atol=0)
    return ("topk", (wi.tolist(), wv.tolist()), (gi.tolist(), gv.float().tolist()), ok, sp, pen)


def rng_ints(rng, hi, n):
    return [int(x) for x in torch.randint(0, max(1, hi), (n,), generator=rng)]


def main():
    rng = torch.Generator().manual_seed(20260813)
    random.seed(20260813)
    fails, runs = [], 0
    # dtype matters: bf16 has ~8 mantissa bits, so a penalty can collapse two logits onto the
    # SAME bf16 value and move a tie — if the two paths disagree anywhere it will be here.
    for dtype in (torch.float32, torch.bfloat16, torch.float16):
        for vocab, clip in ((512, 512), (2048, 1500), (32000, 31000)):
            for mode in ("argmax", "topk"):
                for _ in range(40):
                    runs += 1
                    kind, want, got, ok, sp, pen = case(
                        rng, vocab=vocab, clip=clip, dtype=dtype, mode=mode)
                    if not ok:
                        fails.append((dtype, vocab, clip, kind, want, got, sp, pen))
    print(f"\n{runs} trials  x {{fp32,bf16,fp16}} x {{512,2048,32000}} vocab x {{argmax,topk}}")
    if fails:
        print(f"FAIL: {len(fails)}/{runs}")
        for f in fails[:5]:
            print(f"  dtype={f[0]} vocab={f[1]} clip={f[2]} {f[3]}\n"
                  f"    controller: {f[4]}\n    head:       {f[5]}\n    sp={f[6]}\n    pen={f[7]}")
        sys.exit(1)
    print("PASS — head-side penalised reduction is identical to the controller's full-row path")

    # --- the directive is also the WHOLE point: show what it costs on the wire ------------------
    prompt = list(range(4096))
    hist = list(range(50))
    for n in (64, 256, -1):
        d = ENG._pen_directive(None, prompt, hist,
                               {"repeat_penalty": 1.1, "repeat_last_n": n})
        import json
        print(f"  repeat_last_n={n:>5}: {len(d['rp_ids']):>5} ids, "
              f"{len(json.dumps(d)):>6} B of header  (vs 304 KB for a 152k-vocab bf16 row)")


if __name__ == "__main__":
    main()
