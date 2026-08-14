"""#logits-diet scratch validation (pure-local, NO network).

Drives synthetic logits rows through the REAL worker-side head helpers
(shard_forward._diet_argmax / _diet_topk), the REAL wire manifest packers
(wire._pack_ntensor / _unpack_ntensor) and the REAL controller sampler
(engine_gen.EngineGenMixin._sample — its body never touches self), proving:

  A. argmax mode is BIT-EXACT vs the controller's legacy paths:
     - plain greedy: row.clone(); row[ntok:] = -inf; int(row.float().argmax())
     - spec verify:  int(V[0, i].argmax()) per position, RAW row (no clip)
     across random rows, rows whose true max lies BEYOND ntok, and exact ties
     (first-occurrence), in bf16/fp16/fp32, on CPU and (if present) CUDA,
     with every reply round-tripped through the real pack/unpack.
  B. topk mode preserves the sampling distribution on realistic peaked rows:
     top-4096 covers >>0.9999 of the softmax mass; nucleus (top_p) and min_p
     survivor SETS match the full row; top_k=1 draws are identical; and a flat
     adversarial row shows why INFINITEMODEL_TOPK_WIRE=0 (full-row fallback)
     exists — plus the K=0 gate itself.
"""
import os
import sys

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_forward import _diet_argmax, _diet_topk          # real worker-side helpers
from wire import (_pack_ntensor, _unpack_ntensor,           # real wire manifest
                  NT_TOKEN_IDS, NT_TOPK_VALS, NT_TOPK_IDX)
import engine_gen                                           # real controller sampler + K gate

_sample = engine_gen.EngineGenMixin._sample                 # body never uses self
PASS = FAIL = 0


def check(name, ok, detail=""):
    global PASS, FAIL
    if ok:
        PASS += 1
    else:
        FAIL += 1
        print(f"  FAIL {name}  {detail}")


def _wire_roundtrip(parts):
    """Push parts through the REAL pack -> unpack and return the reassembled dict."""
    tmeta, raw = _pack_ntensor(parts)
    back = _unpack_ntensor(tmeta, raw)
    assert [k for k, _ in back] == [k for k, _ in parts], "kind order changed on the wire"
    for (_, a), (_, b) in zip(parts, back):
        assert a.dtype == b.dtype and list(a.shape) == list(b.shape), "meta drift on the wire"
        assert torch.equal(a.cpu(), b), "payload bytes changed on the wire"
    return dict(back)


def ctrl_greedy(logits, ntok):
    """Controller LEGACY plain-greedy: #21 tail mask then _sample at temperature<=0
    (which is row.float().argmax()) — verbatim semantics of engine_gen._decode_plain."""
    row = logits[0, -1]
    if ntok and ntok < int(row.shape[-1]):
        row = row.clone()
        row[ntok:] = float("-inf")
    return int(_sample(None, row, 0.0, 1.0))


def ctrl_verify(logits):
    """Controller LEGACY spec-verify: RAW per-position argmax (no mask, no .float())."""
    return [int(logits[0, i].argmax()) for i in range(logits.shape[1])]


def run_argmax_suite(dev):
    torch.manual_seed(1234)
    V = 151936                                   # Qwen-sized head
    for dt in (torch.bfloat16, torch.float16, torch.float32):
        # -- plain greedy: random rows + tail-max + ties, packed over the real wire --------
        for trial in range(40):
            logits = (torch.randn(1, 1, V) * 4).to(dt).to(dev)
            ntok = [0, V, V - 4096, 1000, 1][trial % 5]      # incl. no-clip + tiny-clip edges
            if trial % 3 == 1 and 0 < ntok < V:              # plant the GLOBAL max beyond ntok
                logits[0, -1, ntok + (trial % 97)] = 1e4
            if trial % 3 == 2:                               # exact tie inside the clip window
                lo = min(ntok or V, V) - 1
                a, b = (trial * 7) % max(1, lo), (trial * 13 + 1) % max(1, lo)
                mval = (logits[0, -1, : (ntok or V)].max() + 2).to(dt)
                logits[0, -1, a] = mval
                logits[0, -1, b] = mval                      # first occurrence must win
            ids = _diet_argmax(logits, ntok)                 # REAL worker helper (on dev)
            got = int(_wire_roundtrip([(NT_TOKEN_IDS, ids)])[NT_TOKEN_IDS].reshape(-1)[-1])
            want = ctrl_greedy(logits.cpu(), ntok)
            check(f"greedy {dt} {dev} t{trial} ntok={ntok}", got == want, f"{got} != {want}")
            check(f"greedy-dtype {dt} {dev} t{trial}", ids.dtype == torch.int64, str(ids.dtype))
        # -- spec verify: [1, K+1, V] raw argmax per position (clip=0) ---------------------
        for K in (1, 4, 8):
            Vt = (torch.randn(1, K + 1, V) * 4).to(dt).to(dev)
            Vt[0, K // 2, V - 3] = 1e4                       # raw semantics: tail CAN win
            Vt[0, 0, 5] = Vt[0, 0, 9] = (Vt[0, 0].max() + 2).to(dt)   # tie -> first index (5)
            ids = _diet_argmax(Vt, 0)
            got = [int(t) for t in
                   _wire_roundtrip([(NT_TOKEN_IDS, ids)])[NT_TOKEN_IDS].reshape(-1).tolist()]
            want = ctrl_verify(Vt.cpu())
            check(f"verify {dt} {dev} K={K}", got == want, f"{got} != {want}")
        # clip generality on multi-position rows (not used by spec today, but the helper
        # must clip every position identically if a future path asks for it)
        Vt = (torch.randn(1, 3, V) * 4).to(dt).to(dev)
        Vt[0, 1, V - 2] = 1e4
        ids = _diet_argmax(Vt, 1000)
        want = []
        for i in range(3):
            r = Vt[0, i].cpu().clone()
            r[1000:] = float("-inf")
            want.append(int(r.float().argmax()))
        check(f"verify-clip {dt} {dev}", [int(t) for t in ids.tolist()] == want)


def run_topk_suite(dev):
    torch.manual_seed(99)
    V, K = 151936, 4096
    # -- realistic peaked row: a plausible LM decode distribution. Real decode rows put the
    # top candidates 15-25 logits above the bulk (top-1 prob commonly 0.3-0.95); the bulk is
    # a ~N(0, 1.5^2) sea. With that separation the beyond-top-K tail holds ~1e-5 of the mass.
    row = (torch.randn(V) * 1.5)
    row[torch.randperm(V)[:50]] += torch.linspace(16.0, 24.0, 50)    # a peaked head
    logits = row.to(torch.bfloat16).reshape(1, 1, V).to(dev)
    ntok = V - 4096                                                   # exercise the clip too
    vals, idx = _diet_topk(logits, ntok, K)                           # REAL worker helper
    by = _wire_roundtrip([(NT_TOPK_VALS, vals), (NT_TOPK_IDX, idx)])
    vals, idx = by[NT_TOPK_VALS].reshape(-1), by[NT_TOPK_IDX].reshape(-1)
    check("topk shapes", vals.shape[-1] == K and idx.shape[-1] == K)
    check("topk dtypes", vals.dtype == torch.bfloat16 and idx.dtype == torch.int64,
          f"{vals.dtype} {idx.dtype}")
    check("topk respects clip", int(idx.max()) < ntok, f"max id {int(idx.max())} >= {ntok}")
    # full-row reference (controller legacy view of the SAME clipped row)
    full = logits[0, -1].cpu().clone()
    full[ntok:] = float("-inf")
    pf = torch.softmax(full.float(), -1)
    mass = float(pf[idx].sum())
    check("top-4096 softmax mass >>0.9999 on a peaked row", mass > 0.9999, f"mass={mass:.8f}")
    print(f"  [stat] peaked-row top-{K} softmax mass = {mass:.10f}")
    # nucleus (top_p=0.9) survivor set: full row vs candidate space. Exact set equality is NOT
    # the contract — torch.sort is unstable over bf16 ties and the (1 - tail_mass) renorm shifts
    # the cumsum boundary by ~1e-7 — the contract is that any disagreement is confined to
    # boundary/tied tokens of NEGLIGIBLE probability. Assert the symmetric difference's mass.
    def nucleus(p):
        sp, si = torch.sort(p, descending=True)
        keep = torch.cumsum(sp, 0) - sp <= 0.9
        return set(si[keep].tolist())
    pc = torch.softmax(vals.float(), -1)
    nf, nc = nucleus(pf), {int(idx[i]) for i in nucleus(pc)}
    sd = nf ^ nc
    sd_mass = float(pf[list(sd)].sum()) if sd else 0.0
    # An EXACT-TIE swap (two tokens with identical bf16 logits, one picked by each sort) is
    # distribution-NEUTRAL — and the legacy full-row path's own unstable sort already swaps
    # such pairs run-to-run/device-to-device, so it is not divergence the diet introduced.
    pa = sorted(float(pf[t]) for t in (nf - nc))
    pb = sorted(float(pf[t]) for t in (nc - nf))
    tie_swap = len(pa) == len(pb) and all(abs(a - b) <= 1e-9 for a, b in zip(pa, pb))
    check("top_p=0.9 nucleus matches up to exact-tie swaps",
          sd_mass < 1e-4 or tie_swap, f"|sym-diff|={len(sd)} mass={sd_mass:.2e}")
    print(f"  [stat] nucleus |full|={len(nf)} |cand|={len(nc)} sym-diff={len(sd)} "
          f"(mass {sd_mass:.2e}, tie_swap={tie_swap})")
    # min_p survivor set identical (ratios are renormalization-invariant)
    for mp in (0.05, 0.1):
        sf = set(torch.nonzero(pf >= mp * pf.max()).reshape(-1).tolist())
        sc = {int(idx[i]) for i in torch.nonzero(pc >= mp * pc.max()).reshape(-1).tolist()}
        check(f"min_p={mp} survivor set identical", sf == sc, f"|f|={len(sf)} |c|={len(sc)}")
    # top_k=1 (the determinism probe) draws the same token through the REAL _sample
    g1, g2 = torch.Generator().manual_seed(7), torch.Generator().manual_seed(7)
    t_full = int(_sample(None, full, 0.8, 1.0, 0.0, top_k=1, gen=g1))
    t_cand = int(idx[_sample(None, vals, 0.8, 1.0, 0.0, top_k=1, gen=g2)])
    check("top_k=1 identical draw", t_full == t_cand, f"{t_full} != {t_cand}")
    # every seeded candidate-space draw must be a true top-K member of the full row
    true_topk = set(torch.topk(full, K).indices.tolist())
    ok = True
    for s in range(20):
        g = torch.Generator().manual_seed(s)
        t = int(idx[_sample(None, vals, 1.0, 0.95, 0.05, top_k=0, gen=g)])
        ok = ok and (t in true_topk)
    check("20 seeded draws all land in the true top-K", ok)
    # -- adversarial FLAT row: the case the K=0 fallback exists for -------------------------
    flat = torch.zeros(1, 1, V, dtype=torch.bfloat16).to(dev)
    fv, fi = _diet_topk(flat, 0, K)
    fmass = float(torch.softmax(flat[0, -1].cpu().float(), -1)[fi.cpu()].sum())
    check("flat row: top-4096 mass is tiny (truncation would be real)",
          fmass < 0.05, f"mass={fmass:.4f}")
    print(f"  [stat] flat-row top-{K} softmax mass = {fmass:.6f} (= {K}/{V} — why "
          f"INFINITEMODEL_TOPK_WIRE=0 keeps the full row)")
    # the K=0 fallback gate: env=0 -> _wire_topk_k()=0 -> _decode_plain leaves nt_mode=None
    old = os.environ.get("INFINITEMODEL_TOPK_WIRE")
    try:
        os.environ["INFINITEMODEL_TOPK_WIRE"] = "0"
        k0 = engine_gen._wire_topk_k()
        check("INFINITEMODEL_TOPK_WIRE=0 -> K=0", k0 == 0, str(k0))
        _nt_mode, _nt_k = None, 0            # the _decode_plain gate, verbatim
        _pen, temperature = False, 0.8
        if not _pen:
            if not temperature or temperature <= 0:
                _nt_mode = "argmax"
            else:
                _nt_k = engine_gen._wire_topk_k()
                if _nt_k > 0:
                    _nt_mode = "topk"
        check("K=0 leaves sampled decode on the full row", _nt_mode is None, str(_nt_mode))
        os.environ["INFINITEMODEL_TOPK_WIRE"] = "notanint"
        check("bad env falls back to 4096", engine_gen._wire_topk_k() == 4096)
        os.environ.pop("INFINITEMODEL_TOPK_WIRE", None)
        check("default K=4096", engine_gen._wire_topk_k() == 4096)
    finally:
        if old is None:
            os.environ.pop("INFINITEMODEL_TOPK_WIRE", None)
        else:
            os.environ["INFINITEMODEL_TOPK_WIRE"] = old
    # k larger than the clipped row degrades to a full sorted row
    small = (torch.randn(1, 1, 300) * 3).to(torch.bfloat16).to(dev)
    sv, si = _diet_topk(small, 100, 4096)
    check("k>width degrades to full clipped row", sv.shape[-1] == 100 and si.shape[-1] == 100)
    # reference on the SAME device: topk's ordering of EXACTLY-TIED values is device-specific
    # (bf16 rounding makes dupes likely on a 300-wide row); the helper's claim is "== torch.topk
    # on its own device", and a tie permutation is distribution-neutral for sampling anyway.
    ref = torch.topk(small[0, -1, :100], 100)
    check("degraded row bit-equal to torch.topk (same device)",
          torch.equal(sv, ref.values.cpu()) and torch.equal(si, ref.indices.cpu()))


def main():
    devs = ["cpu"] + (["cuda"] if torch.cuda.is_available() else [])
    print(f"[diet-test] devices: {devs}")
    for dev in devs:
        print(f"[diet-test] A: argmax bit-exactness on {dev}")
        run_argmax_suite(dev)
        print(f"[diet-test] B: topk distribution preservation on {dev}")
        run_topk_suite(dev)
    # wire byte-size sanity: what the diet actually saves per decode token (bf16 Qwen head)
    fullrow = (torch.randn(1, 1, 151936)).to(torch.bfloat16)
    _, full_raw = _pack_ntensor([(0, fullrow)])
    ids = _diet_argmax(fullrow, 151669)
    _, ids_raw = _pack_ntensor([(NT_TOKEN_IDS, ids)])
    v, i = _diet_topk(fullrow, 151669, 4096)
    _, tk_raw = _pack_ntensor([(NT_TOPK_VALS, v), (NT_TOPK_IDX, i)])
    print(f"[stat] per-token payload: full row {len(full_raw)} B -> argmax {len(ids_raw)} B "
          f"(x{len(full_raw) / len(ids_raw):.0f}) -> topk4096 {len(tk_raw)} B "
          f"(x{len(full_raw) / len(tk_raw):.1f})")
    print(f"[diet-test] {PASS} passed, {FAIL} failed")
    sys.exit(1 if FAIL else 0)


if __name__ == "__main__":
    main()
