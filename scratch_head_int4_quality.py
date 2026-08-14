"""#8 quality gate: what does int4-g128 quantizing `lm_head` actually cost in output quality?

The head is the most logit-sensitive tensor in the model, and `prepare_fused`'s existing self-check
only compares fused-vs-naive WITHIN int4 — it can't see int4-vs-bf16 degradation at all. So before
any int4 head can be trusted on the serving path, something has to measure the damage. This is that
something, and it is deliberately NOT a threshold pulled out of the air.

THE FRAMING THAT MAKES A NUMBER MEANINGFUL: "head int4 costs X nats" is unjudgeable on its own. What
IS judgeable is the head's cost measured against the DECODER BODY's int4 cost — a degradation the
project already ships, has served for months, and considers acceptable. So this runs three variants
over the same tokens:

    bf16 body + bf16 head    the reference
    int4 body + bf16 head    what is served TODAY  <- the accepted-damage baseline
    int4 body + int4 head    the proposal

If the head's marginal cost is small next to the body's, the answer is a measurement rather than an
opinion. If it is comparable or larger, the head stays bf16 and #8 is closed for good reasons.

Quantization is the project's OWN `shard_compile.pack_linear_int4` round-tripped through
worker_quant's exact `_dequant` convention (w = (q - zero) * scale, low nibble = even column) — this
tests the quantization that would actually ship, not a generic int4.

Metrics per variant: NLL/perplexity on real held-out text (the thing users feel), greedy top-1
agreement with the bf16 reference (how often the SAMPLED token would change), and KL(ref||variant)
(how much the whole distribution moved, which top-1 agreement hides).

CPU-only and device-independent — quantization error is arithmetic, so this answers the quality
question without touching a GPU or the serving fleet. Speed is measured separately, on gfx1151.

Run (mini05 hosts the model store and has 16 cores):
    python3 scratch_head_int4_quality.py --model <ckpt-dir> --tokens 2048
"""
import argparse
import os
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from shard_compile import pack_linear_int4                      # noqa: E402


def dequant_int4(qpacked, scale, zero, in_features, group_size, dtype=torch.bfloat16):
    """worker_quant.QuantLinear4._dequant, verbatim in convention: low nibble = EVEN input column,
    w = (q - zero) * scale, then crop the group padding back off."""
    out = qpacked.shape[0]
    lo = (qpacked & 0x0F).to(torch.int16)
    hi = (qpacked >> 4).to(torch.int16)
    q = torch.stack((lo, hi), dim=2).reshape(out, -1)
    ng = scale.shape[1]
    qf = q.reshape(out, ng, group_size).to(dtype)
    w = (qf - zero.to(dtype).unsqueeze(2)) * scale.to(dtype).unsqueeze(2)
    return w.reshape(out, ng * group_size)[:, :in_features].contiguous()


def roundtrip(W, group_size=128):
    qp, sc, ze, in_f = pack_linear_int4(W, group_size)
    return dequant_int4(qp, sc, ze, in_f, group_size, W.dtype)


def quantize_body_(model, group_size=128):
    """Round-trip every DECODER Linear through int4 in place — the same scope the worker quantizes
    (`.layers.` 2D weights; embed/head/norms/router-gate untouched). Returns the count."""
    n = 0
    for name, mod in model.named_modules():
        if not isinstance(mod, torch.nn.Linear) or ".layers." not in name:
            continue
        with torch.no_grad():
            mod.weight.copy_(roundtrip(mod.weight.data, group_size))
        n += 1
    return n


@torch.no_grad()
def hidden_states(model, ids, chunk=512):
    """Post-final-norm hidden for every position — exactly the tensor the head consumes.
    `model.model(...)` returns it directly (the final norm is inside), so there is no ambiguity
    about whether hidden_states[-1] is pre- or post-norm."""
    outs = []
    past = None
    for i in range(0, ids.shape[1], chunk):
        o = model.model(input_ids=ids[:, i:i + chunk], past_key_values=past, use_cache=True)
        past = o.past_key_values
        outs.append(o.last_hidden_state.float())
    return torch.cat(outs, dim=1)


@torch.no_grad()
def head_metrics(h, W, targets, ref_logp=None, chunk=128):
    """Stream the head over positions so a [T, 151936] fp32 logits tensor never exists at once.
    Returns (nll, top1 ids, mean KL vs ref_logp). ref_logp is the reference log-softmax, streamed
    in the same chunks; None on the reference pass itself."""
    T = h.shape[0]
    nll_sum, kl_sum, top1 = 0.0, 0.0, []
    for i in range(0, T, chunk):
        hs = h[i:i + chunk]
        lg = F.linear(hs, W.float())
        lp = F.log_softmax(lg, dim=-1)
        tg = targets[i:i + chunk]
        nll_sum += float(-lp.gather(1, tg.unsqueeze(1)).sum())
        top1.append(lg.argmax(-1))
        if ref_logp is not None:
            rp = ref_logp[i:i + chunk]
            kl_sum += float((rp.exp() * (rp - lp)).sum())
        del lg, lp
    return nll_sum / T, torch.cat(top1), kl_sum / T


@torch.no_grad()
def ref_logprobs(h, W, chunk=128):
    return torch.cat([F.log_softmax(F.linear(h[i:i + chunk], W.float()), -1)
                      for i in range(0, h.shape[0], chunk)])


def weight_error_screen(d, groups):
    """The CHEAP screen — seconds, no model execution, no GPU: read tensors straight out of the
    safetensors shards and report how badly each one round-trips through int4.

    Run this BEFORE paying for a perplexity experiment. Group min/max quantization is destroyed by
    OUTLIERS inside a group, so the first question about any candidate tensor is simply whether it
    is unusually outlier-heavy compared to the tensors already being quantized. If the head's error
    sits in the same band as the decoder linears, the weights are ordinary and any damage must come
    from WHERE it sits (straight into a softmax, with no downstream layers to average the error
    away) rather than from what it contains — which is exactly what the perplexity run measures.

    Reports kurtosis (3.0 == gaussian) and max/p99.9 alongside the error so an outlier-driven result
    is distinguishable from a merely-wide distribution."""
    import glob
    import json
    from safetensors import safe_open
    idx = json.load(open(os.path.join(d, "model.safetensors.index.json")))["weight_map"]
    files = {os.path.basename(f): f for f in sorted(glob.glob(os.path.join(d, "*.safetensors")))}
    want = [n for n in ("lm_head.weight",) if n in idx]
    for n in idx:
        if n.endswith("proj.weight") and (".layers.0." in n or ".layers.13." in n):
            want.append(n)
    print(f"{os.path.basename(d)}   int4 round-trip error by group size")
    print(f"{'tensor':32s} {'shape':>16s} {'kurt':>7s} {'max/p999':>9s} "
          + " ".join(f"{'g%d' % g:>7s}" for g in groups))
    for name in want:
        with safe_open(files[idx[name]], framework="pt") as fh:
            W = fh.get_tensor(name)
        if W.dim() != 2:
            continue
        Wf = W.float()
        sd = Wf.std()
        kurt = float(((Wf - Wf.mean()) ** 4).mean() / (sd ** 4))
        p999 = float(Wf.abs().flatten().kthvalue(max(1, int(0.999 * Wf.numel()))).values)
        errs = [float((roundtrip(W, g).float() - Wf).abs().mean() / Wf.abs().mean())
                for g in groups]
        print(f"{name.replace('model.layers.', 'L').replace('.weight', ''):32s} "
              f"{str(tuple(W.shape)):>16s} {kurt:7.1f} {float(Wf.abs().max())/(p999+1e-12):9.2f} "
              + " ".join(f"{e:7.4f}" for e in errs))
        del W, Wf


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--tokens", type=int, default=2048)
    ap.add_argument("--group", type=int, default=128)
    ap.add_argument("--text", default="")
    ap.add_argument("--cache", default="", help="path to cache the two trunk passes")
    ap.add_argument("--sweep", default="", help="comma group sizes for the head sweep")
    ap.add_argument("--sweep-only", action="store_true",
                    help="skip the bf16 trunk: sweep head variants against TODAY (int4 body + "
                         "bf16 head). Halves the cost when the reference framing is already known.")
    ap.add_argument("--werr", action="store_true",
                    help="weights-only screen: per-tensor round-trip error, no model execution")
    a = ap.parse_args()
    torch.set_num_threads(os.cpu_count() or 4)
    if a.werr:
        weight_error_screen(a.model, [int(x) for x in (a.sweep or "128,64,32,16").split(",")])
        return
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(a.model)
    corpus = open(a.text, encoding="utf-8", errors="replace").read() if a.text else CORPUS
    ids = tok(corpus, return_tensors="pt").input_ids[:, :a.tokens]
    T = ids.shape[1] - 1
    print(f"model={os.path.basename(a.model)}  tokens={ids.shape[1]} ({T} predictions)  "
          f"group={a.group}  threads={torch.get_num_threads()}")

    t0 = time.time()
    model = AutoModelForCausalLM.from_pretrained(a.model, dtype=torch.bfloat16)
    model.eval()
    W = model.lm_head.weight.data
    print(f"loaded in {time.time()-t0:.0f}s   lm_head {tuple(W.shape)} "
          f"= {W.numel()*2/2**30:.2f} GB bf16 -> {W.numel()*0.53/2**30:.2f} GB int4-g{a.group}")
    targets = ids[0, 1:]

    Wq = roundtrip(W, a.group)
    rel = float((Wq.float() - W.float()).abs().mean() / W.float().abs().mean())
    print(f"head round-trip mean |dW|/|W| = {rel:.4f}")

    # The two trunk passes are the whole cost of this experiment; the head variants are cheap.
    # Cache them so a group-size sweep re-uses one run instead of paying for the model again.
    if a.sweep_only:
        # One trunk pass instead of two. The reference framing (how much the BODY costs) is a
        # separate, already-answered question; what a group-size sweep needs is only the served
        # trunk and a bf16-head baseline to move away from.
        print("\n[trunk] int4 body (what is served today) ...", flush=True)
        t0 = time.time()
        nlin = quantize_body_(model, a.group)
        h_b = hidden_states(model, ids)[0, :-1]
        print(f"      {nlin} decoder Linears round-tripped, {time.time()-t0:.0f}s")
        del model
        refp = ref_logprobs(h_b, W)                  # baseline = TODAY, so KL is vs today
        nll_b, top_b, _ = head_metrics(h_b, W, targets)
        print(f"\n{'head quant':14s} {'bits/w':>7s} {'GB':>6s} {'ppl':>8s} {'dNLL vs today':>14s} "
              f"{'top1 == today':>14s} {'KL(today||x)':>12s}")
        import math as _m
        print(f"{'bf16 (today)':14s} {16.0:7.2f} {W.numel()*2/2**30:6.2f} {_m.exp(nll_b):8.3f} "
              f"{0.0:+14.4f} {100.0:13.2f}% {0.0:12.5f}")
        for G in [int(x) for x in (a.sweep or "128,64,32,16").split(",")]:
            Wg = roundtrip(W, G)
            nll_g, top_g, kl_g = head_metrics(h_b, Wg, targets, refp)
            bits = 4 + 32.0 / G
            print(f"{'int4-g%d' % G:14s} {bits:7.2f} {W.numel()*bits/8/2**30:6.2f} "
                  f"{_m.exp(nll_g):8.3f} {nll_g-nll_b:+14.4f} "
                  f"{100*float((top_g == top_b).float().mean()):13.2f}% {kl_g:12.5f}")
            del Wg
        s8 = W.float().abs().amax(dim=1, keepdim=True) / 127.0
        W8 = (torch.round(W.float() / s8).clamp(-127, 127) * s8).to(W.dtype)
        nll_8, top_8, kl_8 = head_metrics(h_b, W8, targets, refp)
        print(f"{'int8 per-row':14s} {8.0:7.2f} {W.numel()/2**30:6.2f} {_m.exp(nll_8):8.3f} "
              f"{nll_8-nll_b:+14.4f} {100*float((top_8 == top_b).float().mean()):13.2f}% "
              f"{kl_8:12.5f}")
        print("\n(int8 is quality-reference ONLY — QuantLinear.forward materializes the full bf16\n"
              " weight every call, so an int8 head READS 0.55 GB and WRITES+READS 1.09 GB more:\n"
              " strictly slower than leaving it bf16. There is no w8a16 kernel, only w4a16.)")
        return

    cache = a.cache or ""
    if cache and os.path.exists(cache):
        blob = torch.load(cache)
        h_ref, h_b, nlin = blob["h_ref"], blob["h_b"], blob["nlin"]
        print(f"\nreusing cached trunk states from {cache} ({nlin} body Linears)")
    else:
        print("\n[trunk 1/2] bf16 body ...", flush=True)
        t0 = time.time()
        h_ref = hidden_states(model, ids)[0, :-1]
        print(f"      {time.time()-t0:.0f}s")
        print("[trunk 2/2] int4 body (what is served today) ...", flush=True)
        t0 = time.time()
        nlin = quantize_body_(model, a.group)
        h_b = hidden_states(model, ids)[0, :-1]
        print(f"      {nlin} decoder Linears round-tripped, {time.time()-t0:.0f}s")
        if cache:
            torch.save({"h_ref": h_ref, "h_b": h_b, "nlin": nlin}, cache)
    del model

    refp = ref_logprobs(h_ref, W)
    nll_ref, top_ref, _ = head_metrics(h_ref, W, targets)
    nll_b, top_b, kl_b = head_metrics(h_b, W, targets, refp)
    nll_bh, top_bh, kl_bh = head_metrics(h_b, Wq, targets, refp)
    nll_h, top_h, kl_h = head_metrics(h_ref, Wq, targets, refp)

    import math
    rows = [("bf16 body + bf16 head  (reference)", nll_ref, top_ref, 0.0),
            ("bf16 body + INT4 head  (head only)", nll_h, top_h, kl_h),
            ("INT4 body + bf16 head  (TODAY)", nll_b, top_b, kl_b),
            ("INT4 body + INT4 head  (proposal)", nll_bh, top_bh, kl_bh)]
    print(f"\n{'variant':38s} {'NLL':>7s} {'ppl':>8s} {'dNLL':>7s} "
          f"{'top1 == ref':>12s} {'KL(ref||x)':>11s}")
    for name, nll, top, kl in rows:
        agree = float((top == top_ref).float().mean())
        print(f"{name:38s} {nll:7.4f} {math.exp(nll):8.3f} {nll-nll_ref:+7.4f} "
              f"{100*agree:11.2f}% {kl:11.5f}")

    d_body = nll_b - nll_ref
    d_head = nll_bh - nll_b
    print(f"\nbody int4 costs {d_body:+.4f} nats  (shipped, accepted)")
    print(f"head int4 adds  {d_head:+.4f} nats  on top of it")
    if d_body > 1e-6:
        print(f"head marginal cost = {100*d_head/d_body:.1f}% of the damage already accepted")
    print(f"greedy token flips vs today: "
          f"{int((top_bh != top_b).sum())}/{T} ({100*float((top_bh != top_b).float().mean()):.2f}%)")

    if a.sweep:
        # Group size is the obvious lever if g128 is too damaging: the head's error comes from
        # min/max outliers WITHIN a group, so narrower groups clamp less. The cost is scale+zero
        # per group -> bits/weight = 4 + 32/G. Reported against the bf16 head so the quality/size
        # trade is visible in one table rather than argued about.
        print(f"\n{'head quant':14s} {'bits/w':>7s} {'GB':>7s} {'dNLL vs today':>14s} "
              f"{'top1 == today':>14s} {'KL(ref||x)':>11s}")
        print(f"{'bf16 (today)':14s} {16.0:7.2f} {W.numel()*2/2**30:7.2f} "
              f"{0.0:+14.4f} {100.0:13.2f}% {kl_b:11.5f}")
        for G in [int(x) for x in a.sweep.split(",")]:
            Wg = roundtrip(W, G)
            nll_g, top_g, kl_g = head_metrics(h_b, Wg, targets, refp)
            bits = 4 + 32.0 / G
            print(f"{'int4-g%d' % G:14s} {bits:7.2f} {W.numel()*bits/8/2**30:7.2f} "
                  f"{nll_g-nll_b:+14.4f} {100*float((top_g == top_b).float().mean()):13.2f}% "
                  f"{kl_g:11.5f}")
            del Wg
        # int8 for reference — the project ALREADY supports an int8 head (shard_compile
        # `is_int8_head`), so this row is a shippable alternative, not a hypothetical.
        s8 = W.float().abs().amax(dim=1, keepdim=True) / 127.0
        W8 = (torch.round(W.float() / s8).clamp(-127, 127) * s8).to(W.dtype)
        nll_8, top_8, kl_8 = head_metrics(h_b, W8, targets, refp)
        print(f"{'int8 per-row':14s} {8.0:7.2f} {W.numel()/2**30:7.2f} "
              f"{nll_8-nll_b:+14.4f} {100*float((top_8 == top_b).float().mean()):13.2f}% "
              f"{kl_8:11.5f}")


CORPUS = """The distributed inference controller splits one transformer model's layers across
several machines, so a model far larger than any single node's memory can still be served. Each
worker owns a contiguous run of decoder layers and forwards hidden states to the next stage over a
plain TCP connection. The last stage owns the language-model head and returns logits to the
controller, which samples the next token and feeds it back in.

Quantization is what makes this affordable. A seven-billion-parameter model in bfloat16 needs about
fifteen gigabytes of memory; packed to four bits per weight with a group size of one hundred and
twenty-eight, the same model fits in roughly five. The packing is asymmetric and group-wise: within
each group of one hundred and twenty-eight input columns, the minimum and maximum weights define a
scale and a zero point, and every weight in that group is rounded to one of sixteen levels.

The trade is accuracy. Rounding a weight to four bits introduces an error, and those errors
accumulate through the network. In practice the decoder layers tolerate this well, because each
output is a sum over thousands of terms and the individual rounding errors are close to independent,
so they partially cancel. Whether the same argument holds for the output projection is a separate
question, and one worth measuring rather than assuming.

Consider what the head actually does. It takes the final hidden state, a vector of a few thousand
numbers, and produces one score for every token in the vocabulary. The vocabulary here has more than
one hundred and fifty thousand entries. The model's prediction is the argmax of those scores, or a
sample from their softmax. What matters is not the absolute error in any single score but the error
relative to the gaps between the top few scores. When the model is confident, the gap is large and a
small perturbation changes nothing. When it is uncertain, the top candidates are close together and
a small perturbation can reorder them.

There is a second consideration. The head is the one weight matrix whose every row corresponds to a
specific token. A rare token whose row happens to quantize badly will be systematically mispredicted
whenever it should appear, in a way that averages away in aggregate statistics but shows up as a
specific, repeatable failure. Aggregate perplexity can look fine while a particular class of output
degrades. That is the kind of failure a single summary number hides, which is why agreement rates
and divergence measures are reported alongside it.

Memory bandwidth, not arithmetic, is what limits decoding speed on most hardware. Generating one
token requires reading every weight in the model exactly once, and modern accelerators can perform
far more arithmetic per second than they can fetch bytes. This is why quantization speeds up
generation at all: fewer bits per weight means fewer bytes to read, and the time per token falls
roughly in proportion. On an integrated graphics processor sharing system memory with the host, the
effect is especially pronounced, because the available bandwidth is a fraction of what a discrete
card with dedicated memory provides.

Under that model the output head is a meaningful share of the cost. In a seven-billion-parameter
model with a vocabulary of one hundred and fifty thousand tokens, the head alone accounts for well
over a gigabyte of the roughly four and a half gigabytes read per generated token. Leaving it in
sixteen-bit precision while the rest of the model is packed to four bits means paying a quarter of
the per-token bandwidth budget for a single matrix. The question is simply whether the quality cost
of packing it is small enough to be worth the time saved, and that is an empirical question.
"""


if __name__ == "__main__":
    main()
