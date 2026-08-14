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
                         "bf16 head). Halves the cost whe