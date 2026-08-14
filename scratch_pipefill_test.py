"""#pipefill scratch validation — chunk-ordering / KV-position logic, NO network, NO model.

Drives the REAL engine_gen._send_prefill (+ _pipefill_chunk/_pipefill_arch_ok/_pipefill_conf
gates) against a simulated 3-stage pipeline built from plain asyncio queues. Frames go through
the REAL wire codecs end to end: wire._pack_tensor/_unpack_tensor for tensors, the length-
prefixed-JSON frame encoding (byte-identical twin of client/server _write_frame/_read_frame),
and wire._pack_ntensor/_unpack_ntensor for the #logits-diet intermediate replies. Each fake
stage enforces the shard contract that pipefill rests on: reset/pos-0 rebuilds the KV, every
append must land at cache_position == current KV length (in-order), per-frame compute is
sequential per stage (worker_net's sequential inbound loop twin).

Scenarios:
  1. gates      — every _pipefill_chunk gate (env off/size, short prompt, mm, mrope, TP,
                  single-stage, missing cap, hybrid/per-type/mrope3d/omni config sniffs)
  2. happy path — 1200-token prompt, chunk 512, 3 stages: per-stage chunk order + KV positions,
                  token-content reconstruction at stage0, final reply == last chunk's logits,
                  intermediate replies ride the ntensor argmax diet, stages OVERLAP in time,
                  pending maps fully scrubbed
  3. spec-style — final chunk carries nt_mode=argmax/nt_clip=0; the reply is the NT_TOKEN_IDS
                  manifest list exactly like _decode_spec consumes
  4. fail-fast  — stage 1 blows up on chunk 1: the burst fails in ~stage time (NOT GEN_TIMEOUT,
                  audit caveat 3), pending scrubbed, trailing stale chunks only error-frame
  5. recovery   — a fresh burst on the SAME (dirtied) chain succeeds: chunk 0 reset/pos-0 wipes
                  every stage's stale KV (the #kv-reset-on-seqstart contract)
  6. fallback   — a gated-off model takes the classic one-shot _send prefill unchanged

Run:  python scratch_pipefill_test.py     (prints PASS lines; non-zero exit on any failure)
"""
import asyncio
import contextlib
import json
import os
import sys
import tempfile
import time
import types

import torch

import engine_gen
import wire
from wire import (_pack_tensor, _unpack_tensor, _pack_ntensor, _unpack_ntensor,
                  NT_LOGITS, NT_HIDDEN, NT_TOKEN_IDS, NT_TOPK_VALS, NT_TOPK_IDX)

# --- real frame codec twin (client/server _write_frame / _read_frame, minus socket+NET) ---------


def encode_frame(hdr: dict, raw: bytes) -> bytes:
    hdr = {**hdr, "nbytes": len(raw)}
    hb = json.dumps(hdr).encode()
    return len(hb).to_bytes(4, "big") + hb + raw


def decode_frame(buf: bytes):
    hl = int.from_bytes(buf[:4], "big")
    hdr = json.loads(buf[4:4 + hl].decode())
    raw = buf[4 + hl:4 + hl + hdr["nbytes"]]
    assert 4 + hl + hdr["nbytes"] == len(buf), "frame length mismatch"
    return hdr, raw


# --- fake fleet ----------------------------------------------------------------------------------

CAPS = {}          # node_id -> frozenset of wire caps (registry.node_caps twin)


class Stage:
    """One pipeline stage: sequential frame loop (worker_net._data_inbound twin) + the shard
    contract (#kv-reset-on-seqstart; append at cache_position == kv_len). has_head replies to
    the controller queue honoring the nt argmax directive like shard_forward would."""

    def __init__(self, name, delay, has_head=False, fail_on=None):
        self.q: asyncio.Queue = asyncio.Queue()
        self.name, self.delay, self.has_head = name, delay, has_head
        self.fail_on = fail_on            # predicate(hdr) -> raise a synthetic compute error
        self.next = None                  # next Stage (mid) — head replies to ctrl_q instead
        self.kv_len = 0
        self.seen = []                    # (cache_position, q, reset, nt_mode)
        self.chunks = []                  # stage-0 token payloads (content reconstruction)
        self.events = []                  # (name, rid, 'start'|'end', perf_counter)
        self.violations = []              # KV append-order violations (stale-chunk exhaust)

    async def run(self, ctrl_q: asyncio.Queue):
        while True:
            buf = await self.q.get()
            if buf is None:
                return
            hdr, raw = decode_frame(buf)
            try:
                x = _unpack_tensor(hdr, raw)
                cs = int(hdr.get("cache_position", 0))
                reset = bool(hdr.get("reset", True))
                qn = int(x.shape[1])
                if self.fail_on is not None and self.fail_on(hdr):
                    raise RuntimeError(f"synthetic compute failure at {self.name}")
                if reset or cs == 0:      # #kv-reset-on-seqstart twin
                    self.kv_len = 0
                if cs != self.kv_len:     # real shard: mask/append mismatch -> RuntimeError
                    self.violations.append((self.name, hdr.get("req_id"), cs, self.kv_len))
                    raise RuntimeError(f"KV append out of order at {self.name}: "
                                       f"cache_position {cs} != kv_len {self.kv_len}")
                self.seen.append((cs, qn, reset, hdr.get("nt_mode")))
                if x.dtype == torch.int64:
                    self.chunks.append(x.clone())
                self.events.append((self.name, hdr["req_id"], "start", time.perf_counter()))
                await asyncio.sleep(self.delay)                 # simulated per-chunk compute
                self.kv_len += qn
                self.events.append((self.name, hdr["req_id"], "end", time.perf_counter()))
                if self.has_head:
                    if hdr.get("ntensor") and hdr.get("nt_mode") == "argmax":
                        tmeta, oraw = _pack_ntensor(
                            [(NT_TOKEN_IDS, torch.tensor([self.kv_len], dtype=torch.int64))])
                        await ctrl_q.put(encode_frame(
                            {"req_id": hdr["req_id"], "model_id": hdr["model_id"],
                             "kind": "ntensor", "tensors": tmeta}, oraw))
                    else:                 # marker logits: value == KV length after this chunk
                        m, r = _pack_tensor(torch.full((1, 1, 7), float(self.kv_len)))
                        await ctrl_q.put(encode_frame(
                            {"req_id": hdr["req_id"], "model_id": hdr["model_id"],
                             "kind": "logits", **m}, r))
                else:                     # forward the hidden block downstream (worker_net twin:
                    m, r = _pack_tensor(torch.zeros((1, qn, 4)))  # header keys re-propagated)
                    fh = {"req_id": hdr["req_id"], "model_id": hdr["model_id"], "kind": "hidden",
                          "cache_position": cs, "reset": reset,
                          "all_logits": hdr.get("all_logits", False), **m}
                    for k in ("ntensor", "nt_mode", "nt_clip", "nt_k"):
                        if hdr.get(k) is not None:
                            fh[k] = hdr[k]
                    await self.next.q.put(encode_frame(fh, r))
            except Exception as exc:      # stage error -> error frame to the controller
                await ctrl_q.put(encode_frame(
                    {"req_id": hdr.get("req_id"), "model_id": hdr.get("model_id"),
                     "kind": "error", "error": repr(exc)}, b""))


async def controller_rx(ctrl_q: asyncio.Queue, eng):
    """engine_lifecycle._on_data essentials: pop the rid, dispatch by kind."""
    while True:
        buf = await ctrl_q.get()
        if buf is None:
            return
        hdr, raw = decode_frame(buf)
        rid = hdr.get("req_id")
        fut = eng.pending.pop(rid, None)
        eng.pending_model.pop(rid, None)
        eng.pending_friendly.pop(rid, None)
        if fut is None or fut.done():
            continue                      # stale/aborted chunk exhaust — ignored, like _on_data
        if hdr.get("kind") == "error":
            fut.set_exception(RuntimeError(hdr.get("error", "stage error")))
        elif hdr.get("kind") == "ntensor":
            parts = _unpack_ntensor(hdr.get("tensors") or [], raw)
            by = dict(parts)
            fut.set_result(by[NT_LOGITS] if set(by) == {NT_LOGITS} else parts)
        else:
            fut.set_result(_unpack_tensor(hdr, raw))


class Eng(engine_gen.EngineGenMixin):
    def __init__(self):
        self.pending, self.pending_model, self.pending_friendly = {}, {}, {}
        self.req_counter = 0
        self.models = {}

    def next_req(self):
        self.req_counter += 1
        return self.req_counter

    def _stage0_id(self, model):
        return model.stage_node_ids[0] if model and model.stage_node_ids else None


def mk_model(stage0, nodes, target, **kw):
    return types.SimpleNamespace(
        friendly="m", target_id=target, stage_node_ids=list(nodes),
        tp_size=kw.get("tp_size", 1), stage0_writer=stage0, stage0_dial=("x", 1),
        last_send_ts=time.time(), fwd_progress_ts=0.0)


def build_chain(n_stages, delay=0.08, fail_on=None):
    stages = [Stage(f"s{i}", delay, has_head=(i == n_stages - 1),
                    fail_on=fail_on if i == 1 else None) for i in range(n_stages)]
    for a, b in zip(stages, stages[1:]):
        a.next = b
    return stages


def write_cfg(models_dir, target, cfg):
    d = os.path.join(models_dir, target.replace("/", "__"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)


FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


async def main():
    models_dir = tempfile.mkdtemp(prefix="pipefill_models_")
    # inject engine_gen's bound globals (state.bind twin) with the REAL wire codecs
    ctrl_q: asyncio.Queue = asyncio.Queue()

    async def _fake_write_frame(w, hdr, raw):
        buf = encode_frame(hdr, raw)
        await w.q.put(buf)
        return len(buf)

    engine_gen.__dict__.update(
        _pack_tensor=_pack_tensor, _unpack_tensor=_unpack_tensor,
        _pack_ntensor=_pack_ntensor, _unpack_ntensor=_unpack_ntensor,
        NT_LOGITS=NT_LOGITS, NT_HIDDEN=NT_HIDDEN, NT_TOKEN_IDS=NT_TOKEN_IDS,
        NT_TOPK_VALS=NT_TOPK_VALS, NT_TOPK_IDX=NT_TOPK_IDX,
        _write_frame=_fake_write_frame, net_account=lambda *a, **k: None,
        registry=types.SimpleNamespace(node_caps=lambda nid: CAPS.get(nid, frozenset())),
        GEN_TIMEOUT_S=8.0, STAGE0_STALE_S=5.0,
        time=time, asyncio=asyncio, contextlib=contextlib, os=os, json=json,
        MODELS_DIR=models_dir, _safe_name=lambda s: s.replace("/", "__"),
    )
    ALLCAPS = frozenset({"ntensor", "ntdiet", "pipefill"})
    CAPS.update({"n0": ALLCAPS, "n1": ALLCAPS, "n2": ALLCAPS,
                 "nocap": frozenset({"ntensor", "ntdiet"})})
    write_cfg(models_dir, "acme/plain", {"model_type": "llama"})
    write_cfg(models_dir, "acme/hybrid",
              {"model_type": "qwen3_next",
               "layer_types": ["linear_attention", "full_attention"]})
    write_cfg(models_dir, "acme/gemma4ish",
              {"model_type": "gemma4",
               "text_config": {"layer_types": ["sliding_attention", "full_attention"]}})
    write_cfg(models_dir, "acme/vl", {"model_type": "qwen2_5_vl"})
    write_cfg(models_dir, "acme/omni", {"model_type": "omni", "thinker_config": {"x": 1}})

    eng = Eng()
    os.environ["INFINITEMODEL_PIPEFILL"] = "512"
    prompt = torch.arange(1200, dtype=torch.long).reshape(1, 1200)

    # ---- 1. gates -------------------------------------------------------------------------------
    print("[1] gates")
    s = build_chain(3)
    m = mk_model(s[0], ["n0", "n1", "n2"], "acme/plain")
    check("eligible -> chunk 512", eng._pipefill_chunk(m, 1200, None, None) == 512)
    check("short prompt", eng._pipefill_chunk(m, 512, None, None) == 0)
    check("mm excluded", eng._pipefill_chunk(m, 1200, ("mm",), None) == 0)
    check("mrope excluded", eng._pipefill_chunk(m, 1200, None, [[0]]) == 0)
    check("TP excluded",
          eng._pipefill_chunk(mk_model(s[0], ["n0", "n1"], "acme/plain", tp_size=2),
                              1200, None, None) == 0)
    check("single stage excluded",
          eng._pipefill_chunk(mk_model(s[0], ["n0"], "acme/plain"), 1200, None, None) == 0)
    check("missing cap excluded",
          eng._pipefill_chunk(mk_model(s[0], ["n0", "nocap"], "acme/plain"),
                              1200, None, None) == 0)
    check("hybrid excluded",
          eng._pipefill_chunk(mk_model(s[0], ["n0", "n1"], "acme/hybrid"),
                              1200, None, None) == 0)
    check("per-type (text_config.layer_types) excluded",
          eng._pipefill_chunk(mk_model(s[0], ["n0", "n1"], "acme/gemma4ish"),
                              1200, None, None) == 0)
    check("mrope3d model excluded",
          eng._pipefill_chunk(mk_model(s[0], ["n0", "n1"], "acme/vl"), 1200, None, None) == 0)
    check("omni excluded",
          eng._pipefill_chunk(mk_model(s[0], ["n0", "n1"], "acme/omni"), 1200, None, None) == 0)
    check("missing config.json excluded (conservative)",
          eng._pipefill_chunk(mk_model(s[0], ["n0", "n1"], "acme/absent"),
                              1200, None, None) == 0)
    os.environ["INFINITEMODEL_PIPEFILL"] = "0"
    check("env opt-out", eng._pipefill_chunk(m, 1200, None, None) == 0)
    os.environ["INFINITEMODEL_PIPEFILL"] = "off"
    check("env 'off'", eng._pipefill_chunk(m, 1200, None, None) == 0)
    os.environ["INFINITEMODEL_PIPEFILL"] = "1"
    check("tiny chunk clamped to 256", engine_gen._pipefill_conf() == 256)
    os.environ["INFINITEMODEL_PIPEFILL"] = "garbage"
    check("garbage env -> default 2048", engine_gen._pipefill_conf() == 2048)
    os.environ["INFINITEMODEL_PIPEFILL"] = "512"

    # ---- 2. happy path --------------------------------------------------------------------------
    print("[2] happy path (3 stages, 1200 tokens, chunk 512)")
    stages = build_chain(3)
    model = mk_model(stages[0], ["n0", "n1", "n2"], "acme/plain")
    rx = asyncio.create_task(controller_rx(ctrl_q, eng))
    tasks = [asyncio.create_task(st.run(ctrl_q)) for st in stages]
    t0 = time.perf_counter()
    res = await asyncio.wait_for(eng._send_prefill(model, prompt), timeout=15)
    wall = time.perf_counter() - t0
    check("result is the LAST chunk's logits tensor",
          torch.is_tensor(res) and res.shape == (1, 1, 7) and float(res[0, -1, 0]) == 1200.0,
          f"got {type(res)}")
    exp = [(0, 512, True), (512, 512, False), (1024, 176, False)]
    for st in stages:
        check(f"{st.name} chunk order+positions", [t[:3] for t in st.seen] == exp, str(st.seen))
        check(f"{st.name} KV length complete", st.kv_len == 1200, str(st.kv_len))
        check(f"{st.name} no order violations", not st.violations, str(st.violations))
    check("stage0 token content reconstructs the prompt",
          torch.equal(torch.cat(stages[0].chunks, dim=1), prompt))
    check("intermediate chunks ride the argmax diet; final keeps the caller's mode",
          [t[3] for t in stages[0].seen] == ["argmax", "argmax", None], str(stages[0].seen))
    s0_end = {e[1]: e[3] for e in stages[0].events if e[2] == "end"}
    s1_start = {e[1]: e[3] for e in stages[1].events if e[2] == "start"}
    rids0 = sorted(s0_end)
    overlapped = any(s1_start.get(r) is not None and s1_start[r] < s0_end[r2]
                     for i, r in enumerate(rids0) for r2 in rids0[i + 1:])
    check("stages OVERLAP (stage1 starts chunk i before stage0 finishes a later chunk)",
          overlapped)
    check("pipelined makespan ~ (C+S-1) not C*S stage-times", wall < 0.60, f"{wall:.3f}s")
    check("pending maps scrubbed", not eng.pending and not eng.pending_model
          and not eng.pending_friendly)

    # ---- 3. spec-style final diet reply ---------------------------------------------------------
    print("[3] spec-style prefill (final chunk nt argmax)")
    for st in stages:
        st.seen.clear()
    res = await asyncio.wait_for(
        eng._send_prefill(model, prompt, nt_mode="argmax", nt_clip=0), timeout=15)
    ok = isinstance(res, list) and dict(res).get(NT_TOKEN_IDS) is not None
    check("reply is the NT_TOKEN_IDS manifest list (as _decode_spec consumes)", ok, repr(res)[:80])
    if ok:
        check("argmax id == head marker (final KV length)",
              int(dict(res)[NT_TOKEN_IDS].reshape(-1)[-1]) == 1200)
    check("final chunk carried nt argmax on the wire",
          [t[3] for t in stages[0].seen] == ["argmax", "argmax", "argmax"], str(stages[0].seen))

    # ---- 4. fail-fast on an intermediate chunk --------------------------------------------------
    print("[4] fail-fast (stage1 dies on chunk 1)")
    fstages = build_chain(3, fail_on=lambda hdr: int(hdr.get("cache_position", 0)) == 512)
    fmodel = mk_model(fstages[0], ["n0", "n1", "n2"], "acme/plain")
    ftasks = [asyncio.create_task(st.run(ctrl_q)) for st in fstages]
    t0 = time.perf_counter()
    try:
        await asyncio.wait_for(eng._send_prefill(fmodel, prompt), timeout=15)
        check("fail-fast raised", False, "no exception")
    except RuntimeError as exc:
        el = time.perf_counter() - t0
        check("fail-fast raised the stage error", "synthetic compute failure" in str(exc),
              str(exc))
        check("failed in ~stage time, NOT GEN_TIMEOUT (audit caveat 3)", el < 3.0, f"{el:.2f}s")
    await asyncio.sleep(0.6)              # let the stale trailing chunks drain as exhaust
    check("pending maps scrubbed after failure", not eng.pending and not eng.pending_model)
    check("stale trailing chunks only errored downstream (no crash, exhaust ignored)",
          all(v[0] in ("s1", "s2") for st in fstages for v in st.violations),
          str([st.violations for st in fstages]))

    # ---- 5. recovery burst on the SAME dirtied chain --------------------------------------------
    print("[5] recovery burst (reset/pos-0 wipes stale KV)")
    for st in fstages:
        st.fail_on = None
        st.violations.clear()
        st.seen.clear()
    res = await asyncio.wait_for(eng._send_prefill(fmodel, prompt), timeout=15)
    check("fresh burst succeeds after the abort",
          torch.is_tensor(res) and float(res[0, -1, 0]) == 1200.0)
    for st in fstages:
        check(f"{st.name} recovered KV == 1200, no violations",
              st.kv_len == 1200 and not st.violations, f"kv={st.kv_len} {st.violations}")

    # ---- 6. one-shot fallback for a gated-off placement -----------------------------------------
    print("[6] one-shot fallback (single-stage model)")
    one = build_chain(1, delay=0.02)
    omodel = mk_model(one[0], ["n0"], "acme/plain")
    otask = asyncio.create_task(one[0].run(ctrl_q))
    res = await asyncio.wait_for(eng._send_prefill(omodel, prompt), timeout=15)
    check("fallback result correct", torch.is_tensor(res) and float(res[0, -1, 0]) == 1200.0)
    check("fallback sent ONE frame (reset, pos 0, whole prompt)",
          one[0].seen == [(0, 1200, True, None)], str(one[0].seen))

    for t in tasks + ftasks + [otask, rx]:
        t.cancel()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
