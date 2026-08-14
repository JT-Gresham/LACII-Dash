"""#prefix-kv scratch validation — cross-request prefix reuse (crop + suffix prefill), NO
network, NO model. Extends scratch_pipefill_test.py's simulated 3-stage chain.

Drives the REAL engine_gen._prefill_reuse / _prefix_kv_ok / _prefix_kv_min (+ the real
_send_prefill(base=...) suffix burst, _send, _crop) against fake stages that enforce the shard
contract AND track per-position KV CONTENT: stage 0 records actual token ids; it encodes each
token id into channel 0 of the forwarded hidden block so every downstream stage records the
same content-by-position — "identical stage KV state" is checked as exact content equality
against a control chain that full-prefilled the same prompt.

Scenarios:
  1. knobs/gates — INFINITEMODEL_PREFIX_KV off / PREFIX_MIN parsing; every _prefix_kv_ok gate
                   (TP, kv_offload, kv_quant, missing 'pipefill' cap, hybrid arch); per-request
                   misses (no record, below-min LCP, mm, mrope, dead stage0)
  2. agent turn  — full prefill A -> record -> prompt B = A + new turn: crop(len A) + suffix-
                   only prefill, NO reset frame, stage KV content == a control full prefill of B
  3. LCP edges   — identical retry (L=len-1, 1-token suffix), mid-prompt divergence (crop
                   truncates the stale tail), shorter-prompt retry (pure crop-down), below-min
                   divergence -> full prefill, empty record -> full prefill
  4. decode loop — REAL _decode_plain end-to-end: record == prompt + SENT tokens only (the
                   audit off-by-one caveat: final/length token never sent), then the follow-up
                   turn resumes off that record; decode-send failure NULLS the record
  5. invalidate  — reset probe (qcheck twin) nulls; _crop nulls; suffix-burst stage failure
                   nulls + the NEXT full prefill recovers the dirty chain; spec-style verify
                   append + crop + re-publish keeps record == stage KV

Run:  python scratch_prefix_kv_test.py     (prints PASS lines; non-zero exit on any failure)
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
    contract (#kv-reset-on-seqstart; append at cache_position == kv length; crop truncates).
    self.kv is the per-position CONTENT list (token ids at every stage — stage 0 sees real ids
    and encodes them into hidden channel 0 for the downstream stages)."""

    def __init__(self, name, delay, has_head=False, fail_on=None):
        self.q: asyncio.Queue = asyncio.Queue()
        self.name, self.delay, self.has_head = name, delay, has_head
        self.fail_on = fail_on            # predicate(hdr) -> raise a synthetic compute error
        self.next = None                  # next Stage (mid) — head replies to ctrl_q instead
        self.kv = []                      # per-position content (== token ids)
        self.frames = []                  # (cache_position, q, reset, nt_mode) of ids/hidden frames
        self.crops = []                   # crop cache_positions seen
        self.violations = []              # KV append-order violations

    async def run(self, ctrl_q: asyncio.Queue):
        while True:
            buf = await self.q.get()
            if buf is None:
                return
            hdr, raw = decode_frame(buf)
            if hdr.get("kind") == "mm":           # worker stages mm embeds; never forwarded
                continue
            if hdr.get("kind") == "crop":         # worker_net crop twin: truncate + propagate
                length = int(hdr.get("cache_position", 0))
                self.crops.append(length)
                del self.kv[length:]
                if self.next is not None:
                    await self.next.q.put(buf)
                continue
            try:
                x = _unpack_tensor(hdr, raw)
                cs = int(hdr.get("cache_position", 0))
                reset = bool(hdr.get("reset", True))
                qn = int(x.shape[1])
                if self.fail_on is not None and self.fail_on(hdr):
                    raise RuntimeError(f"synthetic compute failure at {self.name}")
                if reset or cs == 0:      # #kv-reset-on-seqstart twin
                    self.kv = []
                if cs != len(self.kv):    # real shard: mask/append mismatch -> RuntimeError
                    self.violations.append((self.name, hdr.get("req_id"), cs, len(self.kv)))
                    raise RuntimeError(f"KV append out of order at {self.name}: "
                                       f"cache_position {cs} != kv_len {len(self.kv)}")
                content = (x[0, :].tolist() if x.dtype == torch.int64
                           else [int(v) for v in x[0, :, 0].tolist()])
                self.frames.append((cs, qn, reset, hdr.get("nt_mode")))
                await asyncio.sleep(self.delay)                 # simulated per-chunk compute
                self.kv.extend(content)
                if self.has_head:
                    if hdr.get("ntensor") and hdr.get("nt_mode") == "argmax":
                        tmeta, oraw = _pack_ntensor(
                            [(NT_TOKEN_IDS,
                              torch.tensor([len(self.kv)], dtype=torch.int64))])
                        await ctrl_q.put(encode_frame(
                            {"req_id": hdr["req_id"], "model_id": hdr["model_id"],
                             "kind": "ntensor", "tensors": tmeta}, oraw))
                    else:                 # marker logits: value == KV length after this frame
                        m, r = _pack_tensor(torch.full((1, 1, 7), float(len(self.kv))))
                        await ctrl_q.put(encode_frame(
                            {"req_id": hdr["req_id"], "model_id": hdr["model_id"],
                             "kind": "logits", **m}, r))
                else:                     # forward hidden with the CONTENT in channel 0
                    h = torch.tensor(content, dtype=torch.float32).reshape(1, qn, 1)
                    m, r = _pack_tensor(h.repeat(1, 1, 4))
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
            continue                      # stale/aborted exhaust — ignored, like _on_data
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
    m = types.SimpleNamespace(
        friendly=kw.get("friendly", "m"), target_id=target, stage_node_ids=list(nodes),
        tp_size=kw.get("tp_size", 1), stage0_writer=stage0, stage0_dial=("x", 1),
        last_send_ts=time.time(), fwd_progress_ts=0.0, kv_ids=None, kv_pos=0,
        eos_ids=kw.get("eos_ids", {-999}), tokenizer=kw.get("tokenizer", list(range(7))))
    for k in ("kv_offload", "kv_quant"):
        if k in kw:
            setattr(m, k, kw[k])
    return m


def build_chain(n_stages, delay=0.02, fail_on=None, fail_stage=1):
    stages = [Stage(f"s{i}", delay, has_head=(i == n_stages - 1),
                    fail_on=fail_on if i == fail_stage else None) for i in range(n_stages)]
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


def ids(seq):
    return list(seq)


async def full_prefill_control(eng, target, prompt_list, n_stages=3):
    """Full prefill of prompt_list on a FRESH control chain; returns its stages' kv lists."""
    st = build_chain(n_stages)
    m = mk_model(st[0], ["n0", "n1", "n2"][:n_stages], target)
    ts = [asyncio.create_task(s.run(eng._ctrl_q)) for s in st]
    await asyncio.wait_for(
        eng._send_prefill(m, torch.tensor([prompt_list], dtype=torch.long)), timeout=15)
    for t in ts:
        t.cancel()
    return [s.kv for s in st]


async def main():
    models_dir = tempfile.mkdtemp(prefix="prefixkv_models_")
    ctrl_q: asyncio.Queue = asyncio.Queue()
    activity = []

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
        log_activity=lambda msg: activity.append(msg),
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

    eng = Eng()
    eng._ctrl_q = ctrl_q
    os.environ["INFINITEMODEL_PIPEFILL"] = "512"
    os.environ["INFINITEMODEL_PREFIX_MIN"] = "256"
    os.environ.pop("INFINITEMODEL_PREFIX_KV", None)
    rx = asyncio.create_task(controller_rx(ctrl_q, eng))

    A = list(range(10_000, 11_200))            # 1200-token "turn 1" prompt (unique ids)

    # ---- 1. knobs + gates -------------------------------------------------------------------
    print("[1] knobs + gates")
    check("PREFIX_MIN=256 honored", engine_gen._prefix_kv_min() == 256)
    os.environ["INFINITEMODEL_PREFIX_MIN"] = "garbage"
    check("garbage PREFIX_MIN -> default 1024", engine_gen._prefix_kv_min() == 1024)
    os.environ["INFINITEMODEL_PREFIX_MIN"] = "3"
    check("PREFIX_MIN clamped >= 16", engine_gen._prefix_kv_min() == 16)
    os.environ["INFINITEMODEL_PREFIX_KV"] = "off"
    check("INFINITEMODEL_PREFIX_KV=off -> 0", engine_gen._prefix_kv_min() == 0)
    os.environ["INFINITEMODEL_PREFIX_KV"] = "0"
    check("INFINITEMODEL_PREFIX_KV=0 -> 0", engine_gen._prefix_kv_min() == 0)
    os.environ.pop("INFINITEMODEL_PREFIX_KV", None)
    os.environ["INFINITEMODEL_PREFIX_MIN"] = "256"

    s = build_chain(3)
    ok_m = mk_model(s[0], ["n0", "n1", "n2"], "acme/plain")
    check("gate: eligible model passes", eng._prefix_kv_ok(ok_m))
    check("gate: TP excluded",
          not eng._prefix_kv_ok(mk_model(s[0], ["n0", "n1"], "acme/plain", tp_size=2)))
    check("gate: kv_offload excluded",
          not eng._prefix_kv_ok(mk_model(s[0], ["n0", "n1"], "acme/plain", kv_offload=True)))
    check("gate: kv_quant excluded",
          not eng._prefix_kv_ok(mk_model(s[0], ["n0", "n1"], "acme/plain", kv_quant="turbo3")))
    check("gate: missing pipefill cap excluded",
          not eng._prefix_kv_ok(mk_model(s[0], ["n0", "nocap"], "acme/plain")))
    check("gate: hybrid arch excluded",
          not eng._prefix_kv_ok(mk_model(s[0], ["n0", "n1"], "acme/hybrid")))
    check("gate: no stage nodes excluded",
          not eng._prefix_kv_ok(mk_model(s[0], [], "acme/plain")))

    # per-request misses: each of these must land as a FULL reset prefill
    async def is_full_prefill(model, prompt_list, stages, **kw):
        for st in stages:
            st.frames.clear()
        await asyncio.wait_for(eng._prefill_reuse(model, prompt_list, **kw), timeout=15)
        f = stages[0].frames
        return bool(f) and f[0][0] == 0 and f[0][2] is True

    stages = build_chain(3)
    model = mk_model(stages[0], ["n0", "n1", "n2"], "acme/plain")
    tasks = [asyncio.create_task(st.run(ctrl_q)) for st in stages]
    check("miss: no record -> full prefill", await is_full_prefill(model, A, stages))
    model.kv_ids = ids(A)
    check("miss: mm excluded -> full prefill",
          await is_full_prefill(model, A, stages, mm=([1], torch.zeros((1, 4)))))
    model.kv_ids = ids(A)
    check("miss: mrope excluded -> full prefill",
          await is_full_prefill(model, A, stages, position_ids=[[0], [0], [0]]))
    model.kv_ids = ids(A)[:100]           # record shorter than min
    check("miss: record below PREFIX_MIN -> full prefill",
          await is_full_prefill(model, A, stages))

    # ---- 2. the agent turn (the money path) ---------------------------------------------------
    print("[2] agent turn: full A -> record -> B = A + new turn resumes")
    for st in stages:
        st.frames.clear()
        st.crops.clear()
    r = await asyncio.wait_for(eng._prefill_reuse(model, A), timeout=15)
    check("turn-1 full prefill reply marker == 1200",
          torch.is_tensor(r) and float(r[0, -1, 0]) == 1200.0)
    check("prefill nulled the record (pre-publish)", model.kv_ids is None)
    model.kv_ids = ids(A)                  # what _decode_plain publishes post-prefill
    B = A + list(range(20_000, 20_400))    # + 400-token turn 2
    for st in stages:
        st.frames.clear()
        st.crops.clear()
    activity.clear()
    r = await asyncio.wait_for(eng._prefill_reuse(model, B), timeout=15)
    check("resume reply marker == 1600", torch.is_tensor(r) and float(r[0, -1, 0]) == 1600.0)
    check("crop(1200) reached EVERY stage", all(st.crops == [1200] for st in stages),
          str([st.crops for st in stages]))
    check("NO reset frame on resume (suffix-only, reset=False)",
          all(not f[2] for st in stages for f in st.frames),
          str(stages[0].frames))
    check("suffix starts at cache_position 1200",
          stages[0].frames and stages[0].frames[0][0] == 1200, str(stages[0].frames[:2]))
    check("suffix total == 400 tokens (not 1600)",
          sum(f[1] for f in stages[0].frames) == 400, str(stages[0].frames))
    control = await full_prefill_control(eng, "acme/plain", B)
    check("stage KV state IDENTICAL to a control full prefill of B (all 3 stages)",
          all(st.kv == c for st, c in zip(stages, control)),
          f"lens {[len(st.kv) for st in stages]} vs {[len(c) for c in control]}")
    check("HIT logged with tokens saved",
          any("#prefix-kv HIT" in m and "1200" in m for m in activity), str(activity))
    check("pending maps scrubbed", not eng.pending and not eng.pending_model
          and not eng.pending_friendly)

    # ---- 3. LCP edge cases ----------------------------------------------------------------------
    print("[3] LCP edges")
    # (a) identical retry: L = len-1, 1-token suffix
    model.kv_ids = ids(B)
    for st in stages:
        st.frames.clear()
        st.crops.clear()
    r = await asyncio.wait_for(eng._prefill_reuse(model, B), timeout=15)
    check("retry: crop to len-1 then 1-token suffix",
          all(st.crops == [1599] for st in stages)
          and stages[0].frames == [(1599, 1, False, None)],
          f"crops {[st.crops for st in stages]} frames {stages[0].frames}")
    check("retry: reply marker == 1600", torch.is_tensor(r) and float(r[0, -1, 0]) == 1600.0)
    check("retry: stage KV content unchanged (== B)",
          all(st.kv == B for st in stages))
    # (b) mid-prompt divergence: crop truncates the stale tail, suffix replaces it
    model.kv_ids = ids(B)
    C = B[:700] + list(range(30_000, 30_700))     # diverges at position 700, len 1400
    for st in stages:
        st.frames.clear()
        st.crops.clear()
    r = await asyncio.wait_for(eng._prefill_reuse(model, C), timeout=15)
    check("divergence: crop(700) on every stage", all(st.crops == [700] for st in stages))
    check("divergence: suffix = 700 tokens at 700, no reset",
          stages[0].frames and stages[0].frames[0][0] == 700
          and sum(f[1] for f in stages[0].frames) == 700
          and all(not f[2] for f in stages[0].frames), str(stages[0].frames))
    control = await full_prefill_control(eng, "acme/plain", C)
    check("divergence: stage KV == control full prefill of C (stale tail gone)",
          all(st.kv == c for st, c in zip(stages, control)))
    # (c) shorter-prompt retry fully inside the cache: crop-down + 1-token suffix
    model.kv_ids = ids(C)
    D = C[:900]
    for st in stages:
        st.frames.clear()
        st.crops.clear()
    r = await asyncio.wait_for(eng._prefill_reuse(model, D), timeout=15)
    check("shorter retry: crop(899) + 1-token suffix -> KV == D",
          all(st.crops == [899] for st in stages)
          and stages[0].frames == [(899, 1, False, None)]
          and all(st.kv == D for st in stages),
          f"crops {[st.crops for st in stages]} kvlen {[len(st.kv) for st in stages]}")
    # (d) divergence below min -> full prefill
    model.kv_ids = ids(D)
    E = D[:100] + list(range(40_000, 41_000))
    for st in stages:
        st.frames.clear()
        st.crops.clear()
    await asyncio.wait_for(eng._prefill_reuse(model, E), timeout=15)
    check("below-min divergence: full reset prefill, no crop",
          stages[0].frames[0][2] is True and all(not st.crops for st in stages))
    check("below-min: stage KV == E", all(st.kv == E for st in stages))
    # (e) env opt-out with a perfect record -> full prefill
    model.kv_ids = ids(E)
    os.environ["INFINITEMODEL_PREFIX_KV"] = "off"
    for st in stages:
        st.frames.clear()
        st.crops.clear()
    await asyncio.wait_for(eng._prefill_reuse(model, E + [1, 2, 3] * 200), timeout=15)
    check("opt-out: full reset prefill despite perfect prefix",
          stages[0].frames[0][2] is True and all(not st.crops for st in stages))
    os.environ.pop("INFINITEMODEL_PREFIX_KV", None)

    # ---- 4. REAL _decode_plain: record bookkeeping + the follow-up turn ------------------------
    print("[4] _decode_plain end-to-end record")
    eng.models["m"] = model               # _decode_plain liveness check
    model.kv_ids = None
    for st in stages:
        st.frames.clear()
        st.crops.clear()
    out = []
    async for item in eng._decode_plain(model, A, 3, 0.0, 1.0):
        out.append(item)
    toks = [t for t, _ in out if t is not None]
    check("decode emitted 3 tokens + length stop",
          len(toks) == 3 and out[-1] == (None, "length"), str(out))
    check("record == prompt + SENT tokens only (audit off-by-one: last emitted never sent)",
          model.kv_ids == A + toks[:2],
          f"rec_len {len(model.kv_ids or [])} vs {len(A) + 2}")
    check("stage KV matches the record exactly", all(st.kv == model.kv_ids for st in stages))
    # the follow-up agent turn resumes off prompt+answer
    F = A + toks + list(range(50_000, 50_400))
    for st in stages:
        st.frames.clear()
        st.crops.clear()
    r = await asyncio.wait_for(eng._prefill_reuse(model, F), timeout=15)
    check("follow-up turn resumed at LCP=1202 (crop + suffix only)",
          all(st.crops == [1202] for st in stages)
          and stages[0].frames and stages[0].frames[0][0] == 1202
          and all(not f[2] for f in stages[0].frames), str(stages[0].frames[:2]))
    control = await full_prefill_control(eng, "acme/plain", F)
    check("follow-up stage KV == control full prefill of F",
          all(st.kv == c for st, c in zip(stages, control)))
    # decode-send failure nulls the record
    model.kv_ids = None
    fstages = build_chain(3, fail_on=lambda hdr: (hdr.get("shape") or [0, 0])[1] == 1)
    fmodel = mk_model(fstages[0], ["n0", "n1", "n2"], "acme/plain", friendly="m")
    eng.models["m"] = fmodel
    ftasks = [asyncio.create_task(st.run(ctrl_q)) for st in fstages]
    try:
        async for item in eng._decode_plain(fmodel, A, 3, 0.0, 1.0):
            pass
        check("decode-send failure raised", False, "no exception")
    except RuntimeError:
        check("decode-send failure raised", True)
    check("decode-send failure NULLED the record", fmodel.kv_ids is None)
    # mm (vision/audio) decode publishes NO record: the shard KV rows at spliced positions
    # came from IMAGE embeds, not the placeholder token ids — a prompt_ids record would let a
    # later text-only prompt whose LCP crosses a splice reuse image-embed KV as text KV
    # (record-contract violation). The read gate already requires mm is None, so publishing
    # on mm requests buys nothing; _decode_plain must leave kv_ids None.
    eng.models["m"] = model
    model.kv_ids = None
    for st in stages:
        st.frames.clear()
        st.crops.clear()
    out = []
    async for item in eng._decode_plain(model, A, 3, 0.0, 1.0,
                                        mm=([1], torch.zeros((1, 4)))):
        out.append(item)
    mtoks = [t for t, _ in out if t is not None]
    check("mm decode emitted 3 tokens + length stop",
          len(mtoks) == 3 and out[-1] == (None, "length"), str(out))
    check("mm decode published NO record (image-embed KV never id-addressable)",
          model.kv_ids is None, f"kv_ids={type(model.kv_ids)}")

    # ---- 5. invalidation + recovery -------------------------------------------------------------
    print("[5] invalidation")
    eng.models["m"] = model
    model.kv_ids = ids(F)
    # reset probe (routes_shards/routes_diag qcheck twin) nulls centrally in _send
    await asyncio.wait_for(
        eng._send(model, torch.tensor([F[:8]], dtype=torch.long), 0, True), timeout=15)
    check("reset probe (qcheck twin) nulls the record via _send", model.kv_ids is None)
    model.kv_ids = ids(F)
    await eng._crop(model, 10)
    check("_crop nulls the record", model.kv_ids is None)
    for st in stages:                      # undo the probe damage for the failure scenario
        st.frames.clear()
        st.crops.clear()
    await asyncio.wait_for(eng._prefill_reuse(model, F), timeout=15)
    model.kv_ids = ids(F)
    # suffix-burst failure: stage 1 dies on the suffix chunk -> record nulled, next full recovers
    stages[1].fail_on = lambda hdr: int(hdr.get("cache_position", 0)) >= 1200 \
        and not hdr.get("reset", True)
    G = F + list(range(60_000, 61_200))    # 1200-token suffix -> chunked burst (cstep 512)
    try:
        await asyncio.wait_for(eng._prefill_reuse(model, G), timeout=15)
        check("suffix-burst failure raised", False, "no exception")
    except RuntimeError as exc:
        check("suffix-burst failure raised the stage error",
              "synthetic compute failure" in str(exc), str(exc))
    check("suffix-burst failure NULLED the record", model.kv_ids is None)
    await asyncio.sleep(0.3)               # drain stale burst exhaust
    stages[1].fail_on = None
    for st in stages:
        st.frames.clear()
        st.crops.clear()
        st.violations.clear()
    r = await asyncio.wait_for(eng._prefill_reuse(model, G), timeout=15)
    check("recovery: full reset prefill rebuilds the dirty chain (no violations)",
          stages[0].frames[0][2] is True and all(st.kv == G for st in stages)
          and all(not st.violations for st in stages))
    check("pending maps scrubbed after failure+recovery",
          not eng.pending and not eng.pending_model and not eng.pending_friendly)
    # spec-style round: verify append (multi-token reset=False) + crop + re-publish
    model.kv_ids = ids(G)
    cur = len(G)
    pending_tok, drafts = 70_001, [70_002, 70_003, 70_004]
    await asyncio.wait_for(
        eng._send(model, torch.tensor([[pending_tok] + drafts], dtype=torch.long),
                  cur, False, all_logits=True), timeout=15)
    m_acc = 2                              # pretend 2 drafts matched
    await eng._crop(model, cur + 1 + m_acc)
    _kv = ids(G) + [pending_tok] + drafts[:m_acc]
    model.kv_ids = _kv                     # what _decode_spec's round loop does
    # _crop is fire-and-forget; the production guarantee is IN-ORDER delivery — the next frame
    # on the same connection sees the cropped cache. Prove it the way the real decode does: a
    # follow-up token append at exactly the post-crop position (any stage that missed the crop
    # trips its append-position violation and errors instead).
    await asyncio.wait_for(
        eng._send(model, torch.tensor([[70_005]], dtype=torch.long),
                  cur + 1 + m_acc, False), timeout=15)
    _kv.append(70_005)
    check("spec round: record == stage KV after verify+crop (in-order proof token appended)",
          all(st.kv == _kv for st in stages) and all(not st.violations for st in stages),
          f"kvlen {[len(st.kv) for st in stages]} vs {len(_kv)} "
          f"violations {[st.violations for st in stages]}")

    for t in tasks + ftasks + [rx]:
        t.cancel()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
