"""#kv-slots scratch validation — per-request KV slots + interleaved decode. NO network, NO model.

Drives the REAL engine_gen paths (_SlotLease, _send, _send_prefill, _prefill_reuse, _crop,
_kv_rec_* helpers), the REAL shard_forward.ShardForwardMixin.forward (slot-scoped orphan
cancel + per-slot cache binding, threads) and the REAL engine_load gates (_kvslots_clamp,
_kvslots_cap_check) against a simulated 3-stage pipeline built from plain asyncio queues —
the scratch_pipefill_test.py pattern, with each fake stage keeping a SLOT-KEYED KV content
dict (slot -> per-position ids) enforcing the shard contract per slot: reset/pos-0 rebuilds
THE SLOT's stream, every append lands at cache_position == len(kv[slot]), and per-frame
compute is sequential per stage (worker_net's sequential inbound loop twin).

Scenarios (the task's binding battery):
  a. C=1 byte-path-identical — a prefill + decodes + crop on a kv_slots-absent model produce
     frames BYTE-IDENTICAL to golden pre-#kv-slots encodings (no 'slot' key anywhere), and
     _SlotLease at C=1 is exactly model.lock (serializes, same object).
  b. C=3 concurrent gens — three interleaved generations through one 3-stage chain, per-slot
     KV content isolation proven at every stage (sim stages track (slot, pos, ids)).
  c. crop on slot 1 leaves slots 0/2 intact (stage-side slot-keyed crop + propagation).
  d. superseded forward on slot 2 doesn't touch slot 0 — REAL ShardForwardMixin.forward in
     threads: a different-slot waiter never cancels the orphan (fails fast, orphan finishes);
     a same-slot waiter cancels it (_ForwardSuperseded); slot 0's cache object is untouched.
  e. #prefix-kv per-slot records — longest-LCP slot selection at admission, record isolation,
     and a REAL crop+suffix resume on slot 1 through the sim chain.
  f. an old-worker (no 'kvslots' cap) chain REFUSES C>1 at load (engine_load._kvslots_cap_check)
     + the load-time hard gates (tp / kv_quant / kv_offload / hybrid / missing config -> C=1).
  g. slot pool mechanics — exhaustion queues + logs once, watchdog-style token-swap reclaim
     never double-frees.

Run:  python scratch_kv_slots_test.py    (prints PASS lines; non-zero exit on any failure)
"""
import asyncio
import contextlib
import json
import os
import sys
import tempfile
import threading
import time
import types

import torch

import engine_gen
import engine_lifecycle
import engine_load
import shard_forward
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
ACTIVITY = []      # log_activity capture


class Stage:
    """One pipeline stage with SLOT-KEYED KV content: kv[slot] = list of per-position ids
    (stage 0 sees real token ids; mid stages track placeholder positions). Enforces the shard
    contract PER SLOT (#kv-reset-on-seqstart; append at cache_position == len(kv[slot]));
    sequential frame loop (worker_net._data_inbound twin); crop frames are slot-scoped and
    propagated downstream verbatim; the head honors the nt argmax directive."""

    def __init__(self, name, delay, has_head=False):
        self.q: asyncio.Queue = asyncio.Queue()
        self.name, self.delay, self.has_head = name, delay, has_head
        self.next = None
        self.kv = {}                      # slot -> per-position content list
        self.seen = []                    # (slot, cache_position, q, reset)
        self.chunks = {}                  # slot -> stage-0 int64 payloads
        self.rawlog = []                  # exact frame bytes as received (byte-identity check)
        self.violations = []

    async def run(self, ctrl_q: asyncio.Queue):
        while True:
            buf = await self.q.get()
            if buf is None:
                return
            hdr, raw = decode_frame(buf)
            self.rawlog.append(buf)
            slot = int(hdr.get("slot", 0) or 0)
            if hdr.get("kind") == "crop":                      # slot-scoped KV rollback
                kvs = self.kv.setdefault(slot, [])
                del kvs[int(hdr.get("cache_position", 0)):]
                if self.next is not None:                      # propagate verbatim downstream
                    await self.next.q.put(buf)
                continue
            try:
                x = _unpack_tensor(hdr, raw)
                cs = int(hdr.get("cache_position", 0))
                reset = bool(hdr.get("reset", True))
                qn = int(x.shape[1])
                if reset or cs == 0:      # #kv-reset-on-seqstart twin — THE SLOT's stream only
                    self.kv[slot] = []
                kvs = self.kv.setdefault(slot, [])
                if cs != len(kvs):        # real shard: mask/append mismatch -> RuntimeError
                    self.violations.append((self.name, slot, hdr.get("req_id"), cs, len(kvs)))
                    raise RuntimeError(f"KV append out of order at {self.name} slot {slot}: "
                                       f"cache_position {cs} != kv_len {len(kvs)}")
                self.seen.append((slot, cs, qn, reset))
                if x.dtype == torch.int64:                     # stage 0: real ids -> content
                    self.chunks.setdefault(slot, []).append(x.clone())
                    kvs.extend(int(v) for v in x.reshape(-1).tolist())
                else:                                          # mid stage: positions only
                    kvs.extend([None] * qn)
                await asyncio.sleep(self.delay)                # simulated per-frame compute
                if self.has_head:
                    if hdr.get("ntensor") and hdr.get("nt_mode") == "argmax":
                        tmeta, oraw = _pack_ntensor(
                            [(NT_TOKEN_IDS, torch.tensor([len(kvs)], dtype=torch.int64))])
                        await ctrl_q.put(encode_frame(
                            {"req_id": hdr["req_id"], "model_id": hdr["model_id"],
                             "kind": "ntensor", "tensors": tmeta}, oraw))
                    else:                 # marker logits: value == THIS SLOT's KV length
                        m, r = _pack_tensor(torch.full((1, 1, 7), float(len(kvs))))
                        await ctrl_q.put(encode_frame(
                            {"req_id": hdr["req_id"], "model_id": hdr["model_id"],
                             "kind": "logits", **m}, r))
                else:                     # forward the hidden block downstream (worker_net twin)
                    m, r = _pack_tensor(torch.zeros((1, qn, 4)))
                    fh = {"req_id": hdr["req_id"], "model_id": hdr["model_id"], "kind": "hidden",
                          "cache_position": cs, "reset": reset,
                          "all_logits": hdr.get("all_logits", False), **m}
                    if slot:              # #kv-slots hop-by-hop propagation twin
                        fh["slot"] = slot
                    for k in ("ntensor", "nt_mode", "nt_clip", "nt_k"):
                        if hdr.get(k) is not None:
                            fh[k] = hdr[k]
                    await self.next.q.put(encode_frame(fh, r))
            except Exception as exc:
                await ctrl_q.put(encode_frame(
                    {"req_id": hdr.get("req_id"), "model_id": hdr.get("model_id"),
                     "kind": "error", "error": repr(exc)}, b""))


async def controller_rx(ctrl_q: asyncio.Queue, eng):
    """engine_lifecycle._on_data essentials: pop the rid (incl. pending_slot), dispatch."""
    while True:
        buf = await ctrl_q.get()
        if buf is None:
            return
        hdr, raw = decode_frame(buf)
        rid = hdr.get("req_id")
        fut = eng.pending.pop(rid, None)
        eng.pending_model.pop(rid, None)
        eng.pending_friendly.pop(rid, None)
        getattr(eng, "pending_slot", {}).pop(rid, None)
        if fut is None or fut.done():
            continue
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
        self.pending_slot = {}
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
        last_send_ts=time.time(), fwd_progress_ts=0.0,
        kv_offload=False, kv_quant="none", kv_ids=None,
        kv_slots=kw.get("kv_slots", 1), lock=asyncio.Lock())
    if m.kv_slots > 1:
        engine_lifecycle.EngineLifecycleMixin._init_slot_pool(None, m)
    return m


def build_chain(n_stages, delay=0.03):
    stages = [Stage(f"s{i}", delay, has_head=(i == n_stages - 1)) for i in range(n_stages)]
    for a, b in zip(stages, stages[1:]):
        a.next = b
    return stages


def write_cfg(models_dir, target, cfg):
    d = os.path.join(models_dir, target.replace("/", "__"))
    os.makedirs(d, exist_ok=True)
    with open(os.path.join(d, "config.json"), "w", encoding="utf-8") as fh:
        json.dump(cfg, fh)
    return d


FAILURES = []


def check(name, cond, detail=""):
    if cond:
        print(f"  PASS  {name}")
    else:
        print(f"  FAIL  {name}  {detail}")
        FAILURES.append(name)


async def main():
    models_dir = tempfile.mkdtemp(prefix="kvslots_models_")
    ctrl_q: asyncio.Queue = asyncio.Queue()

    async def _fake_write_frame(w, hdr, raw):
        buf = encode_frame(hdr, raw)
        await w.q.put(buf)
        return len(buf)

    def _log(msg):
        ACTIVITY.append(str(msg))

    engine_gen.__dict__.update(
        _pack_tensor=_pack_tensor, _unpack_tensor=_unpack_tensor,
        _pack_ntensor=_pack_ntensor, _unpack_ntensor=_unpack_ntensor,
        NT_LOGITS=NT_LOGITS, NT_HIDDEN=NT_HIDDEN, NT_TOKEN_IDS=NT_TOKEN_IDS,
        NT_TOPK_VALS=NT_TOPK_VALS, NT_TOPK_IDX=NT_TOPK_IDX,
        _write_frame=_fake_write_frame, net_account=lambda *a, **k: None,
        registry=types.SimpleNamespace(node_caps=lambda nid: CAPS.get(nid, frozenset())),
        GEN_TIMEOUT_S=8.0, STAGE0_STALE_S=5.0, log_activity=_log,
        time=time, asyncio=asyncio, contextlib=contextlib, os=os, json=json,
        MODELS_DIR=models_dir, _safe_name=lambda s: s.replace("/", "__"),
    )
    engine_lifecycle.__dict__.update(asyncio=asyncio, contextlib=contextlib, time=time)
    engine_load.__dict__.update(
        registry=types.SimpleNamespace(node_caps=lambda nid: CAPS.get(nid, frozenset())),
        os=os, json=json, contextlib=contextlib, log_activity=_log)
    shard_forward.__dict__.update(threading=threading)
    ALLCAPS = frozenset({"ntensor", "ntdiet", "pipefill", "kvslots"})
    CAPS.update({"n0": ALLCAPS, "n1": ALLCAPS, "n2": ALLCAPS,
                 "oldworker": frozenset({"ntensor", "ntdiet", "pipefill"})})
    plain_dir = write_cfg(models_dir, "acme/plain", {"model_type": "llama"})
    hybrid_dir = write_cfg(models_dir, "acme/hybrid",
                           {"model_type": "qwen3_next",
                            "layer_types": ["linear_attention", "full_attention"]})
    os.environ["INFINITEMODEL_PIPEFILL"] = "512"
    os.environ["INFINITEMODEL_PREFIX_MIN"] = "16"
    os.environ.pop("INFINITEMODEL_PREFIX_KV", None)

    eng = Eng()
    rx = asyncio.create_task(controller_rx(ctrl_q, eng))

    # ---- a. C=1 byte-path identity ---------------------------------------------------------------
    print("[a] C=1 byte-path-identical (frames + order + lock semantics)")
    one = build_chain(1, delay=0.005)
    otask = asyncio.create_task(one[0].run(ctrl_q))
    m1 = mk_model(one[0], ["n0"], "acme/plain")
    prompt = torch.arange(100, 700, dtype=torch.long).reshape(1, 600)
    r0 = eng.req_counter
    res = await asyncio.wait_for(eng._send_prefill(m1, prompt), timeout=10)
    res = await asyncio.wait_for(eng._send(
        m1, torch.tensor([[9001]], dtype=torch.long), 600, False), timeout=10)
    res = await asyncio.wait_for(eng._send(
        m1, torch.tensor([[9002]], dtype=torch.long), 601, False), timeout=10)
    await eng._crop(m1, 500)
    await asyncio.sleep(0.15)
    # golden frames: EXACTLY what the pre-#kv-slots controller built (same key order, no 'slot')
    def _golden(hdr, x):
        meta, raw = _pack_tensor(x)
        return encode_frame({**hdr, **meta}, raw)
    g = [_golden({"req_id": r0 + 1, "model_id": "acme/plain", "kind": "ids",
                  "cache_position": 0, "reset": True, "all_logits": False}, prompt),
         _golden({"req_id": r0 + 2, "model_id": "acme/plain", "kind": "ids",
                  "cache_position": 600, "reset": False, "all_logits": False},
                 torch.tensor([[9001]], dtype=torch.long)),
         _golden({"req_id": r0 + 3, "model_id": "acme/plain", "kind": "ids",
                  "cache_position": 601, "reset": False, "all_logits": False},
                 torch.tensor([[9002]], dtype=torch.long)),
         encode_frame({"model_id": "acme/plain", "kind": "crop", "cache_position": 500}, b"")]
    check("frames byte-identical to the pre-#kv-slots golden encodings (same order)",
          one[0].rawlog == g,
          f"got {len(one[0].rawlog)} frames; first mismatch at "
          f"{next((i for i, (a, b) in enumerate(zip(one[0].rawlog, g)) if a != b), '?')}")
    check("no 'slot' key on any C=1 frame",
          all(b'"slot"' not in buf for buf in one[0].rawlog))
    check("C=1 record kept on the classic model.kv_ids attribute (helpers route to it)",
          eng._kv_rec_get(m1, 0) is None and m1.kv_ids is None)
    # _SlotLease at C=1 == model.lock, serialized
    async with engine_gen._SlotLease(m1, [1, 2]) as s:
        check("C=1 lease yields slot 0 and holds model.lock", s == 0 and m1.lock.locked())
        inner = engine_gen._SlotLease(m1, [1])
        try:
            await asyncio.wait_for(inner.__aenter__(), timeout=0.1)
            check("C=1 lease serializes (second acquire must block)", False, "did not block")
            await inner.__aexit__(None, None, None)
        except asyncio.TimeoutError:
            check("C=1 lease serializes (second acquire blocks)", True)
    check("C=1 lease releases model.lock on exit", not m1.lock.locked())

    # ---- b. C=3 concurrent gens: per-slot isolation ----------------------------------------------
    print("[b] C=3 concurrent interleaved gens (3 stages) — per-slot KV isolation")
    stages = build_chain(3, delay=0.02)
    stasks = [asyncio.create_task(st.run(ctrl_q)) for st in stages]
    lm = mk_model(stages[0], ["n0", "n1", "n2"], "acme/plain", kv_slots=3, friendly="mm3")
    prompts = [list(range(1000 * (k + 1), 1000 * (k + 1) + 64)) for k in range(3)]
    dtoks = [list(range(7000 + 10 * k, 7000 + 10 * k + 4)) for k in range(3)]
    got_slots, streams = [], {}

    async def gen(k):
        ids = prompts[k]
        async with engine_gen._SlotLease(lm, ids) as slot:
            got_slots.append(slot)
            res = await eng._prefill_reuse(lm, ids, slot=slot)
            assert float(res[0, -1, 0]) == float(len(ids)), "prefill marker != slot kv len"
            _kv = list(ids)
            eng._kv_rec_set(lm, slot, _kv)
            cur = len(ids)
            for t in dtoks[k]:
                res = await eng._send(lm, torch.tensor([[t]], dtype=torch.long), cur, False,
                                      slot=slot)
                cur += 1
                _kv.append(t)
                assert float(res[0, -1, 0]) == float(cur), "decode marker != slot kv len"
            streams[slot] = _kv

    await asyncio.wait_for(asyncio.gather(gen(0), gen(1), gen(2)), timeout=20)
    check("three concurrent gens got three DISTINCT slots", sorted(got_slots) == [0, 1, 2],
          str(got_slots))
    for st in stages:
        check(f"{st.name} no per-slot order violations", not st.violations, str(st.violations))
        ok_len = all(len(st.kv.get(s, [])) == 68 for s in (0, 1, 2))
        check(f"{st.name} per-slot KV lengths complete (68 each)", ok_len,
              str({s: len(st.kv.get(s, [])) for s in (0, 1, 2)}))
    for s in (0, 1, 2):
        check(f"stage0 slot {s} content == exactly its own stream (no cross-slot bleed)",
              stages[0].kv[s] == streams[s])
    _order = [t[0] for t in stages[1].seen]
    check("stage frames INTERLEAVE across slots (not grouped per gen)",
          any(a != b for a, b in zip(_order, _order[1:])), str(_order))
    check("pending maps scrubbed (incl. pending_slot)",
          not eng.pending and not eng.pending_model and not eng.pending_slot)
    check("slot pool fully returned after the gens",
          sorted(lm.slot_free) == [0, 1, 2] and lm.slots_active == 0 and not lm.slot_owner)

    # ---- e. per-slot #prefix-kv: LCP slot selection + REAL resume on slot 1 ----------------------
    print("[e] per-slot #prefix-kv records (LCP pick + crop/suffix resume on slot 1)")
    # e1: admission picks the free slot with the longest LCP record
    lm_lcp = mk_model(stages[0], ["n0", "n1", "n2"], "acme/plain", kv_slots=3, friendly="lcp")
    lm_lcp.kv_ids_slots = {0: list(range(50)), 1: list(range(200)), 2: None}
    lease = engine_gen._SlotLease(lm_lcp, list(range(200)) + [777])
    s = await lease.__aenter__()
    check("longest-LCP record wins the slot pick (slot 1 reused)", s == 1, f"picked {s}")
    await lease.__aexit__(None, None, None)
    check("record helpers stay slot-isolated",
          eng._kv_rec_get(lm_lcp, 0) == list(range(50))
          and eng._kv_rec_get(lm_lcp, 2) is None)
    # e2: REAL resume — slot 1's record matches a new prompt's prefix -> crop + suffix-only prefill
    suffix = [9101, 9102, 9103, 9104, 9105, 9106, 9107, 9108]
    new_prompt = streams[1] + suffix
    n_seen0 = len(stages[0].seen)
    res = await asyncio.wait_for(eng._prefill_reuse(lm, new_prompt, slot=1), timeout=10)
    check("resume reply marker == full prompt length on slot 1",
          float(res[0, -1, 0]) == float(len(new_prompt)))
    _new = stages[0].seen[n_seen0:]
    check("ONLY the suffix was prefilled (one reset=False frame at base, slot 1)",
          _new == [(1, len(streams[1]), len(suffix), False)], str(_new))
    check("#prefix-kv HIT logged", any("#prefix-kv HIT" in a for a in ACTIVITY))
    for st in stages:
        check(f"{st.name} slot1 resumed to full length; slots 0/2 untouched",
              len(st.kv[1]) == len(new_prompt) and len(st.kv[0]) == 68 and len(st.kv[2]) == 68)
    check("stage0 slot1 content == old stream + suffix (positions continued at L)",
          stages[0].kv[1] == new_prompt)

    # ---- c. crop on slot 1 leaves slots 0/2 intact ------------------------------------------------
    print("[c] crop on slot 1 leaves slots 0/2 intact")
    eng._kv_rec_set(lm, 0, list(streams[0]))   # give 0/2 live records to prove they survive
    eng._kv_rec_set(lm, 2, list(streams[2]))
    await eng._crop(lm, 10, slot=1)
    await asyncio.sleep(0.1)
    for st in stages:
        check(f"{st.name} slot1 cropped to 10; slot0/2 still 68",
              len(st.kv[1]) == 10 and len(st.kv[0]) == 68 and len(st.kv[2]) == 68,
              str({s: len(st.kv[s]) for s in (0, 1, 2)}))
    check("crop nulled ONLY slot 1's record",
          eng._kv_rec_get(lm, 1) is None
          and eng._kv_rec_get(lm, 0) == streams[0] and eng._kv_rec_get(lm, 2) == streams[2])
    check("crop frame carried the slot id", any(
        b'"kind": "crop"' in buf and b'"slot": 1' in buf for buf in stages[0].rawlog))

    # ---- b2. #pipefill chunk burst carries the gen's slot id --------------------------------------
    print("[b2] #pipefill chunk burst rides the slot (1200 tok, chunk 512, slot 2)")
    big = torch.arange(20000, 21200, dtype=torch.long).reshape(1, 1200)
    n0 = len(stages[0].seen)
    res = await asyncio.wait_for(eng._send_prefill(lm, big, slot=2), timeout=15)
    _new = stages[0].seen[n0:]
    check("every chunk carried slot 2, in prompt order (reset only at pos 0)",
          _new == [(2, 0, 512, True), (2, 512, 512, False), (2, 1024, 176, False)], str(_new))
    check("burst reply marker == 1200 (slot 2 stream rebuilt end to end)",
          torch.is_tensor(res) and float(res[0, -1, 0]) == 1200.0)
    check("slot-2 burst left slots 0/1 untouched on every stage",
          all(len(st.kv[0]) == 68 and len(st.kv[1]) == 10 and len(st.kv[2]) == 1200
              for st in stages),
          str([{s: len(st.kv[s]) for s in (0, 1, 2)} for st in stages]))
    check("stage0 slot-2 content == the burst prompt",
          stages[0].kv[2] == [int(v) for v in big.reshape(-1).tolist()])

    # ---- d. superseded forward on slot 2 doesn't touch slot 0 (REAL forward, threads) ------------
    print("[d] slot-scoped orphan cancel (real ShardForwardMixin.forward)")

    class FakeShard(shard_forward.ShardForwardMixin):
        def __init__(self):
            self._fwd_lock = threading.Lock()
            self._fwd_cancel = threading.Event()
            self.kv = None
            self._kv_by_slot = {}
            self.hold = 0.0

        def _forward_impl(self, x, cache_start=0, reset=True, *a, **k):
            hold = self.hold                      # per-call snapshot
            if reset or self.kv is None or cache_start == 0:
                self.kv = {"id": object(), "n": 0}
            t0 = time.time()
            while time.time() - t0 < hold:        # grinding layer loop w/ cooperative cancel
                if self._fwd_cancel.is_set():
                    raise shard_forward._ForwardSuperseded("superseded")
                time.sleep(0.005)
            self.kv["n"] += 1
            return self.kv

    _grace0 = shard_forward._ORPHAN_CANCEL_GRACE_S
    shard_forward._ORPHAN_CANCEL_GRACE_S = 0.4
    try:
        sh = FakeShard()
        x = torch.tensor([[1]], dtype=torch.long)
        kv0 = sh.forward(x, 0, True, slot=0)              # slot 0 prefill -> cache A
        kv1 = sh.forward(x, 0, True, slot=1)              # slot 1 prefill -> cache B
        check("per-slot caches are DISTINCT objects", kv0["id"] is not kv1["id"])
        r = sh.forward(x, 1, False, slot=0)               # slot 0 decode reuses cache A
        check("slot 0 decode re-bound its OWN cache", r is kv0 and kv0["n"] == 2)
        kv0_id, kv0_n = kv0["id"], kv0["n"]
        # orphan on slot 2 (grinds ~1.2s in a thread)
        sh.hold = 1.2
        orphan = {}
        def _run_orphan():
            try:
                orphan["res"] = sh.forward(x, 0, True, slot=2)
            except shard_forward._ForwardSuperseded:
                orphan["superseded"] = True
            except Exception as exc:
                orphan["err"] = exc
        th = threading.Thread(target=_run_orphan)
        th.start()
        time.sleep(0.1)                                    # let it own _fwd_lock
        # different-slot waiter: must NOT cancel the orphan -> fails fast after the grace
        sh.hold = 0.0
        t0 = time.time()
        try:
            sh.forward(x, 1, False, slot=0)
            check("different-slot waiter failed fast (no cancel)", False, "no exception")
        except RuntimeError as exc:
            check("different-slot waiter failed fast (shard busy), orphan NOT cancelled",
                  "shard busy" in str(exc) and th.is_alive()
                  and not sh._fwd_cancel.is_set(), f"{exc} alive={th.is_alive()}")
        check("slot 0's cache untouched by the different-slot contention",
              sh._kv_by_slot[0]["id"] is kv0_id and sh._kv_by_slot[0]["n"] == kv0_n)
        # same-slot supersede: cancels the orphan, new slot-2 forward proceeds
        kv2_new = sh.forward(x, 0, True, slot=2)
        th.join(timeout=3)
        check("same-slot forward superseded the orphan (_ForwardSuperseded raised)",
              orphan.get("superseded") is True and not th.is_alive(), str(orphan))
        check("new slot-2 forward completed with a FRESH cache",
              sh._kv_by_slot[2] is kv2_new and kv2_new["n"] == 1)
        check("superseded slot-2 forward did NOT touch slot 0",
              sh._kv_by_slot[0]["id"] is kv0_id and sh._kv_by_slot[0]["n"] == kv0_n)
        check("shard advertises _slot_capable (worker_net's gate)",
              getattr(sh, "_slot_capable", False) is True)
    finally:
        shard_forward._ORPHAN_CANCEL_GRACE_S = _grace0

    # ---- f. old-worker chain refuses C>1 at load + hard gates -------------------------------------
    print("[f] load-time gates (wire cap all-or-nothing + hard clamps)")
    E2 = type("E2", (engine_load.EngineLoadMixin,), {})
    e2 = E2()
    _stg = [types.SimpleNamespace(node_id="n0"), types.SimpleNamespace(node_id="oldworker")]
    _nbi = {"n0": types.SimpleNamespace(hostname="newbox"),
            "oldworker": types.SimpleNamespace(hostname="oldbox")}
    try:
        e2._kvslots_cap_check(3, _stg, _nbi)
        check("old-worker chain REFUSES kv_slots>1", False, "did not raise")
    except RuntimeError as exc:
        check("old-worker chain REFUSES kv_slots>1 (clear message names the node)",
              "kvslots" in str(exc) and "oldbox" in str(exc), str(exc))
    e2._kvslots_cap_check(1, _stg, _nbi)      # C=1 never checks caps
    check("C=1 passes the cap check on ANY chain", True)
    _stg_ok = [types.SimpleNamespace(node_id="n0"), types.SimpleNamespace(node_id="n1")]
    e2._kvslots_cap_check(4, _stg_ok, {"n0": _nbi["n0"], "n1": _nbi["n0"]})
    check("full-cap chain passes kv_slots>1", True)
    check("clamp: plain arch keeps C", e2._kvslots_clamp(3, 1, "none", False, plain_dir)
          == (3, ""))
    check("clamp: >8 clamps to 8", e2._kvslots_clamp(99, 1, "none", False, plain_dir)[0] == 8)
    check("clamp: tp>1 -> 1", e2._kvslots_clamp(3, 2, "none", False, plain_dir)[0] == 1)
    check("clamp: kv_quant -> 1", e2._kvslots_clamp(3, 1, "turbo3", False, plain_dir)[0] == 1)
    check("clamp: kv_offload -> 1", e2._kvslots_clamp(3, 1, "none", True, plain_dir)[0] == 1)
    check("clamp: hybrid arch -> 1", e2._kvslots_clamp(3, 1, "none", False, hybrid_dir)[0] == 1)
    check("clamp: missing config -> 1 (conservative)",
          e2._kvslots_clamp(3, 1, "none", False, os.path.join(models_dir, "nope"))[0] == 1)
    check("clamp: C=1 stays 1, no reason", e2._kvslots_clamp(1, 1, "none", False, plain_dir)
          == (1, ""))

    # ---- g. slot pool mechanics: exhaustion + watchdog-style reclaim ------------------------------
    print("[g] slot pool: exhaustion queues+logs; token-swap reclaim never double-frees")
    lm2 = mk_model(stages[0], ["n0", "n1", "n2"], "acme/plain", kv_slots=2, friendly="pool2")
    ACTIVITY.clear()
    l1, l2 = engine_gen._SlotLease(lm2, []), engine_gen._SlotLease(lm2, [])
    await l1.__aenter__()
    await l2.__aenter__()
    l3 = engine_gen._SlotLease(lm2, [])
    t3 = asyncio.create_task(l3.__aenter__())
    await asyncio.sleep(0.05)
    check("third acquirer QUEUES when both slots are busy", not t3.done())
    check("slot-pool exhaustion logged once", sum("kv-slots busy" in a for a in ACTIVITY) == 1)
    # watchdog-style reclaim of l1's slot: swap the token, free the slot, release the sem OURSELVES
    _sl = l1.slot
    lm2.slot_owner.pop(_sl, None)
    lm2.slot_state.pop(_sl, None)
    lm2.slot_free.append(_sl)
    lm2.slots_active = max(0, lm2.slots_active - 1)
    lm2.slot_sem.release()
    await asyncio.wait_for(t3, timeout=1)
    check("queued acquirer got the reclaimed slot", l3.slot == _sl)
    await l1.__aexit__(None, None, None)      # the orphan's release must be a NO-OP (token gone)
    check("orphan lease release after reclaim is a no-op (no double-free)",
          lm2.slot_free == [] or _sl not in lm2.slot_free)
    await l2.__aexit__(None, None, None)
    await l3.__aexit__(None, None, None)
    check("pool consistent after churn: both slots free exactly once, none active",
          sorted(lm2.slot_free) == [0, 1] and lm2.slots_active == 0 and not lm2.slot_owner,
          f"free={lm2.slot_free} active={lm2.slots_active} owner={lm2.slot_owner}")

    for t in stasks + [otask, rx]:
        t.cancel()
    print()
    if FAILURES:
        print(f"FAILED: {len(FAILURES)} check(s): {FAILURES}")
        return 1
    print("ALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(asyncio.run(main()))
