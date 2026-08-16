"""downloads: model download / registry lifecycle, relocated VERBATIM from server.py
(code-split Inc 5): _pull_repo_interruptible (module level) and, inside register(app):
_start_download, _do_delete, _resolve_or_404, /download, /download/pause|stop|resume|clear,
/add_model, /delete, /forget, /api/pull, /api/delete.

WHERE THE STATE LIVES (do not "fix" this): every DOWNLOAD_* global (DOWNLOADING,
DOWNLOAD_PROGRESS/_ERROR/_CONTROL/_STATE/_EPOCH, DOWNLOAD_STATE_PATH) is DEFINED in
server.py and only MUTATED IN PLACE here -- the self-updater's idle lambda in server.py
reads DOWNLOADING as a live server global, so moving those definitions (or rebinding them
anywhere) would silently decouple the self-update idle gate: the ENCODING hazard documented
in state.py (ENCODING itself moved to media_encode.py WITH its mutators in Inc 11; the
lambda reads media_encode.ENCODING live). load/save_download_state also stay in server.py
beside their data. The persistence loaders are in-place as of Inc 4, so this module's bound
snapshot of CUSTOM_MODELS/GGUF_FILES/DELETED_MODELS/MODELS stays live across a reload.
Bodies BYTE-IDENTICAL; module globals injected by state.bind() -- see state.py.
Controller-only leaf; never imports server; in EXTRA_UPDATE_FILES.
"""
from __future__ import annotations


def _pull_repo_interruptible(friendly: str, repo_id: str):
    """Download a repo's *.safetensors/*.json into the HF cache ONE FILE AT A TIME so a
    pause/stop (DOWNLOAD_CONTROL[friendly]) can interrupt it between files. Returns
    'done' (all files present), 'paused', or 'stopped'.

    Why per-file and not snapshot_download: the heavy pull runs in a thread, and a
    Python thread can't be force-killed — so the only clean interrupt point is between
    files. The control flag is checked BEFORE each file and the current file is allowed
    to finish, so resume granularity is WHOLE-FILE: every completed shard stays in the
    cache and is skipped on resume, and nothing in flight is abandoned (we never kill a
    file mid-write). The cost is that pause/stop take effect only after the current shard
    finishes — up to a couple minutes for a big one. (A hard crash mid-shard is a
    different matter: huggingface_hub 1.x discards that one partial shard's bytes and
    re-pulls it, but the already-completed shards are kept.) hf_transfer (env) still
    accelerates each individual file. If the repo listing keeps failing, fall back to a
    single (non-interruptible) snapshot_download so a download is never blocked by a flaky
    list call — pause/stop won't bite until it finishes in that rare mode."""
    from huggingface_hub import HfApi, hf_hub_download
    tok = HF_TOKEN or None
    files = None
    for attempt in range(3):                          # transient list failures are common
        try:
            files = HfApi().list_repo_files(repo_id, token=tok)
            break
        except Exception:
            if attempt == 2:
                from huggingface_hub import snapshot_download
                print(f"[model] {friendly}: repo listing failed -> non-interruptible "
                      f"snapshot_download fallback (pause/stop won't apply this run)")
                snapshot_download(repo_id, allow_patterns=["*.safetensors", "*.json", "*.py",
                                                           "*.jinja", "*.txt", "*.model",
                                                           "*.pth", "*.pt"], token=tok)
                return "done"
    # include *.py: trust_remote_code models (auto_map) ship their modeling/configuration code as
    # .py — without them a worker builds the native class for the model_type (wrong arch -> meta
    # tensors, e.g. MiniMax-M2 'minimax' -> lightning Text-01). #78.
    # include *.jinja/*.txt/*.model: chat templates + tokenizer sidecars (merges.txt,
    # sentencepiece .model) — diffusers repos (#t2i) ship the tokenizer under tokenizer/ with
    # these, and Mistral3-style LLMs ship chat_template.jinja. Extension set mirrors
    # _hf_total_bytes so the progress denominator matches what is pulled.
    _ext = [".safetensors", ".json", ".py", ".jinja", ".txt", ".model"]
    # #tts: a repo with NO safetensors ships its weights as .pth/.pt (Kokoro = kokoro-v1_0.pth
    # + voices/*.pt). Pull those too so "+ Add model" fetches a COMPLETE non-safetensors model
    # instead of just config.json. Gated on "no safetensors" so ordinary checkpoints never pull
    # redundant/stray .pt (training snapshots, EMA copies) alongside their real safetensors.
    # #t2music: MusicGen ships weights as pytorch_model.bin (NO safetensors, NOT .pth/.pt). Add the
    # HF weight .bin (pytorch_model*/model*) so "+ Add model" fetches a COMPLETE MusicGen — but NOT
    # the redundant raw state_dict.bin / compression_state_dict.bin it ALSO ships (they duplicate the
    # HF weights ~2x and from_pretrained ignores them; verified medium loads from pytorch_model.bin
    # alone). Gated on "no safetensors" so ordinary checkpoints never pull a stray .bin.
    _hf_bin = []
    if not any(f.endswith(".safetensors") for f in files):
        _ext += [".pth", ".pt"]
        _hf_bin = [f for f in files if f.endswith(".bin")
                   and f.rsplit("/", 1)[-1].startswith(("pytorch_model", "model"))]
    wanted = [f for f in files if f.endswith(tuple(_ext))] + _hf_bin
    for f in wanted:
        ctrl = DOWNLOAD_CONTROL.get(friendly)        # checked BETWEEN files (cheap dict read)
        if ctrl in ("pause", "stop"):
            return "paused" if ctrl == "pause" else "stopped"
        # Windows WinError 32: an AV / a cache scanner can momentarily lock the freshly-pulled
        # blob right as huggingface_hub renames its `.incomplete` -> final, killing the download
        # at ~finalize with "used by another process". The bytes ARE on disk (hf_hub keeps the
        # `.incomplete`), so the finalize just needs to be retried once the handle releases —
        # retry with backoff instead of failing the whole pull. Non-WinError-32 errors re-raise.
        for _att in range(6):
            try:
                hf_hub_download(repo_id, f, token=tok)   # instant if the shard is already cached
                break
            except OSError as exc:                   # PermissionError (WinError 32) is an OSError
                if getattr(exc, "winerror", None) != 32 or _att == 5:
                    raise
                print(f"[model] {friendly}: {f} finalize locked (WinError 32) — "
                      f"retry {_att + 1}/5 after {2 * (_att + 1)}s", flush=True)
                time.sleep(2 * (_att + 1))
    return "done"


# --- #dl-autoresolve: make a bare `org/Model` "just work" without the user choosing a quant or a
# source format. Probe the repo and, for a GGUF-ONLY repo, PREFER its safetensors twin (native,
# higher fidelity, and the ONLY option for arches transformers' GGUF loader can't dequantize — e.g.
# qwen35moe / Qwen3.5-MoE, where the GGUF metadata arch name doesn't match transformers'); else
# auto-pick a single-file quant to normalize to safetensors via the existing converter. -----------
_GGUF_QUANT_PREF = (
    "q4_k_m", "q5_k_m", "q4_k_l", "q5_k_l", "q6_k", "q4_k_s", "q5_k_s", "q8_0",
    "q3_k_l", "q3_k_m", "q4_0", "q5_0", "iq4_xs", "iq4_nl", "q3_k_s", "q2_k",
    "iq3_m", "iq3_xs", "iq2_m", "iq2_xs",
)


def _pick_gguf_quant(gguf_files):
    """Choose the best quant to normalize from a repo's .gguf filenames — a SINGLE-FILE quant OR a
    COMPLETE split (NNNNN-of-NNNNN) set, represented by its part 1. Prefers a medium K-quant — the
    best size/quality for a source we re-quantize to int4 anyway. Raw float dumps
    (*f16/*f32/*bf16.gguf) are a last resort. Returns ``(pick, error)``: the filename to register,
    or ``(None, why)``.

    #gguf-split: split sets used to be dropped on the floor here and the repo then refused
    outright — but the GGUF-only models actually worth ingesting are exactly the big split ones,
    so they are now first-class candidates. A set qualifies ONLY when every part 1..N is present
    in the repo listing. A gap must fail LOUD *here*, at add time, with the repo listing in hand:
    the converter downloads parts by synthesized name, so a set missing part 3 of 5 would either
    404 deep inside a multi-hour pull or — if a reader tolerated it — dequantize a model with whole
    layers absent, which saves, passes its own checksum, and generates garbage.

    A single file WINS ties against a split set of the same quant rank: one file is the path that
    has always worked, needs no sibling resolution, and costs the converter nothing extra."""
    from model_store import gguf_split_info
    base = lambda f: f.rsplit("/", 1)[-1].lower()
    singles, sets = [], {}
    for f in gguf_files:
        if not f.lower().endswith(".gguf"):
            continue
        info = gguf_split_info(f)
        if info is None:
            singles.append(f)                        # incl. names whose series we can't map (see helper)
        else:
            b, idx, tot = info
            sets.setdefault((b, tot), {})[idx] = f
    complete, incomplete = [], []
    for (b, tot), parts in sorted(sets.items()):
        if set(parts) == set(range(1, tot + 1)):
            complete.append(parts[1])                # register part 1; the rest derive from its name
        else:
            incomplete.append((b, tot, [i for i in range(1, tot + 1) if i not in parts]))
    cands = singles + complete
    if not cands:
        if incomplete:
            b, tot, missing = incomplete[0]
            shown = ", ".join(f"{i:05d}" for i in missing[:6]) + (" …" if len(missing) > 6 else "")
            return None, (f"{b}: split GGUF set is INCOMPLETE — {len(missing)} of {tot} parts are "
                          f"missing from the repo (part {shown}). Converting a partial set would "
                          f"produce a model with missing layers, so it is refused. Re-check the "
                          f"repo, or name a different quant in the GGUF field.")
        return None, None
    single_set = set(singles)
    is_float = lambda f: any(t in base(f) for t in ("f16", "f32", "bf16"))

    def _score(f):
        b = base(f)
        for i, q in enumerate(_GGUF_QUANT_PREF):
            if q in b:
                return i
        return len(_GGUF_QUANT_PREF)                 # unknown quant -> after all known, before floats
    # (score, single-before-split, name) — the last term only makes the choice deterministic when a
    # repo ships two same-rank candidates, so the same repo always resolves to the same file.
    _key = lambda f: (_score(f), 0 if f in single_set else 1, f)
    ranked = sorted((f for f in cands if not is_float(f)), key=_key)
    if ranked:
        return ranked[0], None
    return sorted(cands, key=_key)[0], None          # only float dumps -> still translatable


def _resolve_download_source(hf, gguf_file=""):
    """#dl-autoresolve: decide the ACTUAL download source + friendly name for a raw HF id so a bare
    `org/Model` just works. NETWORK (HfApi list) — call via asyncio.to_thread. An explicit
    SINGLE-FILE gguf_file is honored verbatim (power-user override, no probe); an explicit SPLIT
    part is still listed and checked, because "honour it verbatim" there means converting one part
    of N (#gguf-split). Never raises: a list failure (gated / typo / offline) leaves the id
    untouched so the download path surfaces the real error. Returns
    {"target", "gguf_file", "friendly_hint", "note", "error"}."""
    import re as _re
    from huggingface_hub import HfApi
    from model_store import gguf_part_names, gguf_split_info
    tok = HF_TOKEN or None
    api = HfApi()
    gf = (gguf_file or "").strip()
    if gf:
        _info = gguf_split_info(gf)
        if not _info:
            return {"target": hf, "gguf_file": gf, "friendly_hint": "",
                    "note": f"explicit GGUF file {gf}", "error": None}
        # #gguf-split: an EXPLICITLY named part gets the same completeness check as an auto-picked
        # set, and is normalized to part 1 (what the converter opens). Without this, hand-typing
        # "...-00003-of-00005.gguf" would register a single middle part and convert 1/5 of a model.
        _want = gguf_part_names(gf)
        _first, _tot = _want[0], len(_want)
        try:
            _have = set(api.list_repo_files(hf, token=tok))
        except Exception:
            # Same contract as the rest of this function: a listing failure never raises and never
            # blocks — honour the override and let the download path surface the real error.
            return {"target": hf, "gguf_file": _first, "friendly_hint": "",
                    "note": (f"explicit split GGUF {_first} (+{_tot - 1} parts) — repo listing "
                             "unavailable, completeness NOT verified"), "error": None}
        _missing = [p for p in _want if p not in _have]
        if _missing:
            return {"target": hf, "gguf_file": "", "friendly_hint": "", "note": "",
                    "error": (f"{hf}: split GGUF set for {_first} is INCOMPLETE — {len(_missing)} "
                              f"of {_tot} parts are not in the repo ({_missing[0]}"
                              + (" …" if len(_missing) > 1 else "") + "). Refusing: converting a "
                              "partial set would produce a model with missing layers.")}
        return {"target": hf, "gguf_file": _first, "friendly_hint": "",
                "note": f"explicit split GGUF {_first} (+{_tot - 1} parts, all present)",
                "error": None}
    try:
        files = api.list_repo_files(hf, token=tok)
    except Exception:
        return {"target": hf, "gguf_file": "", "friendly_hint": "", "note": "", "error": None}
    ggufs = [f for f in files if f.lower().endswith(".gguf")]
    has_st = any(f.lower().endswith(".safetensors") for f in files)
    if has_st or not ggufs:                          # ordinary safetensors (or .pth/.bin) repo -> as-is
        return {"target": hf, "gguf_file": "", "friendly_hint": "", "note": "", "error": None}
    # --- GGUF-ONLY repo ---
    # 1) prefer a safetensors TWIN (the repo id minus its -GGUF tag): native + higher fidelity, and
    #    the only path for arches the GGUF loader can't dequantize (e.g. Ornith's qwen35moe).
    sib = _re.sub(r"[-_.]?gguf$", "", hf, flags=_re.I)
    if sib and sib != hf:
        try:
            if any(f.lower().endswith(".safetensors") for f in api.list_repo_files(sib, token=tok)):
                return {"target": sib, "gguf_file": "", "friendly_hint": _friendly_from_hf(sib),
                        "note": f"{hf} is GGUF-only — using its safetensors source {sib}",
                        "error": None}
        except Exception:
            pass                                     # no twin -> fall through to GGUF translate
    # 2) no twin -> auto-translate to safetensors (the existing converter). A COMPLETE split set is
    #    a valid pick now (#gguf-split) and comes back as its part 1.
    pick, why = _pick_gguf_quant(ggufs)
    if not pick:
        return {"target": hf, "gguf_file": "", "friendly_hint": "", "note": "",
                "error": (why or f"{hf} lists .gguf files but none is usable as a conversion source")}
    clean = _re.sub(r"-gguf$", "", _friendly_from_hf(hf), flags=_re.I)
    _n = len(gguf_part_names(pick))
    _sel = pick if _n == 1 else f"{pick} (+{_n - 1} parts, split set)"
    return {"target": hf, "gguf_file": pick, "friendly_hint": clean,
            "note": f"{hf} is GGUF-only — auto-selected {_sel} to normalize to safetensors",
            "error": None}


_START_DOWNLOAD = None   # #dl-resume: register() publishes _start_download here so the server's
                         # startup lifespan can re-kick a pull that a controller restart interrupted.


async def resume_interrupted_downloads() -> None:
    """#dl-resume: after load_download_state() repopulates DOWNLOAD_STATE, re-trigger every download
    that was ACTIVE ("downloading") when the controller was killed — completed shards are cached and
    skipped, so each continues where it stopped. Paused/stopped intents are left halted (user-driven).
    No-op if the exposure hook is unset or there are no active entries; one bad entry never blocks the
    rest. Runs as a startup task off the lifespan (downloads pull to the controller's OWN HF cache, so
    no worker-fleet settle is needed)."""
    import asyncio
    if _START_DOWNLOAD is None:
        return
    await asyncio.sleep(5)                       # let startup settle before streaming weights
    pending = [f for f, st in list((DOWNLOAD_STATE or {}).items()) if st == "downloading"]
    for friendly in pending:
        try:
            if friendly in DOWNLOADING:          # already live (shouldn't happen this early)
                continue
            target = MODELS[friendly][0] if friendly in MODELS else friendly
            if model_ready(target):              # finished between kill and restart -> clear, done
                DOWNLOAD_STATE.pop(friendly, None)
                await asyncio.to_thread(save_download_state)
                continue
            log_activity(f"download {friendly}: auto-resuming — interrupted by a controller restart")
            await _START_DOWNLOAD(friendly)
        except Exception as exc:                 # one bad entry must not block the others
            log_activity(f"download {friendly}: auto-resume failed ({exc!r})")


def register(app):

    async def _start_download(friendly: str) -> dict:
        """Kick off a background download of a configured model to the controller
        cache (fire-and-forget). Idempotent: no-op if ready. If a pull is already in
        flight, a pending pause/stop is CANCELLED (so Resume-during-pausing un-pauses
        the live thread instead of being a silent no-op)."""
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        if model_ready(target):
            if DOWNLOAD_STATE.pop(friendly, None) is not None:   # #dl-resume: ready -> drop any stale halt flag
                save_download_state()
            return {"ok": True, "status": "ready"}
        if friendly in DOWNLOADING:
            # Already pulling — if a pause/stop was pending, drop it so the running thread
            # keeps going (the between-files check now sees no signal). No new _dl needed.
            if DOWNLOAD_CONTROL.pop(friendly, None) is not None:
                log_activity(f"download {friendly}: resume (cancelled pending halt)")
                return {"ok": True, "status": "resuming"}
            return {"ok": True, "status": "downloading"}
        DOWNLOADING.add(friendly)
        DOWNLOAD_ERROR.pop(friendly, None)   # clear any prior failure on a fresh attempt
        DOWNLOAD_CONTROL.pop(friendly, None)  # drop any stale pause/stop signal from a prior run
        # #dl-resume: persist an ACTIVE-download marker (overwrites any paused/stopped halt). If the
        # controller is KILLED mid-pull this "downloading" state survives in download_state.json and
        # startup auto-resumes it (cached shards skipped). Cleared on clean completion/error below; a
        # pause/stop overwrites it with the halt intent (kept across a restart, but NOT auto-resumed).
        DOWNLOAD_STATE[friendly] = "downloading"
        save_download_state()
        epoch = DOWNLOAD_EPOCH[friendly] = DOWNLOAD_EPOCH.get(friendly, 0) + 1
        log_activity(f"download {friendly}: starting")

        async def _dl():
            total = await asyncio.to_thread(_hf_total_bytes, target)
            DOWNLOAD_PROGRESS[friendly] = {
                "downloaded": await asyncio.to_thread(_hf_cache_bytes, target),
                "total": total}

            async def _poll():   # update bytes-so-far (+ rolling rate/ETA) while the download runs
                samples: list[tuple[float, int]] = []   # (monotonic ts, bytes) over a ~30s window
                try:
                    while friendly in DOWNLOADING and DOWNLOAD_EPOCH.get(friendly) == epoch:
                        db = await asyncio.to_thread(_hf_cache_bytes, target)
                        pr = DOWNLOAD_PROGRESS.get(friendly)
                        if pr is not None:
                            pr["downloaded"] = db
                            # Rolling average rate over the trailing ~30s window (smooths the
                            # per-file steps), then ETA = bytes-remaining / rate. Both live in pr
                            # so /status can surface them; cleared with pr when the download ends.
                            now = time.monotonic()
                            samples.append((now, db))
                            cutoff = now - 30.0
                            while len(samples) > 2 and samples[0][0] < cutoff:
                                samples.pop(0)
                            dt = samples[-1][0] - samples[0][0]
                            dbytes = samples[-1][1] - samples[0][1]
                            if dt >= 1.0 and dbytes > 0:
                                rate = dbytes / dt          # bytes/sec
                                pr["rate"] = rate
                                tot = pr.get("total") or 0
                                pr["eta_s"] = (tot - db) / rate if tot > db else 0.0
                        await asyncio.sleep(2)
                except asyncio.CancelledError:
                    pass

            poller = asyncio.create_task(_poll())
            halted = None
            try:
                # GGUF source: no safetensors to pull. Normalize the .gguf to a safetensors checkpoint
                # in a SUBPROCESS (download + dequant + save), then it's an ordinary model. Pause/stop
                # don't apply (it's a one-shot subprocess), so skip the interruptible pull entirely.
                if target in GGUF_FILES:
                    # #gguf-split: name the part count so a multi-hour multi-part pull is legible in
                    # the activity log (one "conversion" line for a 5-part 40 GB set was opaque).
                    from model_store import gguf_part_names as _gpn
                    _np = len(_gpn(GGUF_FILES[target]))
                    log_activity(f"download {friendly}: GGUF -> safetensors conversion (subprocess"
                                 + (f", {_np}-part split set" if _np > 1 else "") + ")")
                    await asyncio.to_thread(_controller_model_dir, target)   # triggers convert_gguf_to_model_dir
                    if DOWNLOAD_EPOCH.get(friendly) != epoch:
                        return
                    _invalidate_ready_cache(target)
                    if DOWNLOAD_STATE.pop(friendly, None) is not None:   # #dl-resume: done -> not active
                        await asyncio.to_thread(save_download_state)
                    print(f"[model] converted GGUF {friendly} ({target} :: {GGUF_FILES[target]})")
                    log_activity(f"download {friendly}: complete (GGUF normalized)")
                    return
                # Interruptible per-file pull -> 'done' | 'paused' | 'stopped'.
                result = await asyncio.to_thread(_pull_repo_interruptible, friendly, target)
                # If a clear (or a fresh start) bumped the epoch while we were pulling, THIS
                # run is stale — that handler already cleaned up; don't write/resurrect state.
                if DOWNLOAD_EPOCH.get(friendly) != epoch:
                    return
                if result in ("paused", "stopped"):
                    halted = result
                    DOWNLOAD_STATE[friendly] = result    # persist so a restart keeps it halted
                    await asyncio.to_thread(save_download_state)
                    done = (DOWNLOAD_PROGRESS.get(friendly) or {}).get("downloaded", 0)
                    print(f"[model] download {friendly} {result} at {done / GB:.1f} GB")
                    log_activity(f"download {friendly}: {result}")
                else:
                    # every file now in the HF cache -> migrate to models/ + purge the dup
                    await asyncio.to_thread(_controller_model_dir, target)
                    _invalidate_ready_cache(target)
                    if DOWNLOAD_STATE.pop(friendly, None) is not None:   # #dl-resume: done -> not active
                        await asyncio.to_thread(save_download_state)
                    print(f"[model] downloaded {friendly} ({target})")
                    log_activity(f"download {friendly}: complete")
            except Exception as exc:
                if DOWNLOAD_EPOCH.get(friendly) != epoch:
                    return                               # superseded -> swallow
                msg = f"{type(exc).__name__}: {exc}"
                low = msg.lower()
                if any(k in low for k in ("gated", "403", "401", "awaiting", "access to model",
                                          "restricted", "you must")):
                    msg += "  (gated repo — accept the license for this model on huggingface.co "
                    msg += "with the account whose token the controller uses)"
                DOWNLOAD_ERROR[friendly] = msg[:400]
                # #dl-resume: a CLEAN failure clears the active marker (the error is surfaced; the
                # user resumes manually). Only a process-kill — which never reaches this handler —
                # leaves "downloading" set for startup auto-resume.
                if DOWNLOAD_STATE.pop(friendly, None) is not None:
                    await asyncio.to_thread(save_download_state)
                print(f"[model] download failed for {friendly}: {exc!r}")
                log_activity(f"download {friendly}: FAILED ({type(exc).__name__})")
            finally:
                poller.cancel()
                if DOWNLOAD_EPOCH.get(friendly) == epoch:   # only OUR run owns this state
                    DOWNLOADING.discard(friendly)
                    DOWNLOAD_CONTROL.pop(friendly, None)
                    # done/error -> drop the progress bar; pause/stop -> KEEP it frozen so the
                    # dashboard shows where it halted (and offers Resume from there).
                    if halted is None:
                        DOWNLOAD_PROGRESS.pop(friendly, None)

        asyncio.create_task(_dl())
        return {"ok": True, "status": "downloading"}

    # #dl-resume: publish _start_download at module scope so the server's startup lifespan can
    # re-kick any pull interrupted by a controller restart (see resume_interrupted_downloads).
    globals()["_START_DOWNLOAD"] = _start_download

    async def _do_delete(friendly: str) -> dict:
        """Delete a model COMPLETELY from the controller: its weight/quant CACHE
        (models/<name>/ incl. the _shards/<quant>/ pre-quant caches AND the HF-cache
        copy) AND its registry footprint — EVERY registered name that resolves to the
        same repo (the model + any alias names re-registered against the same HF id),
        its GGUF mark, and any built-in alias pointing at it. Full removal: delete ==
        forget + purge files. Refuses if any of those names is loaded or downloading."""
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        # Every name that resolves to the SAME repo shares the on-disk files we're about
        # to remove, so all of them must be unregistered too — otherwise they'd dangle on
        # a now-missing model. Custom 'aliases' = multiple CUSTOM_MODELS keys -> one HF id;
        # plus any built-in MODEL_ALIASES entry whose key or canonical target is in the set.
        names = {friendly} | {k for k, hf in CUSTOM_MODELS.items() if hf == target}
        names |= {a for a, c in MODEL_ALIASES.items() if a in names or c in names}
        loaded = sorted(n for n in names if n in engine.models)
        if loaded:
            return {"ok": False,
                    "error": f"model is currently loaded ({', '.join(loaded)}) — unload it first"}
        busy = sorted(n for n in names if n in DOWNLOADING)
        if busy:
            return {"ok": False,
                    "error": f"model is downloading ({', '.join(busy)}) — wait for it to finish"}
        deleted = await asyncio.to_thread(delete_model_cache, target)
        # Purge the registry footprint regardless of whether files were present, so a
        # half-registered model (registered, no files) is still fully removed by a delete.
        forgot, hidden = [], []
        for n in list(names):
            if CUSTOM_MODELS.pop(n, None) is not None:
                forgot.append(n)               # custom: persistence via custom_models.json
            elif n in MODELS:
                hidden.append(n)               # built-in: persistence via the deleted hide-set
            MODELS.pop(n, None)                # drop from the live list (no stale download button)
            MODEL_ALIASES.pop(n, None)         # drop any alias keyed by this name
        if forgot:
            GGUF_FILES.pop(target, None)       # all registrations for this repo are gone
            save_custom_models()
        if hidden:
            DELETED_MODELS.update(hidden)
            save_deleted_models()
        removed = bool(deleted or forgot or hidden)
        if removed:
            print(f"[model] deleted {friendly} ({target}) — cache_removed={deleted} "
                  f"unregistered={sorted(forgot) or '[]'} hidden={sorted(hidden) or '[]'}", flush=True)
        return {"ok": removed,
                "error": None if removed else "model not present in cache or registry"}

    @app.post("/download")           # dashboard: pull a configured model to controller
    async def download(model: str) -> JSONResponse:
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        return JSONResponse(await _start_download(friendly))

    def _resolve_or_404(model: str):
        try:
            return resolve_model_name(model), None
        except ValueError as exc:
            return None, JSONResponse({"ok": False, "error": str(exc)}, status_code=404)

    @app.post("/download/pause")     # dashboard: pause an in-flight download (cache kept, resumable)
    async def download_pause(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        if friendly not in DOWNLOADING:
            return JSONResponse({"ok": False, "error": "not currently downloading"}, status_code=409)
        DOWNLOAD_CONTROL[friendly] = "pause"   # the per-file pull stops after the current shard
        log_activity(f"download {friendly}: pause requested")
        return JSONResponse({"ok": True, "status": "pausing"})

    @app.post("/download/stop")      # dashboard: stop an in-flight download (cache kept, resumable)
    async def download_stop(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        if friendly not in DOWNLOADING:
            return JSONResponse({"ok": False, "error": "not currently downloading"}, status_code=409)
        DOWNLOAD_CONTROL[friendly] = "stop"
        log_activity(f"download {friendly}: stop requested")
        return JSONResponse({"ok": True, "status": "stopping"})

    @app.post("/download/resume")    # dashboard: resume a paused/stopped download from the cache
    async def download_resume(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        # _start_download clears the persisted halt + stale control signal, then re-runs the
        # per-file pull — cached files are skipped instantly and the partial file resumes.
        return JSONResponse(await _start_download(friendly))

    @app.post("/download/clear")     # dashboard: wipe a model's cached + partial files ("reset")
    async def download_clear(model: str) -> JSONResponse:
        friendly, err = _resolve_or_404(model)
        if err:
            return err
        if friendly in engine.models:
            return JSONResponse({"ok": False, "error": "model is loaded — unload it first"},
                                status_code=409)
        target = MODELS[friendly][0] if friendly in MODELS else friendly
        if friendly in DOWNLOADING:
            DOWNLOAD_CONTROL[friendly] = "stop"   # ask the in-flight pull to stop between files
        else:
            DOWNLOAD_CONTROL.pop(friendly, None)  # nothing running -> don't leave a stale signal
        # Bump the epoch so the (possibly still-running) _dl for this model sees its result is
        # stale and won't re-persist DOWNLOAD_STATE after we clear it (the clear-resurrect race).
        DOWNLOAD_EPOCH[friendly] = DOWNLOAD_EPOCH.get(friendly, 0) + 1
        DOWNLOADING.discard(friendly)
        DOWNLOAD_PROGRESS.pop(friendly, None)
        DOWNLOAD_ERROR.pop(friendly, None)
        if DOWNLOAD_STATE.pop(friendly, None) is not None:
            save_download_state()
        # Measure both locations first (cache copy + any models/<name>), then delete both.
        # rmtree uses ignore_errors, so a file still locked by an in-flight pull thread is
        # skipped — a second Clear (after the thread exits between files) mops up any residue.
        def _measure() -> int:
            n = _hf_cache_bytes(target)
            mdir = os.path.join(MODELS_DIR, _safe_name(target))
            if os.path.isdir(mdir):
                for root, _dirs, files in os.walk(mdir):
                    for f in files:
                        with contextlib.suppress(OSError):
                            n += os.path.getsize(os.path.join(root, f))
            return n
        freed = await asyncio.to_thread(_measure)
        removed = await asyncio.to_thread(delete_model_cache, target)
        log_activity(f"download {friendly}: cache cleared (~{freed / GB:.1f} GB)")
        print(f"[model] cleared cache for {friendly} ({target})")
        return JSONResponse({"ok": True, "removed": removed, "freed_gb": round(freed / GB, 2)})

    @app.post("/add_model")          # dashboard: register + download ANY Hugging Face id
    async def add_model(model: str, name: str = "", gguf_file: str = "",
                        dry_run: bool = False) -> JSONResponse:
        # `name` (optional): override the client-facing model name instead of deriving it from
        # the HF id. Lets a precision-suffixed repo (e.g. ModelCloud/MiniMax-M2-BF16) be served
        # under a clean, quant-agnostic name (e.g. minimax-m2) — quant is a load-time choice, so
        # it shouldn't live in the name. Re-registering an already-cached HF id under a new name
        # is instant (no re-download — the cache is keyed by HF id, not the friendly name).
        # `dry_run` (optional): resolve the source + name and RETURN the plan WITHOUT registering or
        # downloading — the resolver preview (tests + a UI "what will this actually pull?" hint).
        import asyncio
        hf = (model or "").strip()
        # HF repo ids are colon-free (dash form, e.g. 'Qwen/Qwen3-4B'). A user may paste the Ollama
        # 'family:size' form into the org/name field ('qwen/qwen3:4b'), which 404s on the Hub.
        # Normalize ':' -> '-' in the TARGET id so both forms resolve to the real repo.
        hf = hf.replace(":", "-")
        if "/" not in hf or " " in hf or hf.count("/") > 1:
            return JSONResponse({"ok": False,
                                 "error": "enter a Hugging Face id like org/name"},
                                status_code=400)
        gf_in = (gguf_file or "").strip()
        if gf_in and not gf_in.lower().endswith(".gguf"):
            return JSONResponse({"ok": False,
                                 "error": ("gguf_file must be a .gguf filename in the repo "
                                           "(any part of a split NNNNN-of-NNNNN set is fine — it "
                                           "is normalized to part 1 and the set is verified)")},
                                status_code=400)
        # #dl-autoresolve: probe the repo so a bare `org/Model` just works. A GGUF-only repo
        # redirects to its safetensors twin (or auto-picks a quant — single-file or a complete
        # split set — to normalize), and the friendly name is derived from the RESOLVED target. An
        # explicit gguf_file skips the source probe (but a split part is still verified complete).
        res = await asyncio.to_thread(_resolve_download_source, hf, gf_in)
        if res.get("error"):
            return JSONResponse({"ok": False, "error": res["error"]}, status_code=400)
        target, gf, note = res["target"], res["gguf_file"], (res.get("note") or "")
        # friendly: explicit `name` override (collapsed to the canonical dash key) > resolver hint
        # (clean, twin-derived) > derived from the resolved target.
        if (name or "").strip():
            friendly = _normalize_model_request(name)
        else:
            friendly = res.get("friendly_hint") or _friendly_from_hf(target)
        if not re.fullmatch(r"[a-z0-9._-]+", friendly):
            return JSONResponse({"ok": False,
                                 "error": "name must be lowercase [a-z0-9._-] (':' allowed as the size separator)"},
                                status_code=400)
        if dry_run:                              # resolve-only preview: no registration, no download
            return JSONResponse({"ok": True, "dry_run": True, "input": hf, "friendly": friendly,
                                 "target": target, "gguf_file": gf or None, "note": note or None})
        if friendly not in MODELS:
            MODELS[friendly] = (target, target)  # draft = target (no speculative)
            CUSTOM_MODELS[friendly] = target
            if gf:
                GGUF_FILES[target] = gf          # mark this target as GGUF-sourced
            save_custom_models()
            log_activity(f"added model {friendly} ({target})"
                         + (f" [GGUF {gf}]" if gf else "") + (f" — {note}" if note else ""))
        elif gf and GGUF_FILES.get(target) != gf:
            GGUF_FILES[target] = gf              # update the chosen quant for an already-registered repo
            save_custom_models()
        if friendly in DELETED_MODELS:           # re-adding a previously deleted model un-hides it
            DELETED_MODELS.discard(friendly)
            save_deleted_models()
        r = await _start_download(friendly)
        return JSONResponse({"ok": True, "friendly": friendly, "target": target,
                             "gguf_file": gf or None, "note": note or None,
                             "status": r.get("status")})

    @app.post("/delete")             # dashboard: delete a model from controller
    async def delete_model(model: str) -> JSONResponse:
        try:
            friendly = resolve_model_name(model)
        except ValueError as exc:
            return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        r = await _do_delete(friendly)
        return JSONResponse(r, status_code=200 if r["ok"] else 409)

    @app.post("/forget")             # remove a custom-model REGISTRY entry but KEEP its weight files
    async def forget_model(model: str) -> JSONResponse:
        """Unregister a custom (added) model: drop its friendly->HF mapping from CUSTOM_MODELS +
        MODELS + custom_models.json. UNLIKE /delete, this does NOT delete the cached weight files
        — the model stays on disk, just no longer registered. Refuses if currently loaded."""
        # Prefer the LITERAL registered entry over an alias redirect: a custom model whose name
        # is shadowed by a built-in MODEL_ALIASES entry (e.g. 'qwen2.5:14b', shadowed by
        # qwen2.5-14b -> qwen2.5-14b-instruct) is otherwise UNFORGETTABLE — resolve_model_name
        # would redirect to the alias target and report it "loaded". (#forget-shadow)
        literal = _normalize_model_request(model)
        if literal in CUSTOM_MODELS:
            friendly = literal
        else:
            try:
                friendly = resolve_model_name(model)
            except ValueError as exc:
                return JSONResponse({"ok": False, "error": str(exc)}, status_code=404)
        if friendly in engine.models:
            return JSONResponse({"ok": False, "error": "model is loaded — unload it first"},
                                status_code=409)
        if friendly not in CUSTOM_MODELS:
            # Not a custom entry. A BUILT-IN can still be removed from the list: "forget" HIDES it
            # (persisted in the deleted hide-set) while KEEPING any downloaded weights — unlike
            # /delete, which also purges files. Re-adding via /add_model un-hides it. Without this a
            # built-in (e.g. mixtral:8x7b) flashed "built-ins can't be forgotten" and never left the
            # list. (#forget-builtin)
            if friendly in MODELS:
                MODELS.pop(friendly, None)
                MODEL_ALIASES.pop(friendly, None)   # drop any alias keyed by this name
                DELETED_MODELS.add(friendly)
                save_deleted_models()
                print(f"[model] forgot built-in {friendly} (hidden from list; weight files KEPT)",
                      flush=True)
                return JSONResponse({"ok": True, "forgot": friendly, "hf": None,
                                     "files_kept": True, "builtin": True})
            return JSONResponse({"ok": False, "error": f"'{friendly}' is not a registered model"},
                                status_code=404)
        hf = CUSTOM_MODELS.pop(friendly, None)
        MODELS.pop(friendly, None)
        if hf and not any(v == hf for v in CUSTOM_MODELS.values()):
            GGUF_FILES.pop(hf, None)   # last registry entry for this repo gone -> drop its GGUF mark
        save_custom_models()
        print(f"[model] forgot registry entry {friendly} ({hf}) — weight files KEPT", flush=True)
        return JSONResponse({"ok": True, "forgot": friendly, "hf": hf, "files_kept": True})

    @app.post("/api/pull")           # Ollama-compat pull -> background download
    async def api_pull(req: Request) -> JSONResponse:
        body = await req.json()
        name = body.get("model") or body.get("name") or ""
        try:
            friendly = resolve_model_name(name)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        r = await _start_download(friendly)
        return JSONResponse({"status": "success" if r.get("status") == "ready"
                             else "pulling manifest"})

    @app.delete("/api/delete")       # Ollama-compat delete
    async def api_delete(req: Request) -> JSONResponse:
        body = await req.json()
        name = body.get("model") or body.get("name") or ""
        try:
            friendly = resolve_model_name(name)
        except ValueError as exc:
            return JSONResponse({"error": str(exc)}, status_code=404)
        r = await _do_delete(friendly)
        return (JSONResponse({"status": "success"}) if r["ok"]
                else JSONResponse({"error": r["error"]}, status_code=409))
