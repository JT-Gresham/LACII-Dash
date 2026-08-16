"""Standalone GGUF -> HF safetensors converter.

GGUF (llama.cpp's format) packs weights in k-quant / i-quant block layouts that the GGML
tensor library runs — not transformers/PyTorch. Rather than port those kernels (huge, and
pointless: we already have a fast int4 path), InfiniteModel NORMALIZES a GGUF model to a
standard HuggingFace safetensors checkpoint ONCE at add/download time. After that it is an
ordinary model in the system: chunk-streamed to workers, int4/int8 shard-cached, and run on the
distributed pipeline — no GGUF awareness anywhere downstream. This mirrors how fp8/nvfp4 source
checkpoints are handled (dequantize to bf16, then re-quantize to our int4 for serving).

Run as a SUBPROCESS by the controller (model_store.convert_gguf_to_model_dir) so a big
`from_pretrained` (which fully materializes the model in RAM) can OOM the SUBPROCESS without
taking down the controller box it co-hosts. Usage:

    python gguf_convert.py <repo_id> <gguf_file> <dst_dir>

`<gguf_file>` may be a single-file quant OR any part of a SPLIT set ("...-00001-of-00005.gguf"):
the whole set is fetched and converted as one model (#gguf-split — see _split_part_names, and
_verify_fully_loaded for the guard that stops a partially-read set from being saved).

The HF token (if any) is read from the HF_TOKEN env var (never a CLI arg — process listings leak
args). Prints a final ``GGUF_CONVERT_OK <dst_dir>`` line on success; exits non-zero on failure.
"""
import os
import re
import sys
import subprocess


def _ensure_deps() -> None:
    """transformers' GGUF loader needs `gguf` (to parse the file) AND `accelerate` (it loads via the
    low-memory/device_map path). Auto-install whatever's missing on demand (the m4c84 worker pattern)
    so a controller env with torch+transformers but not these optional extras can still convert
    without a manual pip step on the (SSH-less) box."""
    need = []
    try:
        import gguf  # noqa: F401
    except Exception:
        need.append("gguf")
    try:
        import accelerate  # noqa: F401
    except Exception:
        need.append("accelerate")
    # Building the model's tokenizer from a GGUF (then saving it as a FAST tokenizer.json so the
    # controller loads it without a slow->fast conversion at serve time) needs sentencepiece/tiktoken,
    # and sentencepiece's converter needs protobuf. Without these the save leaves only a slow tokenizer
    # and the LATER load fails ("need sentencepiece or tiktoken to convert a slow tokenizer to a fast one").
    try:
        import sentencepiece  # noqa: F401
    except Exception:
        need.append("sentencepiece")
    try:
        import tiktoken  # noqa: F401
    except Exception:
        need.append("tiktoken")
    try:
        import google.protobuf  # noqa: F401
    except Exception:
        need.append("protobuf")
    if need:
        print(f"[gguf-convert] installing missing deps: {', '.join(need)}", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "--quiet", *need], check=False)


# --- #gguf-split -------------------------------------------------------------------------------
# llama.cpp's `gguf-split` names its parts "<base>-00001-of-0000N.gguf" (llama_split_path:
# "%s-%05d-of-%05d.gguf"), so BOTH the part index and the total live in the filename — the full set
# is derivable from any one part with no listing call.
#
# ⚠ The parts are NOT a byte-split of one GGUF. Every part has its own header, and only part 1
# carries the full KV metadata (arch, hyper-parameters, tokenizer); parts 2..N are a stub header
# plus their share of the tensors. `cat part1 part2 > merged.gguf` produces a file that OPENS
# (part 1's header sits at offset 0) and is silently truncated garbage. Never join them bytewise.
# We hand the reader part 1 with its siblings physically beside it in the same cache directory,
# and then PROVE nothing was left unread (_verify_fully_loaded).
#
# Deliberately duplicated from model_store.gguf_split_info/gguf_part_names rather than imported:
# this script's whole reason to exist is that it runs as an isolated subprocess and never pulls in
# the controller's module graph (so its OOM stays its own). The format is pinned by llama.cpp, so
# the copies cannot drift on their own — but if one is edited, edit both.
_SPLIT_RE = re.compile(r"^(?P<base>.+)-(?P<idx>\d{5})-of-(?P<tot>\d{5})\.(?P<ext>gguf)$", re.I)


def _split_part_names(part: str):
    """(ordered part list, index of `part`) for a split GGUF; ([part], 1) for a single file.
    An out-of-range series (-00000-of-00003, -00004-of-00003) is NOT interpreted — it falls through
    as a single file, where the reader's own error is a better answer than a guessed part set."""
    m = _SPLIT_RE.match((part or "").strip())
    if not m:
        return [part], 1
    idx, tot = int(m.group("idx")), int(m.group("tot"))
    if tot < 1 or not (1 <= idx <= tot):
        return [part], 1
    base, ext = m.group("base"), m.group("ext")       # keep the extension's CASE: a .GGUF repo
    return [f"{base}-{i:05d}-of-{tot:05d}.{ext}" for i in range(1, tot + 1)], idx


def _gguf_inventory(paths):
    """Tensor inventory across the downloaded parts, read from the GGUF INDEXES only (mmap; the
    tensor data is never faulted in). Returns (n_tensors, n_params) or None if the `gguf` package
    is unavailable — the inventory is diagnostics plus two hard checks a filename cannot make:

      * a part that is not a readable GGUF at all — a truncated pull, or an HTML error page saved
        under the right name (hf_hub_download has produced both), and
      * a tensor NAME present in two parts — parts of one split set partition the tensors, so an
        overlap means these files are not one set (two quants' parts collided) and converting them
        would mix weight layouts.

    Raises ValueError on either; the caller turns that into a refusal."""
    try:
        from gguf import GGUFReader
    except Exception as exc:                          # deps install failed / very old gguf
        print(f"[gguf-convert] tensor inventory skipped ({exc!r}) — the load-coverage check below "
              "is the guard that matters", flush=True)
        return None
    seen, n_params = {}, 0
    for p in paths:
        try:
            rd = GGUFReader(p)
            tensors = list(rd.tensors)
        except Exception as exc:
            raise ValueError(f"{os.path.basename(p)} is not a readable GGUF ({exc!r}) — the "
                             f"download is corrupt; delete it from the HF cache and retry") from exc
        if not tensors:
            raise ValueError(f"{os.path.basename(p)} contains no tensors")
        for t in tensors:
            if t.name in seen:
                raise ValueError(
                    f"tensor {t.name!r} appears in BOTH {seen[t.name]} and "
                    f"{os.path.basename(p)} — these parts are not one split set")
            seen[t.name] = os.path.basename(p)
            n = getattr(t, "n_elements", None)
            if n is None:
                n = 1
                for d in t.shape:
                    n *= int(d)
            n_params += int(n)
        del rd, tensors                               # drop the mmap before the big load
    return len(seen), n_params


def _verify_fully_loaded(info, config, n_parts: int) -> list:
    """Return the NON-benign keys transformers reported it could not fill from the checkpoint.

    This is the whole safety story for split GGUF. If the installed transformers reads only part 1
    of N (older versions had no sharded-GGUF support at all), it does NOT raise: it builds the full
    architecture, fills what part 1 had, and RANDOMLY INITIALIZES every later layer. That model
    saves cleanly, is the right size, passes any structural check — and generates garbage. The one
    signal that separates it from a good load is `missing_keys`.

    Two families of missing key are genuinely benign and must not trip it:
      * RoPE `inv_freq` — a non-persistent buffer computed at __init__, in no checkpoint anywhere;
      * `lm_head.weight` under tie_word_embeddings — tied to the embedding AFTER load, so it is
        legitimately absent from the source."""
    out = []
    for k in (info or {}).get("missing_keys") or []:
        if k.endswith("inv_freq") or ".rotary_emb." in k:
            continue
        if k in ("lm_head.weight", "model.lm_head.weight") and getattr(
                config, "tie_word_embeddings", False):
            continue
        out.append(k)
    return out


def main() -> int:
    if len(sys.argv) < 4:
        print("usage: gguf_convert.py <repo_id> <gguf_file> <dst_dir>", file=sys.stderr)
        return 2
    repo_id, gguf_file, dst = sys.argv[1], sys.argv[2], sys.argv[3]
    token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None

    # #gguf-split: the controller always registers part 1, but this script is also documented as
    # runnable by hand — accept any part and open the set at part 1 (only it has the metadata).
    parts, named_idx = _split_part_names(gguf_file)
    n_parts = len(parts)
    if n_parts > 1 and named_idx != 1:
        print(f"[gguf-convert] {gguf_file} is part {named_idx} of {n_parts} — opening the set at "
              f"{parts[0]} (only part 1 carries the model metadata)", flush=True)

    _ensure_deps()
    import torch
    from huggingface_hub import hf_hub_download
    from transformers import AutoModelForCausalLM, AutoTokenizer

    kw = {"token": token} if token else {}
    paths = []
    for i, p in enumerate(parts, 1):
        print(f"[gguf-convert] downloading {p} from {repo_id}"
              + (f" [{i}/{n_parts}]" if n_parts > 1 else ""), flush=True)
        try:
            paths.append(hf_hub_download(repo_id, p, **kw))
        except Exception as exc:
            if n_parts == 1:
                raise
            # The set is derived from the name, so a 404 here means the repo does not actually hold
            # the whole series. Refuse: the alternative is dequantizing a model missing whole layers.
            print(f"[gguf-convert] ERROR: split part {p} ({i} of {n_parts}) is not in {repo_id} "
                  f"({exc!r}) — the set is INCOMPLETE and will not be converted", file=sys.stderr)
            return 5
    src_dirs = {os.path.dirname(p) for p in paths}
    if len(src_dirs) != 1:
        # The reader finds the siblings BY DIRECTORY. Parts landing in two snapshot dirs means the
        # repo was re-committed mid-pull, so the set on disk mixes revisions — refuse, don't mix.
        print(f"[gguf-convert] ERROR: split parts landed in different cache snapshots "
              f"({sorted(src_dirs)}) — the repo changed revision mid-download; retry",
              file=sys.stderr)
        return 6
    src_dir, fn = os.path.dirname(paths[0]), os.path.basename(paths[0])

    if n_parts > 1:
        try:
            inv = _gguf_inventory(paths)
        except ValueError as exc:
            print(f"[gguf-convert] ERROR: {exc}", file=sys.stderr)
            return 7
        if inv:
            n_tensors, n_params = inv
            print(f"[gguf-convert] split set: {n_parts} parts, {n_tensors} tensors, "
                  f"{n_params / 1e9:.2f}B parameters -> ~{n_params * 2 / 1024 ** 3:.1f} GiB of RAM "
                  f"once dequantized to bf16 (this load is NOT streamed)", flush=True)

    print(f"[gguf-convert] dequantizing {fn}"
          + (f" (+{n_parts - 1} parts)" if n_parts > 1 else "")
          + " -> bf16 (transformers GGUF loader)", flush=True)
    # output_loading_info gives us `missing_keys` — the ONLY thing that distinguishes a full read
    # of an N-part set from a part-1-only read that silently random-inits the rest. Requested for
    # both paths; only SPLIT treats it as a gate (see below).
    info = None
    try:
        model, info = AutoModelForCausalLM.from_pretrained(
            src_dir, gguf_file=fn, dtype=torch.bfloat16, output_loading_info=True)
    except TypeError as exc:
        if "output_loading_info" not in str(exc):
            raise
        if n_parts > 1:
            print(f"[gguf-convert] ERROR: this transformers does not accept output_loading_info "
                  f"({exc}), so a split-GGUF read CANNOT be proven to have covered all {n_parts} "
                  "parts. Refusing rather than saving a checkpoint whose later layers may be "
                  "randomly initialized. Upgrade transformers, or name a single-file quant.",
                  file=sys.stderr)
            return 8
        model = AutoModelForCausalLM.from_pretrained(src_dir, gguf_file=fn, dtype=torch.bfloat16)

    unloaded = _verify_fully_loaded(info, model.config, n_parts)
    if unloaded and n_parts > 1:
        print(f"[gguf-convert] ERROR: {len(unloaded)} tensors were NOT read from the {n_parts}-part "
              f"GGUF set and would have been saved randomly initialized: "
              f"{', '.join(unloaded[:12])}{' …' if len(unloaded) > 12 else ''}. This is what a "
              "transformers without sharded-GGUF support looks like — it reads part 1 and invents "
              "the rest. Refusing. Upgrade transformers, or name a single-file quant.",
              file=sys.stderr)
        return 9
    if unloaded:
        # Single-file path: warn, do not gate. This path ships and works today, and a new hard
        # refusal here would reject conversions that currently produce good models — the failure
        # mode above (a silently partial SET) does not exist for one file.
        print(f"[gguf-convert] WARNING: {len(unloaded)} tensors were not in the checkpoint and are "
              f"randomly initialized: {', '.join(unloaded[:12])}"
              f"{' …' if len(unloaded) > 12 else ''}", flush=True)
    elif info is not None:
        print(f"[gguf-convert] load verified: 0 unloaded tensors across {n_parts} "
              f"part{'s' if n_parts > 1 else ''}", flush=True)
    # transformers may dequantize to fp32 regardless of dtype on some versions; force bf16 so the
    # saved checkpoint is the size our planner/streamer expects (we re-quantize to int4 anyway).
    model = model.to(torch.bfloat16)

    os.makedirs(dst, exist_ok=True)
    print(f"[gguf-convert] saving safetensors -> {dst}", flush=True)
    model.save_pretrained(dst, safe_serialization=True)

    # part 1's basename: the tokenizer lives in the split set's metadata, which only part 1 carries.
    if not _save_tokenizer(src_dir, fn, repo_id, dst, token):
        print("[gguf-convert] ERROR: could not produce a serve-loadable tokenizer "
              "(GGUF-embedded slow tokenizer needs sentencepiece/tiktoken to convert, and the base "
              "repo had none) — model saved but unusable; aborting", file=sys.stderr)
        return 4

    print(f"GGUF_CONVERT_OK {dst}", flush=True)
    return 0


def _save_tokenizer(src_dir: str, gguf_file: str, repo_id: str, dst: str, token) -> bool:
    """Produce a tokenizer in `dst` that the controller can load at SERVE time WITHOUT a slow->fast
    conversion (which needs sentencepiece/tiktoken — C/Rust extensions that may have no wheel on a
    bleeding-edge Python). Strategy, each VERIFIED by reloading from `dst`:
      1) the GGUF-embedded tokenizer (works when the slow->fast deps are installable), then
      2) the base model's native (already-fast) tokenizer — most GGUF repos are '<base>-GGUF', and the
         base repo ships a fast tokenizer.json that loads with no extra deps.
    Returns True if a reload-verified tokenizer was saved."""
    import re
    import shutil
    from transformers import AutoTokenizer

    def _try(make, why) -> bool:
        try:
            tok = make()
            tok.save_pretrained(dst)
            # CRITICAL: the controller is a long-running process that cached "sentencepiece/tiktoken
            # unavailable" at startup (this subprocess pip-installed them AFTER), so it can NOT convert
            # a slow tokenizer to fast at serve time. Require a fast `tokenizer.json` (loads purely via
            # the `tokenizers` Rust lib, which transformers always has) so the serve-time load needs no
            # conversion deps. A slow-only save would "verify" HERE (this subprocess has the deps) yet
            # fail on the controller — reject it so we fall through to the base repo's native fast one.
            if not os.path.exists(os.path.join(dst, "tokenizer.json")):
                raise RuntimeError("save produced no fast tokenizer.json (slow-only tokenizer)")
            AutoTokenizer.from_pretrained(dst)   # sanity: reloads
            print(f"[gguf-convert] tokenizer: {why} (fast tokenizer.json, verified)", flush=True)
            return True
        except Exception as exc:
            print(f"[gguf-convert] tokenizer via {why} failed: {exc!r}", flush=True)
            # wipe a partial/slow-only tokenizer so the next attempt (or the load) isn't fooled
            for f in os.listdir(dst):
                if "token" in f.lower() or f in ("vocab.json", "merges.txt", "special_tokens_map.json"):
                    with __import__("contextlib").suppress(Exception):
                        os.remove(os.path.join(dst, f))
            return False

    kw = {"token": token} if token else {}
    if _try(lambda: AutoTokenizer.from_pretrained(src_dir, gguf_file=gguf_file), "GGUF-embedded"):
        return True
    base = re.sub(r"[-_.]?gguf$", "", repo_id, flags=re.I)
    if base and base != repo_id:
        if _try(lambda: AutoTokenizer.from_pretrained(base, **kw), f"base repo {base}"):
            return True
    return False


if __name__ == "__main__":
    sys.exit(main())
