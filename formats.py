#!/usr/bin/env python3
"""
InfiniteModel — pure format / helper functions for the controller (server-only leaf module).

Extracted from server.py (#38, step A) to shrink that file. These are SELF-CONTAINED helpers:
Ollama API formatting (tag/model-info), detokenization safety, and the Anthropic Messages API /
tool-calling / mRoPE / token-estimation helpers. None of them touch controller state (engine,
registry, MODELS, METRICS, app routes, …) — they take everything they need as arguments, use only
stdlib + ModelSpec.

This is a controller-only leaf module: it must NEVER ``import server`` (no back-import -> no import
cycle). It is listed in server.py's EXTRA_UPDATE_FILES so the multi-file self-update keeps it in
sync across the fleet, and server.py imports its symbols back via a convergence-bridge import.
"""
from __future__ import annotations

import contextlib
import hashlib
import json
import os
import re
import time
from datetime import datetime, timezone
from typing import Optional

from placement import ModelSpec


# ---------------------------------------------------------------------------
# Ollama-compatible helpers
# ---------------------------------------------------------------------------

def _iso(ts: Optional[float] = None) -> str:
    return datetime.fromtimestamp(ts if ts else time.time(), timezone.utc).isoformat()


def _digest(s: str) -> str:
    return "sha256:" + hashlib.sha256(s.encode()).hexdigest()


def _human_params(spec: ModelSpec) -> str:
    p = spec.param_count
    return f"{p/1e9:.1f}B" if p >= 1e9 else f"{p/1e6:.0f}M"


def _details(spec: ModelSpec) -> dict:
    return {"parent_model": "", "format": "safetensors", "family": spec.arch,
            "families": [spec.arch], "parameter_size": _human_params(spec),
            "quantization_level": "BF16"}


def _model_info(spec: ModelSpec) -> dict:
    a = spec.arch
    return {
        "general.architecture": a,
        "general.parameter_count": spec.param_count,
        "general.file_type": 32,  # bf16-ish marker
        f"{a}.context_length": spec.max_ctx,
        f"{a}.block_count": spec.num_layers,
        f"{a}.embedding_length": spec.hidden_size,
        f"{a}.feed_forward_length": spec.intermediate_size,
        f"{a}.attention.head_count": spec.num_heads,
        f"{a}.attention.head_count_kv": spec.num_kv_heads,
        f"{a}.attention.key_length": spec.head_dim,
        f"{a}.attention.value_length": spec.head_dim,
        f"{a}.vocab_size": spec.vocab_size,
        "tokenizer.ggml.model": "gpt2",
    }


def _to_id_list(enc) -> list[int]:
    """Coerce a tokenizer result (list, BatchEncoding/dict, tensor, or batched
    nested list) into a flat list[int]."""
    import torch
    if hasattr(enc, "input_ids"):
        enc = enc.input_ids
    elif isinstance(enc, dict):
        enc = enc["input_ids"]
    if isinstance(enc, torch.Tensor):
        enc = enc.tolist()
    if enc and isinstance(enc[0], (list, tuple)):
        enc = enc[0]
    return [int(x) for x in enc]


# ---------------------------------------------------------------------------
# Detokenization safety (#21)
# ---------------------------------------------------------------------------
_DECODE_WARNED = False


def _safe_decode(tok, ids) -> str:
    """Decode token ids to text, surviving ids the tokenizer can't map.

    This model's lm_head vocab can be WIDER than the text tokenizer (a multimodal
    head carries vision/audio placeholder ids), so a stray out-of-range id makes a
    plain ``tok.decode`` raise "list index out of range". We mask those ids at the
    sampler now, but keep this as belt-and-suspenders: on failure, log once (with the
    offending ids) and decode id-by-id, skipping anything out of range/undecodable."""
    global _DECODE_WARNED
    try:
        return tok.decode(ids, skip_special_tokens=True)
    except Exception as exc:
        try:
            ntok = len(tok)
        except Exception:
            ntok = int(getattr(tok, "vocab_size", 0) or 0)
        if not _DECODE_WARNED:
            _DECODE_WARNED = True
            bad = [i for i in ids if ntok and i >= ntok]
            print(f"[decode] {type(exc).__name__}: {exc} — recovering id-by-id; "
                  f"len(tok)={ntok} vocab_size={getattr(tok, 'vocab_size', '?')} "
                  f"out_of_range={bad[:8]} ids_tail={ids[-8:]}")
        out = []
        for i in ids:
            if ntok and i >= ntok:
                continue
            with contextlib.suppress(Exception):
                out.append(tok.decode([i], skip_special_tokens=True))
        return "".join(out)


def _harmony_final_text(tok, ids):
    """OpenAI-harmony (gpt-oss) channel filter -> the user-facing FINAL channel text only.

    A harmony assistant turn is a sequence of channels:
        <|channel|>analysis<|message|> {chain-of-thought} <|end|>
        <|start|>assistant<|channel|>final<|message|> {answer} <|return|>
    The `analysis` (and `commentary`) channels are model reasoning, NOT shown to the user; only the
    `final` channel is the answer. Markers are special tokens (stripped by skip_special_tokens), so we
    split at the TOKEN level: find the <|message|> that closes a `<|channel|>final` header and decode
    everything after it. Returns:
      - None  if this isn't a harmony stream (no <|channel|> token) -> caller does a plain decode;
      - ""    if the final channel hasn't started yet (so STREAMING suppresses the analysis channel);
      - the decoded final-channel answer otherwise.
    """
    try:
        chan = tok.convert_tokens_to_ids("<|channel|>")
        msg = tok.convert_tokens_to_ids("<|message|>")
    except Exception:
        return None
    unk = getattr(tok, "unk_token_id", None)
    if chan in (None, unk) or msg in (None, unk) or chan not in ids:
        return None
    n = len(ids)
    final_start = None
    i = 0
    while i < n:
        if ids[i] == chan:                       # a channel header: <|channel|> NAME <|message|>
            j = i + 1
            name = []
            while j < n and ids[j] != msg:
                name.append(ids[j])
                j += 1
            if j < n:                            # found the closing <|message|>
                with contextlib.suppress(Exception):
                    if tok.decode(name, skip_special_tokens=True).strip() == "final":
                        final_start = j + 1      # answer begins right after <|message|>
            i = j
        i += 1
    if final_start is None:
        return ""                                # final channel not reached yet
    return _safe_decode(tok, ids[final_start:])


def _decode_visible(tok, ids) -> str:
    """User-facing detok: harmony-aware (gpt-oss returns only the final channel, dropping the
    analysis CoT); otherwise a plain _safe_decode. Byte-identical for every non-harmony model."""
    h = _harmony_final_text(tok, ids)
    return h if h is not None else _safe_decode(tok, ids)


class IncrementalDetok:
    """#inc-detok: stateful replacement for per-token `_decode_visible(tok, produced)`.

    The old pattern re-decoded the FULL cumulative id list (and, for harmony, re-scanned every
    id for channel markers) on EVERY generated token — O(n) work per token, O(n^2) per
    generation, measured 6-9 ms/token at 16-32k-id thinking outputs, paid ON the event loop.

    Scheme: anchored suffix decode with VERIFIED folds. `head` caches the exact decode of
    ids[:anchor]; each step decodes only ids[anchor:] (the tail, usually a few tokens) and
    returns head + tail-decode. Every _FOLD_EVERY tokens one full decode runs and the anchor
    advances ONLY IF head_candidate + suffix == full — so the emitted text is byte-identical
    to the old full re-decode by construction: sentencepiece leading-space stripping and
    cleanup-spaces junction effects depend on the FIRST suffix token, which is pinned at the
    verified fold; a tokenizer that ever fails verification simply never folds and degrades
    to the old cost, never to wrong output. The harmony channel walk is a small state machine
    fed only NEW ids (same semantics as _harmony_final_text, including a LATER `final` header
    re-basing the visible text, which resets the anchor).

    Kill switch: INFINITEMODEL_INC_DETOK=0 delegates every call to the old full path."""

    _FOLD_EVERY = 64     # steps between fold attempts (1 full decode amortized over 64 tokens)
    _KEEP_TAIL = 8       # ids kept behind the anchor so the "�" holdback window never folds away

    def __init__(self, tok):
        self.tok = tok
        self.ids: list[int] = []
        self.n = 0                       # public token count (serving state["tokens"])
        self._full = os.environ.get("INFINITEMODEL_INC_DETOK", "1") in ("0", "false", "no")
        self._anchor = 0                 # ids[:anchor] are folded into _head
        self._head = ""                  # exact decode(ids[:_anchor]), fold-verified
        self._since_fold = 0
        # harmony state: _hmy None=unknown, False=plain model, True=harmony markers seen
        self._hmy = None
        self._chan = self._msg = None    # marker ids (resolved once, None if unresolvable)
        self._hdr: list[int] | None = None   # ids of a channel header still awaiting <|message|>
        self._final_start = None         # id index where the final channel's answer begins

    def _resolve_markers(self):
        if self._chan is None and self._hmy is None:
            try:
                chan = self.tok.convert_tokens_to_ids("<|channel|>")
                msg = self.tok.convert_tokens_to_ids("<|message|>")
                unk = getattr(self.tok, "unk_token_id", None)
                if chan in (None, unk) or msg in (None, unk):
                    self._hmy = False            # tokenizer has no harmony markers
                else:
                    self._chan, self._msg = chan, msg
            except Exception:
                self._hmy = False

    def _feed_harmony(self, tid: int):
        """Mirror _harmony_final_text's scan, one id at a time (only ever sees NEW ids)."""
        if self._hmy is False:
            return
        self._resolve_markers()
        if self._hmy is False:
            return
        if self._hdr is not None:                # inside a header: collecting the channel name
            if tid == self._msg:                 # header closed — is it the `final` channel?
                with contextlib.suppress(Exception):
                    if _safe_decode(self.tok, self._hdr).strip() == "final":
                        self._final_start = self.n          # answer begins AFTER this id
                        self._anchor, self._head, self._since_fold = self.n, "", 0
                self._hdr = None
            else:                                # NOTE: a nested <|channel|> is collected into the
                self._hdr.append(tid)            # name, exactly like _harmony_final_text's scan
                                                 # (it only breaks on <|message|>)
        elif tid == self._chan:
            self._hmy = True                     # first marker: this IS a harmony stream
            self._hdr = []

    def push(self, tid: int) -> str:
        """Append one produced id; return the CURRENT full visible text (old-path-identical)."""
        self.ids.append(tid)
        self.n += 1
        self._feed_harmony(tid)
        return self.current()

    def current(self) -> str:
        if self._full:                           # kill switch: exact old behavior
            return _decode_visible(self.tok, self.ids)
        if self._hmy:                            # harmony stream: only the final channel shows
            if self._final_start is None:
                return ""                        # final channel not reached yet (hides CoT)
            base = self._final_start
        else:
            base = 0
        lo = max(self._anchor, base)
        text = self._head + _safe_decode(self.tok, self.ids[lo:])
        self._since_fold += 1
        if self._since_fold >= self._FOLD_EVERY and self.n - lo > self._KEEP_TAIL:
            self._since_fold = 0
            new_anchor = self.n - self._KEEP_TAIL
            # one true full decode of the visible span, used BOTH to verify and to fold:
            full = _safe_decode(self.tok, self.ids[base:])
            suffix = _safe_decode(self.tok, self.ids[new_anchor:])
            if full == text and full.endswith(suffix) and not full[:len(full) - len(suffix)].endswith("�"):
                self._anchor = new_anchor
                self._head = full[:len(full) - len(suffix)]
            # verification failed -> keep the old anchor; output above is still the old-path
            # text ONLY if head+tail matched the full decode — if it didn't, repair now so a
            # mismatch can never be emitted (belt-and-suspenders; costs one already-done decode).
            elif full != text:
                self._anchor, self._head = base, ""
                text = full
        return text


# ---------------------------------------------------------------------------
# Anthropic Messages API helpers (so Claude Code can use the fleet as a backend)
# ---------------------------------------------------------------------------
_ID_CTR = 0
_TOOLCALL_RE = re.compile(r"<tool_call>\s*(\{.*?\})\s*</tool_call>", re.DOTALL)
_THINK_RE = re.compile(r"<think>.*?</think>", re.DOTALL)
_TOOL_OPEN = "<tool_call>"
_TOOL_CLOSE = "</tool_call>"
# A model not given native tool framing improvises the format, and qwen3.6-35b-a3b is
# wildly inconsistent — across runs it has emitted ALL of these for the same call:
#   Hermes JSON:   <tool_call>{"name": "f", "arguments": {...}}</tool_call>
#   Claude XML:    <invoke name="f"><parameter name="k">v</parameter></invoke>
#   hybrid XML:    <tool_call><function=f><parameter=k>v</parameter></function></tool_call>
# So parse them ALL liberally rather than passing a clear tool-call intent through as text.
_INVOKE_RE = re.compile(
    r"<invoke\b[^>]*?\bname\s*=\s*[\"']?([^\"'>\s]+)[\"']?[^>]*>(.*?)</invoke>", re.DOTALL)
# <function=name>..</function> or <function name="name">..</function> (the inner block of
# the hybrid form, and a bare form some runs emit without the <tool_call> wrapper).
_FUNC_RE = re.compile(
    r"<function\b(?:\s*=\s*|[^>]*?\bname\s*=\s*)[\"']?([^\"'>\s]+)[\"']?[^>]*>(.*?)</function>",
    re.DOTALL)
_PARAM_RE = re.compile(
    r"<parameter(?:\s+name\s*=\s*[\"']?([^\"'>\s]+)[\"']?|\s*=\s*([^>\s]+))\s*>(.*?)</parameter>",
    re.DOTALL)
_FUNCCALLS_RE = re.compile(r"</?function_calls>", re.IGNORECASE)
# Earliest of any of these in a stream means "a tool call is starting" — stop emitting
# plain text and buffer from here so the markup never leaks to the client as text.
_TOOL_OPENERS = ("<tool_call>", "<invoke", "<function_calls>", "<function")


def _parse_params(body: str) -> dict:
    """Pull <parameter name="k">v</parameter> / <parameter=k>v</parameter> pairs from an
    XML tool-call body. Values are JSON-typed when possible, else kept as trimmed strings."""
    args = {}
    for pm in _PARAM_RE.finditer(body):
        key = pm.group(1) or pm.group(2)
        if not key:
            continue
        raw = pm.group(3).strip()
        try:
            args[key] = json.loads(raw)
        except Exception:
            args[key] = raw
    return args


def _parse_tool_calls(text: str) -> list:
    """Find every tool call in `text`, in any of the formats above. Returns a list of
    {"name","arguments"} dicts (the shape _tool_to_block expects)."""
    calls: list = []
    for m in _TOOLCALL_RE.finditer(text):       # Hermes JSON
        with contextlib.suppress(Exception):
            calls.append(json.loads(m.group(1)))
    for m in _INVOKE_RE.finditer(text):         # Claude <invoke>
        calls.append({"name": m.group(1), "arguments": _parse_params(m.group(2))})
    for m in _FUNC_RE.finditer(text):           # <function=..> (incl. inside <tool_call>)
        calls.append({"name": m.group(1), "arguments": _parse_params(m.group(2))})
    return calls


def _strip_reasoning(text: str) -> str:
    """Remove <think>…</think> reasoning. Also handles reasoning models (Qwen3) whose
    template OPENS <think> in the prompt, so the generation begins mid-thought and only
    emits a dangling </think>: everything up to that first close is reasoning."""
    text = _THINK_RE.sub("", text)
    if "<think>" not in text and "</think>" in text:
        text = text.split("</think>", 1)[1]
    return text


def _tool_instruction(hf_tools) -> str:
    """A text block that lists the tools and the exact <tool_call> output format — injected
    into the prompt when the model's chat template can't render tools natively (e.g. a
    multimodal-remapped tokenizer whose template throws on `tools=`), so tools aren't lost."""
    lines = ["You can call tools. To call one, output EXACTLY (and nothing else for that call):",
             '<tool_call>{"name": "<tool_name>", "arguments": {<json arguments>}}</tool_call>',
             "Available tools:"]
    for t in (hf_tools or []):
        fn = t.get("function", {}) if isinstance(t, dict) else {}
        lines.append(f"- {fn.get('name')}: {fn.get('description', '')} "
                     f"parameters={json.dumps(fn.get('parameters', {}))}")
    return "\n".join(lines)


def _anth_id(prefix: str) -> str:
    global _ID_CTR
    _ID_CTR += 1
    h = hashlib.sha256(f"{prefix}{time.time()}{_ID_CTR}".encode()).hexdigest()[:24]
    return f"{prefix}_{h}"


def _anth_flatten(content) -> str:
    """Flatten an Anthropic content value (str | list of blocks) to plain text."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    parts = []
    for blk in content:
        if not isinstance(blk, dict):
            parts.append(str(blk))
            continue
        t = blk.get("type")
        if t == "text":
            parts.append(blk.get("text", ""))
        elif t == "tool_result":
            parts.append(_anth_flatten(blk.get("content")))
    return "".join(parts)


def _anthropic_messages_to_chat(system, messages, keep_images: bool = False,
                                keep_audio: bool = False) -> list:
    """Convert Anthropic system+messages into an HF chat-template message list.
    tool_use blocks -> assistant.tool_calls; tool_result blocks -> tool-role msgs.
    keep_images=True: a user message's images become {"type":"image"} content entries (in
    order) so the vision chat template emits one <|image_pad|> placeholder per image (#22
    inc 3b — the controller then expands each to its grid-derived count + splices embeds).
    keep_audio=True: likewise, audio clips become {"type":"audio"} entries so the Omni
    template emits <|audio_bos|><|AUDIO|><|audio_eos|> per clip (#22 inc 5c — expanded to
    its token count + spliced). keep_*=False (default): that modality flattens to a text
    marker (text-only behavior)."""
    chat: list = []
    sys_text = _anth_flatten(system).strip()
    if sys_text:
        chat.append({"role": "system", "content": sys_text})
    for m in (messages or []):
        role = m.get("role", "user")
        content = m.get("content")
        if role == "system":
            # Off-spec but seen in the wild: a system-role entry INSIDE messages[] (the
            # Anthropic API only defines the top-level `system` param). Strict templates
            # (qwen3.6: "System message must be at the beginning.") reject any system
            # message past index 0, so MERGE it into the leading system message instead
            # of passing it through (lenient templates tolerated the pass-through).
            _st = _anth_flatten(content).strip()
            if _st:
                if chat and chat[0].get("role") == "system":
                    chat[0]["content"] = chat[0]["content"] + "\n\n" + _st
                else:
                    chat.insert(0, {"role": "system", "content": _st})
            continue
        if isinstance(content, str):
            chat.append({"role": role, "content": content})
            continue
        if content is None:
            chat.append({"role": role, "content": ""})
            continue
        text_parts, tool_calls, tool_results, n_images, n_audio = [], [], [], 0, 0
        for blk in content:
            if not isinstance(blk, dict):
                text_parts.append(str(blk))
                continue
            t = blk.get("type")
            if t == "text":
                text_parts.append(blk.get("text", ""))
            elif t == "tool_use":
                # arguments as a DICT — the HF chat-template convention. Strict templates
                # (qwen3.6) iterate it as a mapping and raise "Can only get item pairs from
                # a mapping" on the old JSON-string form, which failed BOTH the native-tools
                # render AND the fallback, dropping the prompt to the last-ditch flat render
                # — which a chat-tuned model answers with an instant EOS (the empty
                # tool_result-turn symptom). tojson-style templates (qwen2.5) also render a
                # dict correctly (the string form double-encoded).
                tool_calls.append({"type": "function", "id": blk.get("id"),
                                   "function": {"name": blk.get("name"),
                                                "arguments": blk.get("input") or {}}})
            elif t == "tool_result":
                tool_results.append(_anth_flatten(blk.get("content")))
            elif t in ("thinking", "redacted_thinking"):
                # assistant reasoning echoed back in the conversation history (clients replay
                # the thinking blocks the serve path now returns) — never re-render reasoning
                # into the prompt; the template's own <think> handling owns that channel.
                continue
            elif t in ("image", "image_url"):
                if keep_images:
                    n_images += 1
                else:
                    text_parts.append("[image omitted: text-only model]")
            elif t in ("audio", "audio_url", "input_audio"):
                if keep_audio:
                    n_audio += 1
                else:
                    text_parts.append("[audio omitted: text-only model]")
        if role == "assistant":
            msg = {"role": "assistant", "content": "".join(text_parts)}
            if tool_calls:
                msg["tool_calls"] = tool_calls
            chat.append(msg)
        else:
            for rtext in tool_results:
                chat.append({"role": "tool", "content": rtext})
            txt = "".join(text_parts)
            if (keep_images and n_images) or (keep_audio and n_audio):
                # media entries FIRST (audio then image, in order), then the text -> the
                # template renders the per-clip/per-image placeholder markers, then text.
                # (the "audio" value is a placeholder; the actual waveform is processed
                # separately and its embeds are spliced at the <|AUDIO|> positions.)
                parts = [{"type": "audio", "audio": ""} for _ in range(n_audio)]
                parts += [{"type": "image"} for _ in range(n_images)]
                if txt:
                    parts.append({"type": "text", "text": txt})
                chat.append({"role": "user", "content": parts})
            elif txt.strip() or not tool_results:
                chat.append({"role": "user", "content": txt})
    return chat


def _expand_image_placeholders(ids, image_token_id, counts,
                               grid_rc=None, break_id=None, end_id=None):
    """The vision chat template emits ONE image_token (image_token_id) per image; the LM
    needs `counts[i]` of them for image i (= its merged-token count). Replace each single
    placeholder with a run of that many and record the absolute positions (which align, in
    order, with the rows of the encoder's image_embeds). Returns (new_ids, positions, found).

    #150 Pixtral/Mistral3 row structure: when grid_rc=[(H,W),...] AND break_id/end_id are given,
    image i's run is emitted with the layout the model was TRAINED with — [IMG]×W then a
    [IMG_BREAK] after each patch row, the LAST row's break replaced by [IMG_END] — instead of a
    flat [IMG]×(H·W) run. `positions` still lists ONLY the image_token_id (embed) slots, row-major
    and 1:1 with the encoder's image_embeds rows; the break/end tokens are ordinary vocab ids that
    keep their own learned embeddings (never spliced). Without grid_rc — or if a grid doesn't match
    its count (safety fallback) — the flat run is emitted, byte-identical to before (every other
    arch, and the audio path, are untouched)."""
    out: list[int] = []
    positions: list[int] = []
    ci = 0
    for tid in ids:
        if tid == image_token_id:
            c = counts[ci] if ci < len(counts) else 1
            rc = grid_rc[ci] if (grid_rc and ci < len(grid_rc)) else None
            ci += 1
            if (rc and break_id is not None and end_id is not None
                    and len(rc) == 2 and int(rc[0]) * int(rc[1]) == c and c > 0):
                H, W = int(rc[0]), int(rc[1])
                for r in range(H):
                    start = len(out)
                    out.extend([image_token_id] * W)
                    positions.extend(range(start, start + W))
                    out.append(int(end_id) if r == H - 1 else int(break_id))
            else:
                start = len(out)
                out.extend([image_token_id] * c)
                positions.extend(range(start, start + c))
        else:
            out.append(tid)
    return out, positions, ci


def _mrope_position_ids(ids, grid_list, image_token_id, merge):
    """#22 inc 4: compute Qwen3-VL 3D (t/h/w) mRoPE position ids for an EXPANDED prompt (one
    run of image_token_id per image). Faithful to transformers get_rope_index/get_vision_
    position_ids (validated against the reference): text tokens advance all 3 dims by 1; an
    image's tokens get t=start, h=start+row, w=start+col over its merged grid (h,w // merge,
    t // 1), and AFTER the image the counter advances by only max(h,w)//merge (positions
    'grow slowly'). The interleaving across freq bands is done by the worker's rotary.
    Returns (position_ids [3][seq] lists, base) where base = max position + 1 (decode start)."""
    t_row: list[int] = []
    h_row: list[int] = []
    w_row: list[int] = []
    cur = 0
    gi = 0
    i = 0
    n = len(ids)
    while i < n:
        if ids[i] == image_token_id:
            j = i
            while j < n and ids[j] == image_token_id:
                j += 1
            t, h, w = grid_list[gi] if gi < len(grid_list) else (1, merge, merge)
            gi += 1
            lt, lh, lw = int(t) // 1, int(h) // merge, int(w) // merge
            for ti in range(lt):
                for hi in range(lh):
                    for wi in range(lw):
                        t_row.append(cur + ti)   # time_interval=1
                        h_row.append(cur + hi)
                        w_row.append(cur + wi)
            cur += max(int(h), int(w)) // merge
            i = j
        else:
            j = i
            while j < n and ids[j] != image_token_id:
                j += 1
            for k in range(j - i):
                t_row.append(cur + k)
                h_row.append(cur + k)
                w_row.append(cur + k)
            cur += (j - i)
            i = j
    pos = [t_row, h_row, w_row]
    base = (max(t_row + h_row + w_row) + 1) if t_row else 0
    return pos, base


def _audio_position_ids(seq_len: int):
    """#22 inc 5c: 3D (t/h/w) TMRoPE position ids for an AUDIO-ONLY-plus-text prompt.
    Per Qwen2.5-Omni get_rope_index, the audio branch assigns each audio token
    `arange(audio_len) + st_idx` IDENTICALLY across t/h/w (no spatial split), with
    st_idx = prev_max + 1, while text/bos/eos advance +1 — so the WHOLE sequence is just
    sequential 0..seq-1 broadcast to all 3 dims (unlike images, audio positions do NOT grow
    slowly). Returns (position_ids [3][seq], base) where base = seq_len (decode start)."""
    row = list(range(seq_len))
    return [row, list(row), list(row)], seq_len


def _anthropic_tools_to_hf(tools):
    """Anthropic tool defs -> OpenAI/HF function-tool defs for apply_chat_template."""
    if not tools:
        return None
    out = []
    for t in tools:
        if not isinstance(t, dict):
            continue
        out.append({"type": "function", "function": {
            "name": t.get("name"),
            "description": t.get("description", ""),
            "parameters": t.get("input_schema") or {"type": "object", "properties": {}}}})
    return out or None


def _tool_to_block(tb: dict) -> dict:
    """A parsed <tool_call> JSON object -> an Anthropic tool_use content block."""
    args = tb.get("arguments")
    if not isinstance(args, dict):
        args = tb.get("parameters") if isinstance(tb.get("parameters"), dict) else {}
    return {"type": "tool_use", "id": _anth_id("toolu"),
            "name": tb.get("name"), "input": args}


def _extract_tools(text: str):
    """Split a full generation into (clean_text, [tool-call dicts]). Strips reasoning
    FIRST (so tool markup inside <think> isn't taken as a real call), then pulls out
    every tool call in any supported format."""
    no_think = _strip_reasoning(text)
    tools = _parse_tool_calls(no_think)
    clean = _TOOLCALL_RE.sub("", no_think)
    clean = _INVOKE_RE.sub("", clean)
    clean = _FUNC_RE.sub("", clean)
    clean = _FUNCCALLS_RE.sub("", clean)
    clean = clean.replace(_TOOL_OPEN, "").replace(_TOOL_CLOSE, "")   # leftover wrapper tags
    return clean.strip(), tools


def _partial_suffix_len(s: str, tag: str) -> int:
    """Longest suffix of s that is a proper prefix of tag — a possibly-incomplete
    opening tag we hold back rather than stream as plain text."""
    for k in range(min(len(s), len(tag) - 1), 0, -1):
        if s[-k:] == tag[:k]:
            return k
    return 0


def _split_reasoning(raw: str, starts_in_think: bool = False):
    """Split a COMPLETED generation into (visible, reasoning).

    `starts_in_think` means the chat template opened <think> in the PROMPT, so the model begins
    already mid-thought and emits reasoning, then a bare `</think>`, then the answer — with no
    opening tag anywhere in its output. Returning that raw is not merely untidy, it is malformed:
    the client receives an unpaired closing tag and cannot separate thought from answer.
    Budget exhaustion inside the thought (no `</think>` at all) yields ("", raw) — an empty answer
    is honest, whereas emitting the raw chain-of-thought as the answer is not.

    Without the flag, only COMPLETE <think>…</think> pairs the model opened itself are lifted out."""
    if starts_in_think:
        c = raw.find("</think>")
        if c == -1:
            return "", raw
        return raw[c + len("</think>"):].lstrip("\n"), raw[:c]
    think = "\n".join(m.group(0)[len("<think>"):-len("</think>")] for m in _THINK_RE.finditer(raw))
    return _THINK_RE.sub("", raw).lstrip("\n"), think


class _ReasonGate:
    """Streaming counterpart of _split_reasoning for the plain (no-tools) streamers.

    Buffers until the closing `</think>` arrives, then releases everything after it and passes
    subsequent pieces straight through. Only engaged when the prompt opened the thought; a model
    that opens its own <think> still streams as before, because that output is well-formed and
    some clients deliberately display it.

    Three states, not two. Releasing on `</think>` and stripping the blank lines after it in one
    step only works when both land in the same chunk — at one token per piece the closer arrives
    alone, the gate opens with nothing left to strip, and the following newlines stream through as
    the answer's first characters. So leading-newline suppression survives the release as its own
    state until real text appears. Found by testing the gate at every chunk size from 1 up, not by
    reading it."""

    __slots__ = ("active", "buf", "_lead")

    def __init__(self, active: bool):
        self.active = bool(active)
        self.buf = ""
        self._lead = bool(active)

    def feed(self, piece: str) -> str:
        if self.active:
            self.buf += piece
            c = self.buf.find("</think>")
            if c == -1:
                return ""
            piece = self.buf[c + len("</think>"):]
            self.active, self.buf = False, ""
        if self._lead:
            piece = piece.lstrip("\n")
            if not piece:
                return ""
            self._lead = False
        return piece


# Every piece of markup that changes how the raw stream is segmented. A boundary is only safe
# to declare "settled" once none of these can start inside it, straddle it, or be completed by
# whatever the model emits next.
_TAGS = _TOOL_OPENERS + ("<think>", "</think>")
_TAG_EDGE = max(len(t) for t in _TAGS) - 1     # longest partial tag that can straddle a boundary


def _hold_len(s: str) -> int:
    """Trailing chars of s that might be the START of any tag, so must not be emitted yet."""
    # Every tag begins with '<' and none is longer than _TAG_EDGE+1, so a partial tag must put a
    # '<' inside the last _TAG_EDGE chars. Checking that first skips six suffix scans on the token
    # that carries ordinary prose, which is nearly every token: this guard is most of the win.
    if "<" not in s[-_TAG_EDGE:]:
        return 0
    return max((_partial_suffix_len(s, t) for t in _TAGS), default=0)


def _first_tag(s: str):
    """(index, tag) of the earliest tag occurrence in s, or (-1, "") when s is markup-free."""
    best, which = -1, ""
    for t in _TAGS:
        i = s.find(t)
        if i != -1 and (best == -1 or i < best):
            best, which = i, t
    return best, which


def _segment_tools(raw: str, starts_in_think: bool = False):
    """Prefix-stable split of streamed raw text into (visible_plain, completed_tools).
    Reasoning is stripped and tool markup held back until complete, so neither leaks to
    the client as text. `starts_in_think` (the template opened <think> in the prompt, so
    the model begins mid-thought) holds EVERYTHING back until the closing </think>.
    Visible plain only ever grows; tools are every COMPLETE call after the first opener.

    This is the whole-string form, still used by the Anthropic SSE streamer. _ToolGate below
    is its incremental twin and MUST stay behaviourally identical to it — scratch_segment_tools_test.py
    is the equivalence proof, so run it after touching either one."""
    s = raw
    if starts_in_think:                     # began inside reasoning -> hold until it closes
        c = s.find("</think>")
        if c == -1:
            return "", []
        s = s[c + len("</think>"):]
    s = _THINK_RE.sub("", s)                # drop any finished <think>…</think> pairs
    ti = s.rfind("<think>")                 # unclosed reasoning -> hold back from it on
    if ti != -1 and "</think>" not in s[ti:]:
        s = s[:ti]
    hits = [s.find(o) for o in _TOOL_OPENERS]
    hits = [i for i in hits if i != -1]
    if not hits:                            # no opener yet — stream plain, hold any partial
        hold = max((_partial_suffix_len(s, o) for o in _TOOL_OPENERS + ("<think>", "</think>")),
                   default=0)
        return s[:len(s) - hold], []
    cut = min(hits)                         # stable: an opener's position doesn't move
    return s[:cut], _parse_tool_calls(s[cut:])


class _ToolGate:
    """Incremental form of _segment_tools for the tool-aware streamers.

    _segment_tools is a pure function of the WHOLE generation so far, and the OpenAI/Ollama tool
    streamers called it once per token: a regex sub, an rfind and six find/partial-suffix scans
    over every character emitted so far, again for the next token, and again for the one after —
    O(N^2) in the answer length. That is only reachable when the request sent tools, i.e. exactly
    the coding-agent path, which is also where answers are longest. This class produces the same
    segmentation while only ever looking at the tail that can still change.

    WHY A TAIL IS ENOUGH. Text is *settled* once nothing the model appends can rewrite it: it
    holds no <think>/</think> and no tool opener, and no tag can start inside it and finish later
    (guaranteed by only ever settling up to `len(x) - _hold_len(x)` — if a longer partial tag
    reached further back, those chars would not have been settled). Under that condition no
    <think>…</think> match, no truncation point and no opener can fall in the settled part, so
    segment(whole) == settled + segment(tail) for every future extension. Complete <think>…</think>
    pairs are folded away as they close — the same first-opener/first-closer pairing re.sub does —
    so a model that opens its own reasoning block does not pin the boundary at character 0. The
    fold is refused when the text in front of the pair ends in something that merely LOOKS like the
    start of a tag, because splicing it onto what follows the pair could manufacture markup the
    model never emitted; that costs a re-scan, never correctness.

    feed() returns the DELTA to send, not the whole visible string. The high-water-mark logic
    ("emit only what is past the furthest we have already emitted") used to live in each streamer
    and is folded in here, so the emitted byte stream is unchanged even in the pathological case
    where a nested <think> makes the visible text SHRINK. Doing it any other way would reintroduce
    an O(N) slice per token and undo the point of the exercise. `.plain` reconstructs what
    _segment_tools would have returned; it exists for the equivalence test and for debugging.

    ONE THING IS DELIBERATELY NOT IDENTICAL, and only in `.plain`, never in the emitted bytes.
    _segment_tools' "visible plain only ever grows" is not actually true: on "x<<think>abc" it
    returns "x<" at "x<<" — streaming that "<" — and then RETRACTS to "x" once the <think>
    completes and truncation re-exposes the "<" as a partial tag. Text already settled here cannot
    be taken back, which is the honest behaviour: the byte was sent. The fuzz in
    scratch_segment_tools_test.py hits this 341 times and asserts the client-visible stream is
    identical through it.

    Memory drops too: the streamers no longer accumulate the raw generation, and what is kept here
    is the visible text only — reasoning and tool markup are dropped as they settle."""

    __slots__ = ("_wait", "_hold", "_buf", "_parts", "_edge", "_pre_len",
                 "_pending", "_tail", "_emitted")

    def __init__(self, starts_in_think: bool = False):
        self._wait = bool(starts_in_think)   # holding everything until the prompt's <think> closes
        self._hold = ""                      # raw seen while waiting for that </think>
        self._buf = ""                       # unsettled tail: every char that can still change
        self._parts: list = []               # settled visible text, in arrival order
        self._edge = ""                      # last _TAG_EDGE chars of the settled text
        self._pre_len = 0                    # total length of the settled visible text
        self._pending = ""                   # settled this feed, not yet handed to the caller
        self._tail = ""                      # visible part of _buf as of the last feed
        self._emitted = 0                    # high-water mark of visible text handed out

    @property
    def plain(self) -> str:
        """The full visible text _segment_tools would return right now. O(N) — tests/diagnostics."""
        return "".join(self._parts) + self._tail

    def feed(self, piece: str):
        """Absorb the next raw piece; return (new_visible_text, completed_tool_calls)."""
        if self._wait:
            # A </think> split across two pieces is missed by searching only the new piece, and
            # re-searching the whole hold every time is the very cost being removed: restart the
            # search len("</think>")-1 chars back from the old end instead.
            start = max(0, len(self._hold) - 7)
            self._hold += piece
            c = self._hold.find("</think>", start)
            if c == -1:
                return "", []
            self._buf, self._hold, self._wait = self._hold[c + 8:], "", False
        else:
            self._buf += piece
        self._settle()
        return self._emit()

    def _edge_of(self, s: str) -> str:
        """Last _TAG_EDGE chars of (settled + s) — the only window a partial tag can occupy."""
        return s[-_TAG_EDGE:] if len(s) >= _TAG_EDGE else (self._edge + s)[-_TAG_EDGE:]

    def _push(self, n: int) -> None:
        """Move n chars off the front of the unsettled tail into the settled visible text."""
        if n <= 0:
            return
        seg, self._buf = self._buf[:n], self._buf[n:]
        self._parts.append(seg)
        self._pending += seg
        self._pre_len += n
        self._edge = (self._edge + seg)[-_TAG_EDGE:]

    def _settle(self) -> None:
        """Advance the settled boundary as far right as it provably goes."""
        while True:
            b = self._buf
            i, tag = _first_tag(b)
            if i == -1:                      # nothing but plain text in flight
                self._push(len(b) - _hold_len(self._edge_of(b)))
                return
            h = _hold_len(self._edge_of(b[max(0, i - _TAG_EDGE):i]))
            j = b.find("</think>", i + 7) if tag == "<think>" else -1
            if j == -1 or h:
                # A live boundary the core has to re-examine on every token: an unclosed <think>,
                # a stray closer, or a tool opener (h != 0 = the refused fold described above).
                self._push(i - h)
                return
            self._push(i)                    # text in front of the pair is plain and settled
            self._buf = self._buf[j + 8 - i:]   # the pair itself: exactly what re.sub would delete

    def _emit(self):
        """_segment_tools' body, run over the unsettled tail only, then diffed against the mark."""
        if not self._buf:
            # _settle drained the tail, which is what happens on a token of ordinary prose: there
            # is provably no markup left to look for, so skip the regex sub and the five scans.
            self._tail, tools = "", []
        else:
            s = _THINK_RE.sub("", self._buf)
            ti = s.rfind("<think>")
            if ti != -1 and "</think>" not in s[ti:]:
                s = s[:ti]
            hits = [s.find(o) for o in _TOOL_OPENERS]
            hits = [k for k in hits if k != -1]
            if hits:
                self._tail, tools = s[:min(hits)], _parse_tool_calls(s[min(hits):])
            else:
                # _hold_len sees the settled edge too, so a partial tag straddling the boundary is
                # caught; it can never reach further back than the tail, because a straddle that
                # long would have kept those chars unsettled in the first place.
                self._tail, tools = s[:len(s) - _hold_len(self._edge_of(s))], []
        base = self._pre_len - len(self._pending)   # index in the visible text of _pending[0]
        out = (self._pending + self._tail)[self._emitted - base:]
        self._pending = ""
        if not out:                          # visible text hasn't passed the high-water mark
            return "", tools
        self._emitted = self._pre_len + len(self._tail)
        return out, tools


def _estimate_tokens(chat: list) -> int:
    chars = sum(len(m.get("content", "") or "") for m in chat)
    return max(1, chars // 4)
