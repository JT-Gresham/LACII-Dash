#!/usr/bin/env python3
"""Equivalence + cost proof for formats._ToolGate against formats._segment_tools.

WHY THIS EXISTS. The tool-aware streamers in serving.py used to call _segment_tools(raw, ...) on
the WHOLE accumulated generation once per token. _segment_tools does a regex sub, an rfind and six
find/partial-suffix scans over everything it is handed, so the per-token cost grew with the answer:
O(N^2), and only on the path a request with `tools` takes — which is the coding-agent path, where
answers are longest. _ToolGate does the same segmentation incrementally, re-scanning only the tail
that can still change.

That is an optimisation, so the bar is not "looks right" but "byte-identical". This runs BOTH
implementations over the same stream, at several chunk sizes (a token boundary can land anywhere,
including inside a tag), and asserts at EVERY step that:

  * the visible plain text matches exactly,
  * the completed tool-call list matches exactly, and
  * the bytes a client would actually receive match exactly — the streamers' high-water-mark
    delta logic moved INTO the gate, so it has to be checked rather than assumed. This is the one
    that catches the pathological case where nested <think> tags make the visible text SHRINK:
    a naive "just concatenate the deltas" gate diverges there and the first two checks do not see it.

The corpus is hand-written adversarial cases plus a fuzz over an alphabet of tag fragments, so tags
get split across chunk boundaries in every way that matters. Finally it measures the cost curve, to
show the O(N^2) is actually gone rather than merely reorganised.

Pure CPU, no model, no fleet: run it anywhere with `python3 scratch_segment_tools_test.py`.
"""
from __future__ import annotations

import random
import sys
import time

from formats import _ToolGate, _segment_tools


def chunks(raw: str, k: int):
    """Feed `raw` in k-char pieces (k=0 -> one piece), the way a token stream arrives."""
    if k <= 0:
        return [raw] if raw else []
    return [raw[i:i + k] for i in range(0, len(raw), k)]


def old_run(pieces, sit: bool):
    """What serving.py did: re-segment the whole accumulated string every token, and emit the
    slice past the high-water mark. Verbatim from the two streamers' bodies."""
    raw, emitted_plain, steps = "", 0, []
    for p in pieces:
        raw += p
        plain, tools = _segment_tools(raw, sit)
        delta = ""
        if len(plain) > emitted_plain:
            delta, emitted_plain = plain[emitted_plain:], len(plain)
        steps.append((plain, tools, delta))
    return steps


def new_run(pieces, sit: bool):
    """What serving.py does now: one gate, fed pieces, handing back the delta directly."""
    gate, steps = _ToolGate(sit), []
    for p in pieces:
        delta, tools = gate.feed(p)
        steps.append((gate.plain, tools, delta))
    return steps


def compare(raw: str, sit: bool, k: int) -> int:
    """Assert the two implementations agree at every step. Returns how many steps hit the one
    documented divergence (see below), so the caller can report that it is being exercised."""
    old, new = old_run(chunks(raw, k), sit), new_run(chunks(raw, k), sit)
    assert len(old) == len(new), f"step count {len(old)} != {len(new)}"
    retracted, mark = 0, 0
    for n, (o, w) in enumerate(zip(old, new)):
        ctx = f"\n  raw={raw!r}\n  starts_in_think={sit} chunk={k} step={n}"
        assert o[1] == w[1], f"TOOLS differ: {o[1]!r} != {w[1]!r}{ctx}"
        # The contract that actually reaches a client: the bytes sent, in order. Checked FIRST
        # because it is the one that must never differ for any input whatsoever.
        assert o[2] == w[2], f"EMITTED delta differs: {o[2]!r} != {w[2]!r}{ctx}"
        mark = max(mark, len(o[0]))
        if o[0] == w[0]:
            continue
        # THE ONE DOCUMENTED DIVERGENCE. _segment_tools' docstring claims the visible text is
        # prefix-stable; it is not. On e.g. "x<<think>abc" it returns "x<" at "x<<" — streaming
        # that "<" to the client — and then RETRACTS it to "x" once the <think> completes and the
        # truncation re-exposes the "<" as a partial tag. A stream cannot un-send a byte, so the
        # gate keeps the character it already handed over. The delta assertion above proves the
        # client sees the same bytes either way; all that differs is the bookkeeping value.
        assert len(o[0]) < mark, f"PLAIN differs without a retraction: {o[0]!r} != {w[0]!r}{ctx}"
        assert w[0].startswith(o[0]), f"PLAIN is not an extension: {o[0]!r} vs {w[0]!r}{ctx}"
        assert len(w[0]) <= mark, f"PLAIN ran past what was ever sent: {w[0]!r}{ctx}"
        retracted += 1
    return retracted


# The hand-written half. Each entry is a full generation; the point of most of them is a tag that
# a chunk boundary can be dropped into, or a shape that makes the visible text move backwards.
HERMES = '<tool_call>{"name": "read", "arguments": {"path": "/etc/hosts"}}</tool_call>'
XML = '<invoke name="write"><parameter name="path">/tmp/x</parameter></invoke>'
HYBRID = '<tool_call><function=grep><parameter=pat>needle</parameter></function></tool_call>'
CASES = [
    "",
    "plain text with no markup at all",
    "a < b and c <= d, in code, which is not a tag",       # bare '<' must not stall the boundary
    "trailing partial <fun",
    "trailing partial <think",
    "trailing partial <",
    "x<<think>abc",                                        # partial in front of a real tag
    "a<<think>x</think>b",                                 # refused fold: '<' abuts the pair
    "<think>reasoning</think>the answer",
    "before<think>mid</think>after<think>second</think>end",
    "a<think>b<think>c</think>",                           # nested: visible text SHRINKS here
    "a<think>b<think>c",                                   # two unclosed openers
    "stray </think> closer with no opener",
    "<think>unterminated reasoning that never closes",
    "answer text " + HERMES,
    "answer text " + HERMES + " tail after the call",
    HERMES + HERMES,
    "answer " + XML,
    "answer " + HYBRID,
    XML + HERMES,                                          # ordering: parser groups BY FORMAT
    "<function_calls>" + XML + "</function_calls>",
    "<think>plan</think>prose " + HERMES,
    "prose <function name=\"f\">body</function> tail",
    "reasoning</think>answer " + HERMES,                   # the starts_in_think shape
    "no closer here at all, so a starts_in_think stream stays silent",
    "}}}>>><<<",
]

# Fuzz alphabet: whole tags, half tags, and the characters that terminate them. Assembling raw
# strings from these produces the malformed, interleaved and truncated markup real models emit.
FRAGS = ["a", "bb", " ", "\n", "<", "<t", "<th", "think>", "<think>", "</think>", "</thi",
         "<tool_call>", "</tool_call>", '{"name": "f", "arguments": {"x": 1}}', "<invoke",
         '<invoke name="g">', "</invoke>", '<parameter name="k">v</parameter>', "<function",
         "<function=h>", "</function>", "<function_calls>", "</function_calls>", ">", "}", "<f"]

CHUNKS = (1, 2, 3, 5, 8, 13, 0)   # 0 = the whole generation in one piece


def main() -> int:
    n = r = 0
    for raw in CASES:
        for sit in (False, True):
            for k in CHUNKS:
                r += compare(raw, sit, k)
                n += 1
    print(f"hand-written: {len(CASES)} generations x 2 x {len(CHUNKS)} chunk sizes = {n} runs OK")

    rng = random.Random(20260816)
    for _ in range(3000):
        raw = "".join(rng.choice(FRAGS) for _ in range(rng.randint(1, 14)))
        r += compare(raw, rng.random() < 0.5, rng.choice(CHUNKS))
        n += 1
    print(f"fuzz: 3000 random generations over {len(FRAGS)} tag fragments OK ({n} runs total)")
    print(f"steps where the OLD plain retracted already-sent text (bytes still identical): {r}")

    # Cost curve. The old form is quadratic in the answer length; the point of the gate is that
    # doubling the answer roughly doubles the work instead of quadrupling it. Two profiles, because
    # the cheap "does the tail even contain a '<'" guard does NOT fire on the second: prose, and
    # code-shaped output (a '<' every few tokens, which is what a coding agent actually emits)
    # ending in a tool call, so the gate is measured with its boundary pinned by live markup too.
    def prose(n):
        return ["word%d " % i for i in range(n)]

    def code(n):
        out = ["if (a%d < b) { x <= y; }\n" % i if i % 3 else "line %d\n" % i for i in range(n)]
        return out + list(HERMES)

    for label, mk in (("prose", prose), ("code+call", code)):
        print(f"\n{label:>10} {'tokens':>8} {'old ms':>10} {'new ms':>10} {'speedup':>8}")
        for ntok in (500, 1000, 2000, 4000):
            pieces = mk(ntok)
            t0 = time.perf_counter()
            raw, emitted = "", 0
            for p in pieces:
                plain, _ = _segment_tools(raw := raw + p, False)
                if len(plain) > emitted:
                    emitted = len(plain)
            t1 = time.perf_counter()
            gate = _ToolGate(False)
            for p in pieces:
                gate.feed(p)
            t2 = time.perf_counter()
            old_ms, new_ms = (t1 - t0) * 1e3, (t2 - t1) * 1e3
            print(f"{'':>10} {ntok:>8} {old_ms:>10.1f} {new_ms:>10.1f} "
                  f"{old_ms / max(new_ms, 1e-9):>7.1f}x")
    print("\nALL EQUIVALENCE CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
