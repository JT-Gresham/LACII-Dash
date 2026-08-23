"""#bazarr-asr — the two pure adapters, against the REAL functions.

Bazarr's contract has exactly two places it can silently go wrong on our side:

  1. `encode=false` means the body is HEADERLESS s16le/mono/16k PCM. If the WAV header we prepend
     is wrong, Whisper still transcribes something — at the wrong speed, or as noise — so the
     failure surfaces as a bad subtitle rather than an error. The header is therefore checked by
     DECODING it back with the stdlib `wave` module and comparing sample-for-sample.
  2. SRT that is subtly malformed (non-monotonic cues, zero-length cues, wrong separator) is
     accepted by some players and silently dropped by others.

Imports the real helpers from routes_api rather than restating them — a transcription would only
assert that I can copy my own code.
"""
import io
import re
import sys
import wave

from routes_api import _wav_from_pcm16, _srt_ts, _segments_to_srt, _segments_to_vtt

failures: list[str] = []

# ---------------------------------------------------------------- 1. the WAV header
import struct
pcm = b"".join(struct.pack("<h", (i * 137) % 30000 - 15000) for i in range(16000))  # 1.0 s @16k
wav = _wav_from_pcm16(pcm)
if len(wav) != len(pcm) + 44:
    failures.append(f"header must be exactly 44 bytes; got {len(wav) - len(pcm)}")
try:
    with wave.open(io.BytesIO(wav), "rb") as w:
        if w.getnchannels() != 1:  failures.append(f"channels={w.getnchannels()}, want 1 (mono)")
        if w.getframerate() != 16000: failures.append(f"rate={w.getframerate()}, want 16000")
        if w.getsampwidth() != 2:  failures.append(f"sampwidth={w.getsampwidth()}, want 2 (s16)")
        if w.getnframes() != 16000:
            failures.append(f"frames={w.getnframes()}, want 16000 — duration would be wrong, "
                            f"which shifts every subtitle timestamp")
        back = w.readframes(w.getnframes())
        if back != pcm:
            failures.append("PCM did not survive the round trip — samples are being mangled")
except Exception as exc:
    failures.append(f"stdlib wave could not read our header ({exc!r}) — soundfile will not either")

# odd-length payload must not corrupt the header arithmetic
try:
    w2 = _wav_from_pcm16(b"\x01\x02\x03")
    if len(w2) != 47:
        failures.append("odd-length PCM broke the header length")
except Exception as exc:
    failures.append(f"odd-length PCM raised {exc!r}")

# ---------------------------------------------------------------- 2. timestamp format
for t, want in ((0.0, "00:00:00,000"), (1.5, "00:00:01,500"), (61.25, "00:01:01,250"),
                (3661.007, "01:01:01,007"), (-5.0, "00:00:00,000")):
    got = _srt_ts(t)
    if got != want:
        failures.append(f"_srt_ts({t}) = {got!r}, want {want!r}")

# ---------------------------------------------------------------- 3. SRT structure
segs = [
    {"start": 0.0,  "end": 2.0,  "text": "Hello there."},
    {"start": 1.5,  "end": 3.0,  "text": "Overlapping start."},   # must be pushed monotonic
    {"start": 5.0,  "end": 5.0,  "text": "Zero length."},         # must be given a duration
    {"start": 7.0,  "end": 8.0,  "text": "   "},                  # blank -> dropped, no gap in numbering
    {"start": 9.0,  "end": 10.0, "text": "Multi\nline  text"},    # newlines/space collapsed
]
srt = _segments_to_srt(segs)
blocks = [b for b in srt.strip().split("\n\n") if b.strip()]
if len(blocks) != 4:
    failures.append(f"expected 4 cues (one blank dropped), got {len(blocks)}")
nums = [int(b.splitlines()[0]) for b in blocks if b.splitlines()[0].strip().isdigit()]
if nums != list(range(1, len(blocks) + 1)):
    failures.append(f"cue numbering must be 1..N with no gaps; got {nums}")

TS = re.compile(r"^(\d{2}):(\d{2}):(\d{2}),(\d{3}) --> (\d{2}):(\d{2}):(\d{2}),(\d{3})$")
prev_end = -1.0
for b in blocks:
    lines = b.splitlines()
    if len(lines) < 3:
        failures.append(f"cue must be number/timing/text; got {lines!r}"); continue
    m = TS.match(lines[1])
    if not m:
        failures.append(f"malformed timing line: {lines[1]!r}"); continue
    a = int(m[1])*3600 + int(m[2])*60 + int(m[3]) + int(m[4])/1000
    e = int(m[5])*3600 + int(m[6])*60 + int(m[7]) + int(m[8])/1000
    if e <= a:
        failures.append(f"zero/negative-length cue never displays: {lines[1]}")
    if a < prev_end - 1e-9:
        failures.append(f"cues must be monotonic; {lines[1]} starts before previous end {prev_end}")
    prev_end = e
    if any(not ln.strip() for ln in lines[2:]):
        failures.append(f"blank line inside a cue truncates it: {lines!r}")

if "Multi line text" not in srt:
    failures.append("embedded newline/whitespace must be collapsed (a raw \\n splits the cue)")
if "\n\n\n" in srt:
    failures.append("double blank line between cues breaks SRT parsing")

# ---------------------------------------------------------------- 4. empty input
if _segments_to_srt([]) != "":
    failures.append("no segments must give empty output, not a stray cue")

# ---------------------------------------------------------------- 5. VTT
vtt = _segments_to_vtt(segs)
if not vtt.startswith("WEBVTT"):
    failures.append("VTT must start with the WEBVTT magic")
if "," in vtt.split("\n", 2)[-1].split("-->")[0][-8:]:
    failures.append("VTT timings must use '.' not ',' as the decimal separator")

if failures:
    print("FAIL — #bazarr-asr adapters:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"PASS — WAV header round-trips through stdlib wave (1.0 s, mono, 16k, s16); "
      f"SRT is monotonic, gap-free, no zero-length cues; VTT ok")
