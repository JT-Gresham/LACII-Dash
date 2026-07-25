"""worker_stt: the worker-side speech-to-TEXT engine (#stt-serve, Whisper).

The transcription sibling of worker_tts / worker_t2a: serves an OpenAI Whisper
checkpoint (``WhisperForConditionalGeneration`` + ``WhisperProcessor``) as a
single-node speech-recognition leaf. Whisper is a small (~0.8-1.5B) ENCODER-
DECODER model, so the WHOLE model runs on ONE worker — it never touches the
pipeline layer-split machinery the decoder-only LLMs use.

Data flow (mirror of tts/t2a, but the audio travels IN and the result is TEXT):
the controller ships the raw uploaded audio file as base64 over the control link
(``stt_transcribe``); this worker DECODES it (soundfile -> 16 kHz mono float32,
scipy/​numpy resample), runs Whisper, and returns the transcript in ``stt_done``.
Because the deliverable is text on the control reply — not a WAV on a shared
filesystem — an STT model works #media-anywhere: any capable GPU/CPU worker, not
only one co-located with the controller (a remote worker ``snapshot_download``s
the checkpoint itself, exactly like the t2a path).

DEPENDS ON (pip): transformers (Whisper is built in), soundfile (decode the input
audio; wav/flac/ogg), and — only when the input isn't already 16 kHz — scipy for
a high-quality resample (a numpy linear-interp fallback covers a scipy-less box).
Heavy imports live inside methods so importing this module costs nothing.

Worker-side leaf: imported lazily by worker_load's stt branch (fetch-if-missing
via worker_update._fetch_repo_file); listed in client.py's worker update file
list + server.py's EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

import io
import os
import threading
import time

GB = 1024 ** 3
SR = 16000                 # Whisper's fixed input sample rate (Hz)
_CHUNK_S = 30              # Whisper's native receptive window (30 s of 16 kHz audio)


class _suppress:
    """Tiny contextlib.suppress(Exception) without importing contextlib at module top."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return True


def _decode_audio(audio_bytes: bytes) -> "object":
    """Raw audio file bytes -> 16 kHz mono float32 numpy array. soundfile decodes
    wav/flac/ogg; a multi-channel signal is averaged to mono; a non-16 kHz signal is
    resampled (scipy.resample_poly, or a numpy linear-interp fallback)."""
    import numpy as np
    import soundfile as sf
    y, sr = sf.read(io.BytesIO(audio_bytes), dtype="float32", always_2d=False)
    if getattr(y, "ndim", 1) > 1:                 # stereo / multi-channel -> mono
        y = y.mean(axis=1)
    y = np.ascontiguousarray(y, dtype=np.float32)
    if sr != SR and y.size:
        try:
            from math import gcd
            from scipy.signal import resample_poly
            g = gcd(int(sr), SR) or 1
            y = resample_poly(y, SR // g, int(sr) // g).astype(np.float32)
        except Exception:
            # scipy-less fallback: linear interpolation (adequate for ASR features).
            n = int(round(len(y) * SR / float(sr)))
            if n > 0:
                xp = np.linspace(0.0, 1.0, num=len(y), endpoint=False)
                xq = np.linspace(0.0, 1.0, num=n, endpoint=False)
                y = np.interp(xq, xp, y).astype(np.float32)
    return y


class WhisperPipeline:
    """One resident Whisper ASR model on this worker. Stored in worker.shards[model_id]
    like a KokoroPipeline / T2APipeline; ``kind`` lets dispatchers tell it apart. One
    transcribe at a time per model (_gen_lock) — the controller also serializes on
    LoadedModel.lock, this is the worker-side belt."""

    kind = "stt"

    def __init__(self, model_dir: str, device: str = "", quant: str = "none",
                 offload: bool = False):
        import torch
        self.model_dir = model_dir
        self.quant = "none"          # Whisper is a small bf16/fp16 net; no quant tiers in v1
        self.offload = False
        self._gen_lock = threading.Lock()
        self._doomed = False
        t0 = time.time()
        try:
            from transformers import (WhisperForConditionalGeneration,  # noqa: F401
                                      WhisperProcessor)
        except Exception as exc:
            raise RuntimeError(
                "stt serving needs a transformers with Whisper support on this worker "
                f"— import failed: {exc!r}") from exc
        self._Cls = WhisperForConditionalGeneration
        self._processor = WhisperProcessor.from_pretrained(model_dir)

        want = str(device or "")
        if not want or "gpu" in want:
            want = "cuda" if torch.cuda.is_available() else "cpu"
        # gfx1151 / ROCm: Whisper stays on the GPU (decode is fast once the kernels are warm), but
        # the FIRST GPU inference JIT-compiles conv/attention kernels — ~8 min on this chip, and (on
        # TheRock ROCm 7.13) the result is NOT cached to disk, so a warmup on the load would pay it
        # every time AND race the controller restart. So on ROCm we DEFER the warmup: the load
        # returns instantly, the first transcribe pays the JIT once, and the compiled kernels then
        # stay hot in-process for the worker's lifetime (a restart re-pays it — rare). NVIDIA/CUDA
        # warms on load (fast there, no MIOpen). Set INFINITEMODEL_STT_CPU=1 to force CPU instead
        # (instant load, but ~30x-realtime transcribes on gfx1151).
        _rocm = bool(want.startswith("cuda") and getattr(torch.version, "hip", None))
        if want.startswith("cuda") and os.environ.get("INFINITEMODEL_STT_CPU") == "1":
            print("[stt] INFINITEMODEL_STT_CPU=1 — forcing Whisper onto CPU", flush=True)
            want, _rocm = "cpu", False
        elif _rocm:
            print("[stt] ROCm/gfx1151 — Whisper on GPU with warmup DEFERRED to the first transcribe "
                  "(the one-time ~8-min MIOpen JIT won't block the load)", flush=True)
        # Build on the requested device (+ warmup unless deferred on ROCm). A GPU path that raises a
        # HIP/MIOpen kernel-COMPILE error still falls back to CPU transparently.
        self.device = self._build_and_warm(want, warm=not _rocm)

        self.loaded_params = sum(p.numel() for p in self.model.parameters())
        self.loaded_bytes = sum(p.numel() * p.element_size()
                                for p in self.model.parameters()) + \
            sum(b.numel() * b.element_size() for b in self.model.buffers())
        self.gpu_bytes = self.loaded_bytes if str(self.device).startswith("cuda") else 0
        self.last_gen_s = 0.0
        self.last_audio_s = 0.0
        print(f"[stt] whisper ready on {self.device} in {time.time() - t0:.1f}s "
              f"({self.loaded_params / 1e6:.0f}M params, "
              f"{self.loaded_bytes / GB:.2f} GB)", flush=True)

    def _build_and_warm(self, device: str, warm: bool = True) -> str:
        import torch
        import numpy as np

        def _build(dev: str):
            dt = torch.float16 if str(dev).startswith("cuda") else torch.float32
            m = self._Cls.from_pretrained(self.model_dir, torch_dtype=dt)
            return m.to(dev).eval()

        try:
            self.model = _build(device)
            if warm and str(device).startswith("cuda"):
                # Warm the encoder/decoder kernels on 0.2 s of silence — this is where
                # gfx1151 MIOpen JIT-fails if it's going to. Skipped when the caller defers the
                # warmup (ROCm: the ~8-min JIT rides the first real transcribe instead of the load).
                self._run(np.zeros(SR // 5, dtype=np.float32), "", "transcribe")
            return device
        except Exception as exc:
            msg = repr(exc)
            hip = any(s in msg for s in ("MIOpen", "HIPRTC", "hiprtc", "HIP error",
                                         "hipErrorNoBinaryForGpu", "miopen",
                                         "Code object build failed"))
            if str(device).startswith("cuda") and hip:
                print(f"[stt] GPU kernel-compile failed on {device} ({exc!r}) — "
                      "falling back to CPU (Whisper is small; CPU is fine)", flush=True)
                with _suppress():
                    del self.model
                    torch.cuda.empty_cache()
                self.model = _build("cpu")
                return "cpu"
            raise

    def media_info(self) -> dict:
        """Static metadata for the controller's /status + detail modal (reported in the load
        reply). Device is the ACTUAL device after any GPU->CPU fallback."""
        return {"kind": "stt", "engine": "whisper", "device": str(self.device),
                "sample_rate": SR, "params": int(self.loaded_params),
                "loaded_bytes": int(self.loaded_bytes)}

    # -- unload -------------------------------------------------------------------------

    def release_vram(self) -> None:
        """Free GPU tensor storages on unload. RENDER-SAFE: under a held _gen_lock we only
        mark _doomed and the transcribe's own finally frees when it completes (mirrors
        KokoroPipeline / T2APipeline)."""
        if self._gen_lock.locked():
            self._doomed = True
            print("[stt] unload during a live transcription — VRAM release deferred to end",
                  flush=True)
            return
        self._free_now()

    def _free_now(self) -> None:
        import torch
        with _suppress():
            for t in list(self.model.parameters(recurse=True)) + \
                    list(self.model.buffers(recurse=True)):
                if t is not None and getattr(t, "device", None) is not None \
                        and t.device.type == "cuda":
                    t.data = torch.empty(0, dtype=t.dtype, device=t.device)
        with _suppress():
            if str(self.device).startswith("cuda"):
                import gc
                gc.collect()
                torch.cuda.empty_cache()
        print(f"[stt] {os.path.basename(self.model_dir)}: GPU storages released", flush=True)

    # -- transcription ------------------------------------------------------------------

    def _run(self, samples, language: str, task: str) -> str:
        """One <=30 s window of 16 kHz mono float32 -> text. The processor pads/truncates
        to Whisper's 30 s mel window on its own."""
        import torch
        feats = self._processor(samples, sampling_rate=SR,
                                return_tensors="pt").input_features
        dtype = next(self.model.parameters()).dtype
        feats = feats.to(self.model.device, dtype=dtype)
        kw: dict = {}
        if language:
            kw["language"] = language
        if task:
            kw["task"] = task
        with torch.no_grad():
            ids = self.model.generate(feats, **kw)
        out = self._processor.batch_decode(ids, skip_special_tokens=True)
        return (out[0] if out else "").strip()

    def transcribe(self, audio_bytes: bytes, language: str = "",
                   task: str = "transcribe") -> tuple:
        """Decode + transcribe raw audio file bytes; returns (text, wall_seconds, audio_seconds).
        Runs in a worker thread (asyncio.to_thread) — one at a time per model via _gen_lock."""
        try:
            return self._transcribe(audio_bytes, language, task)
        finally:
            if self._doomed:
                print("[stt] deferred VRAM release: freeing the unloaded model post-transcription",
                      flush=True)
                self._free_now()

    def _transcribe(self, audio_bytes: bytes, language: str, task: str) -> tuple:
        with self._gen_lock:
            t0 = time.time()
            y = _decode_audio(audio_bytes)
            audio_s = len(y) / float(SR) if len(y) else 0.0
            if not len(y):
                raise RuntimeError("empty / undecodable audio")
            task = (task or "transcribe").strip().lower()
            if task not in ("transcribe", "translate"):
                task = "transcribe"
            win = SR * _CHUNK_S
            parts = []
            if len(y) <= win:
                parts.append(self._run(y, language, task))
            else:
                # Long-form: split into <=30 s windows and concatenate (Whisper's native
                # window is 30 s; a segment shorter than 0.2 s is silence tail, skip it).
                for i in range(0, len(y), win):
                    seg = y[i:i + win]
                    if len(seg) < SR // 5:
                        continue
                    parts.append(self._run(seg, language, task))
            text = " ".join(p for p in parts if p).strip()
            self.last_gen_s = time.time() - t0
            self.last_audio_s = audio_s
            print(f"[stt] {audio_s:.1f}s audio -> {len(text)} chars in {self.last_gen_s:.1f}s "
                  f"(RTF {self.last_gen_s / max(audio_s, 1e-6):.2f})", flush=True)
            return text, self.last_gen_s, audio_s
