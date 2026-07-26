"""worker_t2music: the worker-side text-to-MUSIC engine (#t2music-serve, MusicGen).

The music-generation sibling of worker_t2a (ACE-Step), but a DIFFERENT architecture
on purpose: MusicGen is an AUTOREGRESSIVE transformer that decodes discrete EnCodec
audio tokens step by step (vs ACE-Step's continuous latent DIFFUSION). That matters
for portability:

  * It ships INSIDE HuggingFace `transformers` (`MusicgenForConditionalGeneration` +
    `AutoProcessor`) — no separate `acestep` package, no GitHub source install.
  * Text->music needs NO `torchaudio`: generate() returns an audio tensor and we write
    the WAV with `soundfile`. (ACE-Step hard-depends on torchaudio, which has no ROCm
    build — the wall that makes ACE-Step un-runnable on gfx1151.)
  * The heavy compute is the transformer decode (pure attention/matmul — fast on every
    backend incl. ROCm/gfx1151, no MIOpen). The ONLY conv is EnCodec's one-shot decode
    at the end, a bounded MIOpen JIT-once-then-fast tax exactly like Whisper's encoder.
  * There IS a CPU fallback (slow, but it runs) — ACE-Step has none.

So this leaf runs on AMD GPU (ROCm), NVIDIA GPU (CUDA) and CPU. Whole model on ONE
worker (like tts/stt/t2a) — it never touches the pipeline layer-split machinery.

Data flow (mirror of t2a): the controller ships a text prompt over the control link
(`t2music_gen`); this worker runs MusicGen, encodes the waveform to a WAV in memory
(soundfile), and returns it as base64 in `t2music_done` (#media-anywhere — no shared
FS needed, so any capable GPU/CPU worker can serve it, not only a co-located one).

DEPENDS ON (pip): transformers (MusicGen is built in) + soundfile (WAV encode) — both
already required by the stt leaf, so a `can_stt` worker is also `can_t2music`. Heavy
imports live inside methods so importing this module costs nothing.

Worker-side leaf: imported lazily by worker_load's t2music branch (fetch-if-missing via
worker_update._fetch_repo_file); listed in client.py's worker update file list +
server.py's EXTRA_UPDATE_FILES.
"""
from __future__ import annotations

import io
import os
import threading
import time

GB = 1024 ** 3


class _suppress:
    """Tiny contextlib.suppress(Exception) without importing contextlib at module top."""
    def __enter__(self):
        return self

    def __exit__(self, *a):
        return True


class MusicGenPipeline:
    """One resident MusicGen model on this worker. Stored in worker.shards[model_id] like a
    T2APipeline / WhisperPipeline; ``kind`` lets dispatchers tell it apart. One render at a
    time per model (_gen_lock) — the controller also serializes on LoadedModel.lock, this is
    the worker-side belt."""

    kind = "t2music"

    def __init__(self, model_dir: str, device: str = "", quant: str = "none",
                 offload: bool = False):
        import torch
        self.model_dir = model_dir
        self.quant = "none"          # MusicGen is a small bf16/fp16 net; no quant tiers in v1
        self.offload = False
        self._gen_lock = threading.Lock()
        self._doomed = False
        t0 = time.time()
        try:
            from transformers import (AutoProcessor,  # noqa: F401
                                      MusicgenForConditionalGeneration)
        except Exception as exc:
            raise RuntimeError(
                "t2music serving needs a transformers with MusicGen support on this worker "
                f"— import failed: {exc!r}") from exc
        self._Cls = MusicgenForConditionalGeneration
        self._processor = AutoProcessor.from_pretrained(model_dir)

        want = str(device or "")
        if not want or "gpu" in want:
            want = "cuda" if torch.cuda.is_available() else "cpu"
        # gfx1151 / ROCm: KEEP MusicGen on the GPU — the bottleneck is the autoregressive
        # transformer (pure attention/matmul, no MIOpen, fast on ROCm). The ONLY conv is
        # EnCodec's decode, which JIT-compiles through MIOpen the FIRST time it runs; that
        # cost rides the first render (not the load), then stays hot for the worker's life.
        # We therefore do NOT warm on load (a warmup render is expensive anyway). CUDA warms
        # naturally on first render (no MIOpen). Set INFINITEMODEL_T2MUSIC_CPU=1 to force CPU.
        self._rocm = bool(want.startswith("cuda") and getattr(torch.version, "hip", None))
        if want.startswith("cuda") and os.environ.get("INFINITEMODEL_T2MUSIC_CPU") == "1":
            print("[t2music] INFINITEMODEL_T2MUSIC_CPU=1 — forcing MusicGen onto CPU", flush=True)
            want, self._rocm = "cpu", False
        elif self._rocm:
            print("[t2music] ROCm/gfx1151 — MusicGen on GPU (transformer is MIOpen-free; the "
                  "one-time EnCodec-decode MIOpen JIT rides the FIRST render)", flush=True)
        self.device = self._build(want)

        ae = getattr(self.model.config, "audio_encoder", None)
        self.sample_rate = int(getattr(ae, "sampling_rate", 32000) or 32000)
        # tokens/sec of generated audio — duration -> max_new_tokens. EnCodec@32k = 50 Hz.
        self.frame_rate = float(getattr(ae, "frame_rate", 50.0) or 50.0)
        # #t2music-cap: the audio-token decoder has a FIXED position table
        # (decoder.max_position_embeddings, 2048 for MusicGen). Its delay pattern adds
        # num_codebooks-1 positions + a BOS, so generating past ~that many tokens indexes OUT of the
        # table -> a CUDA device-side assert (which then WEDGES the worker's CUDA context) or a CPU
        # crash. Clamp EVERY render to this hard ceiling. ~2036 tokens ≈ 40 s at 50 Hz.
        _dec = getattr(self.model.config, "decoder", None)
        _pos = int(getattr(_dec, "max_position_embeddings", 2048) or 2048)
        self.num_codebooks = int(getattr(_dec, "num_codebooks", 4) or 4)
        self.max_tokens = max(16, _pos - self.num_codebooks - 8)
        self.max_duration_s = round(self.max_tokens / self.frame_rate, 1)
        self.loaded_params = sum(p.numel() for p in self.model.parameters())
        self.loaded_bytes = sum(p.numel() * p.element_size()
                                for p in self.model.parameters()) + \
            sum(b.numel() * b.element_size() for b in self.model.buffers())
        self.gpu_bytes = self.loaded_bytes if str(self.device).startswith("cuda") else 0
        self.last_gen_s = 0.0
        self.last_audio_s = 0.0
        print(f"[t2music] musicgen ready on {self.device} in {time.time() - t0:.1f}s "
              f"({self.loaded_params / 1e6:.0f}M params, {self.loaded_bytes / GB:.2f} GB, "
              f"{self.sample_rate} Hz)", flush=True)

    def _build(self, device: str) -> str:
        import torch

        def _b(dev: str):
            dt = torch.float16 if str(dev).startswith("cuda") else torch.float32
            m = self._Cls.from_pretrained(self.model_dir, torch_dtype=dt)
            return m.to(dev).eval()

        try:
            self.model = _b(device)
            return device
        except Exception as exc:
            msg = repr(exc)
            hip = any(s in msg for s in ("MIOpen", "HIPRTC", "hiprtc", "HIP error",
                                         "hipErrorNoBinaryForGpu", "miopen",
                                         "Code object build failed", "out of memory",
                                         "CUDA out of memory"))
            if str(device).startswith("cuda") and hip:
                print(f"[t2music] GPU build failed on {device} ({exc!r}) — falling back to CPU "
                      "(MusicGen runs on CPU; slower)", flush=True)
                with _suppress():
                    del self.model
                    torch.cuda.empty_cache()
                self.model = _b("cpu")
                return "cpu"
            raise

    def media_info(self) -> dict:
        """Static metadata for the controller's /status + detail modal (reported in the load
        reply). Device is the ACTUAL device after any GPU->CPU fallback. Rich on purpose —
        the model-detail page surfaces all of it."""
        _sz = self.loaded_params
        _label = ("large" if _sz > 2.5e9 else "medium" if _sz > 1.0e9
                  else "small" if _sz > 5e8 else "melody" if _sz > 3e8 else "musicgen")
        return {"kind": "t2music", "engine": "musicgen", "device": str(self.device),
                "sample_rate": self.sample_rate, "frame_rate": self.frame_rate,
                "params": int(self.loaded_params), "loaded_bytes": int(self.loaded_bytes),
                "variant": _label,
                "max_duration_s": int(self.max_duration_s), "default_guidance": 3.0,
                "backend": ("ROCm" if self._rocm else
                            "CUDA" if str(self.device).startswith("cuda") else "CPU"),
                "channels": 1, "codec": "EnCodec", "arch": "autoregressive-transformer"}

    # -- unload -------------------------------------------------------------------------

    def release_vram(self) -> None:
        """Free GPU tensor storages on unload. RENDER-SAFE: under a held _gen_lock we only mark
        _doomed and the render's own finally frees when it completes (mirrors T2APipeline)."""
        if self._gen_lock.locked():
            self._doomed = True
            print("[t2music] unload during a live render — VRAM release deferred to end",
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
        print(f"[t2music] {os.path.basename(self.model_dir)}: GPU storages released", flush=True)

    # -- generation ---------------------------------------------------------------------

    def _wav_bytes(self, samples, sr: int) -> bytes:
        """float32 mono [-1,1] -> 16-bit PCM WAV bytes via soundfile (NO torchaudio)."""
        import numpy as np
        import soundfile as sf
        y = np.asarray(samples, dtype=np.float32).reshape(-1)
        peak = float(np.max(np.abs(y))) if y.size else 0.0
        if peak > 1.0:                       # guard against clipping if the decoder overshoots
            y = y / peak
        buf = io.BytesIO()
        sf.write(buf, y, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue()

    def generate(self, prompt: str, duration: float = 15.0, guidance: float = 3.0,
                 temperature: float = 1.0, top_k: int = 250, top_p: float = 0.0,
                 seed=None, on_step=None) -> tuple:
        """Text prompt -> (wav_bytes, wall_seconds, audio_seconds). Runs in a worker thread
        (asyncio.to_thread) — one at a time per model via _gen_lock."""
        try:
            return self._generate(prompt, duration, guidance, temperature, top_k, top_p, seed,
                                  on_step)
        finally:
            if self._doomed:
                print("[t2music] deferred VRAM release: freeing the unloaded model post-render",
                      flush=True)
                self._free_now()

    def _generate(self, prompt, duration, guidance, temperature, top_k, top_p, seed,
                  on_step) -> tuple:
        import torch
        with self._gen_lock:
            t0 = time.time()
            prompt = (prompt or "").strip()
            if not prompt:
                raise RuntimeError("empty prompt")
            dur = max(1.0, min(float(self.max_duration_s), float(duration)))
            # NEVER exceed the decoder's position table — overrunning it is the CUDA device-side
            # assert (which wedges the worker) / CPU crash. self.max_tokens is the hard ceiling.
            max_new = min(max(16, int(round(dur * self.frame_rate))), self.max_tokens)
            if seed is not None:
                with _suppress():
                    torch.manual_seed(int(seed))
                    if str(self.device).startswith("cuda"):
                        torch.cuda.manual_seed_all(int(seed))
            inputs = self._processor(text=[prompt], padding=True, return_tensors="pt")
            inputs = {k: v.to(self.model.device) for k, v in inputs.items()
                      if hasattr(v, "to")}
            gen_kw = dict(max_new_tokens=max_new, do_sample=True,
                          guidance_scale=float(guidance),
                          temperature=max(1e-3, float(temperature)),
                          top_k=int(top_k))
            if top_p and float(top_p) > 0:
                gen_kw["top_p"] = float(top_p)
            # optional coarse token-progress (best-effort; never breaks generation)
            crit = _mk_progress(on_step, max_new) if on_step else None
            if crit is not None:
                try:
                    from transformers import StoppingCriteriaList
                    gen_kw["stopping_criteria"] = StoppingCriteriaList([crit])
                except Exception:
                    pass
            with torch.no_grad():
                audio = self.model.generate(**inputs, **gen_kw)
            arr = audio[0].detach().to("cpu", dtype=torch.float32).numpy().reshape(-1)
            audio_s = len(arr) / float(self.sample_rate) if len(arr) else 0.0
            wav = self._wav_bytes(arr, self.sample_rate)
            self.last_gen_s = time.time() - t0
            self.last_audio_s = audio_s
            print(f"[t2music] '{prompt[:48]}' -> {audio_s:.1f}s audio in {self.last_gen_s:.1f}s "
                  f"(RTF {self.last_gen_s / max(audio_s, 1e-6):.2f}, {len(wav) / 1e6:.2f} MB)",
                  flush=True)
            return wav, self.last_gen_s, audio_s


def _mk_progress(on_step, total: int):
    """A StoppingCriteria that never stops — it just reports decode progress. Guarded so any
    transformers-version API drift degrades to 'no progress' rather than a failed render."""
    try:
        from transformers import StoppingCriteria
    except Exception:
        return None

    _every = max(1, total // 40)          # ~40 progress messages max — don't flood the control link

    class _Prog(StoppingCriteria):
        def __init__(self):
            self._n0 = None
            self._last = -10 ** 9

        def __call__(self, input_ids, scores, **kw):
            try:
                n = int(input_ids.shape[-1])
                if self._n0 is None:
                    self._n0 = n
                cur = max(0, n - self._n0)
                if cur - self._last >= _every or cur >= total:
                    self._last = cur
                    on_step(min(cur, total), total)
            except Exception:
                pass
            return False       # never stop — this criteria only reports progress

    with _suppress():
        return _Prog()
    return None
