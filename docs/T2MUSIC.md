# Text-to-music serving (MusicGen)

InfiniteModel can serve **text-to-music** with **MusicGen** (Meta) — a text prompt in, an
instrumental WAV out — through the same OpenAI-style `POST /v1/audio/music` endpoint used by
ACE-Step, on the same fleet that serves the LLMs.

MusicGen is InfiniteModel's **second** music engine, and it is deliberately a **different
architecture** from ACE-Step (see [T2A.md](T2A.md)): where ACE-Step is a **latent-diffusion** model,
MusicGen is an **autoregressive transformer** that decodes discrete **EnCodec** audio tokens one step
at a time (then EnCodec decodes those tokens back to a waveform). That architectural difference is
the whole point — it is what lets MusicGen run **where ACE-Step can't**:

- it ships **inside `transformers`** — no separate `acestep` package, no source install;
- text→music needs **no `torchaudio`** (the finished waveform is written with `soundfile`), so it is
  not blocked on the ROCm/TheRock builds that ship no matching `torchaudio` ABI;
- its heavy compute is the **transformer decode** (pure attention/matmul — no MIOpen), and its only
  convolution is EnCodec's one-shot decode (a bounded JIT-once tax like Whisper's encoder);
- it has a real **CPU fallback**.

So MusicGen serves on **AMD (ROCm) / NVIDIA (CUDA) / CPU**, with a transparent GPU→CPU fallback.

> **This is an OPTIONAL component,** but a light one: it needs only `transformers` + `soundfile` on
> the serving worker (the *same* deps as Whisper STT — a `can_stt` worker is automatically
> `can_t2music`). No `acestep`/`torchaudio`/`diffusers` required. A fleet with no capable worker
> simply has no MusicGen model; everything else is unaffected.

---

## Architecture (what runs where)

Like TTS / STT / t2i / ACE-Step, MusicGen is **not** layer-split across the fleet — the whole model
(T5 text encoder + audio-token decoder + EnCodec) runs on **one worker**, as a single-node **media
leaf** (`worker_t2music.py`, `MusicGenPipeline`). It works **#media-anywhere**:

- **Placement** picks the controller-co-located worker **or any worker advertising the runtime**
  (`can_t2music` — an import-free `find_spec` probe = `transformers` + `soundfile`, shown per node in
  `/status`). Placement **honors the per-node tier toggles** (`NODE_CONFIG`): a node with **both**
  tiers disabled is opted out and never receives a render (keeps music off a benched/off-limits box).
- **Model delivery** needs no shared filesystem: a remote worker fetches the checkpoint itself via
  `snapshot_download(<hf-repo-id>)`. A co-located worker uses the fast local path.
- **Result delivery** needs no shared filesystem either: the finished WAV returns as **base64 over
  the control link** in `t2music_done`.

## Getting the models

MusicGen comes in four sizes (register any subset via the dashboard **+ Add model** or
`POST /add_model?model=<repo>&name=<friendly>`):

| Friendly | HF repo | Params | Notes |
|---|---|---|---|
| `musicgen-small`  | `facebook/musicgen-small`  | 300M | fastest; has safetensors |
| `musicgen-medium` | `facebook/musicgen-medium` | 1.5B | quality/speed sweet spot |
| `musicgen-large`  | `facebook/musicgen-large`  | 3.3B | best quality, slowest |
| `musicgen-melody` | `facebook/musicgen-melody` | 1.5B | melody-capable (v1 serves text-only) |

> **`.bin`-only weights.** Except `-small`, the MusicGen repos ship **no safetensors** — the weights
> are `pytorch_model.bin`, alongside redundant `state_dict.bin` / `compression_state_dict.bin`
> (original-format duplicates that `from_pretrained` ignores). `+ Add model` fetches **only**
> `pytorch_model.bin` (skipping the duplicates) so downloads stay lean, and the model is **served
> whole from its HF-cache snapshot** (never migrated to `models/`, like Kokoro). Licenses are
> **CC-BY-NC** — fine for personal/research use.

## Enabling MusicGen on a box

Nothing heavy — the two deps are already needed by Whisper STT:

```bash
pip install transformers soundfile
```

Then **restart the worker** — it advertises `can_t2music=true` on (re)registration. Note `POST
/refresh_backends` re-probes only the **controller's** own backends, *not* a worker's capability, so
it will **not** flip `can_t2music` — the worker process must restart.
NVIDIA and CPU work out of the box. On **AMD ROCm** the transformer runs on the GPU; the first render
JIT-compiles EnCodec's decode through MIOpen once (see [ROCM.md](ROCM.md)).

## The API — `POST /v1/audio/music`

`/v1/audio/music` is **polymorphic**: it dispatches to MusicGen or ACE-Step by the loaded model's
type, so the same endpoint serves both. MusicGen ignores ACE-Step's `lyrics`/`steps` and instead
takes sampling knobs:

```bash
curl -X POST http://<controller>:21434/v1/audio/music \
  -H 'Content-Type: application/json' \
  -d '{
        "model": "musicgen-medium",
        "prompt": "upbeat 80s synthwave, driving electronic drums, warm analog bass, bright arpeggiated lead",
        "duration": 12,
        "guidance": 3
      }' --output music.wav
```

| Field | Default | Notes |
|---|---|---|
| `model` | — | a registered MusicGen checkpoint |
| `prompt` (or `input`) | required | free-text description of the music |
| `duration` | `15` | seconds, clamped to `1`–`60` (MusicGen is trained ~10–30 s; longer just extends) |
| `guidance` | `3` | classifier-free guidance — higher = more on-prompt, less varied |
| `temperature` | `1.0` | sampling temperature |
| `top_k` | `250` | top-k sampling |
| `top_p` | `0` (off) | nucleus sampling (optional) |
| `seed` | random | integer for reproducibility |
| `response_format` | `wav` | 32 kHz mono WAV bytes returned |

Coarse per-render progress mirrors back as `t2music_step` (the dashboard shows a live `%`).
Generation is serialized per model. Errors: `400` bad request, `404` unknown model, `503` at
capacity / no capable worker.

## Dashboard — the **Generate music** panel

Open a MusicGen model's card (**Load 🎵** shows a short confirm dialog; the first render warms the
EnCodec decoder once). Once loaded, the detail page has a **Generate music** panel: a prompt box +
`duration / guidance / temperature / top-k / seed`, a **Generate ▶** button, an **inline `<audio>`
player**, and a **⬇ download wav** link. The card also shows a rich `media_info` block — variant,
backend (ROCm/CUDA/CPU), device, params, weight size, sample rate, token rate, codec (EnCodec), and
last-render RTF.

## Device behavior (AMD / NVIDIA / CPU)

| Backend | Load | Render |
|---|---|---|
| **NVIDIA (CUDA)** | fp16, fast | fast, warm immediately (no MIOpen) |
| **AMD (ROCm, gfx1151)** | fp16 on GPU | transformer is GPU-fast; **first** render pays a one-time EnCodec MIOpen JIT (~a minute), then fast (medium ≈ 2× realtime warm) |
| **CPU** | fp32 | **works but slow** — autoregressive on CPU is ~orders of magnitude off realtime; use a GPU for anything but tiny clips |

Force CPU with `INFINITEMODEL_T2MUSIC_CPU=1` on the worker. A GPU build that hits a HIP/MIOpen
compile error (or OOM) falls back to CPU transparently.

## MusicGen vs ACE-Step — which to use

- **ACE-Step** (diffusion, `#t2a`): higher-fidelity, longer, lyric-conditioned songs — but needs a
  **CUDA** GPU and the heavy `acestep` stack; **not available on ROCm/AMD or CPU** (see
  [T2A.md](T2A.md#why-ace-step-is-not-implemented-on-rocm-amd)).
- **MusicGen** (autoregressive, `#t2music`): lighter, `transformers`-native, **runs on
  AMD/NVIDIA/CPU** — the music option for a ROCm box, or any fleet that doesn't want the ACE-Step
  install.

Both are served through `/v1/audio/music`; load whichever the box can run.

## Limitations (v1)

- One worker per model — MusicGen is not layer-split (small enough to fit one device).
- Text-only conditioning (the melody variant's audio-melody conditioning is not wired up yet).
- ~30 s of well-formed music per prompt (longer durations extend but can drift); 32 kHz mono.
- CPU rendering is functional but far from realtime — GPU is the practical path.
