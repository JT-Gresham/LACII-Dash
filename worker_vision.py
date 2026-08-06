"""#vision-on-node: run a multimodal model's VISION TOWER on a worker, not the controller.

The controller used to materialize the tower in-process and run its forward there
(media_encode.py is imported by server.py only). On a GPU-less controller that is a ~230 s
CPU forward per image — the iM VM sat pinned at 400% doing nothing but this. A model forward
belongs on a NODE, like every other part of a model.

DELIBERATELY STANDALONE, like worker_t2a.py / worker_tts.py: it must NOT import multimodal.py
or media_encode.py — those are controller modules whose globals are injected by state.bind()
and are not importable on a worker. Everything here needs only torch + transformers +
safetensors.

Split of work (only the expensive part moves):
  * CONTROLLER keeps image preprocessing (`load=0.0s` — free) and all config-derived metadata
    (out_hidden / merge / image_token_id / counts), which it reads from a META-built model —
    no weights, no torch compute.
  * WORKER builds the tower ONCE per model (weights fetched from the controller's
    /vision_weights) and runs `visual(pixel_values, grid_thw)` on its GPU, returning the
    LM-ready MERGED tokens.

The forward mirrors media_encode._encode_images' standard branch EXACTLY — call the tower's
own forward (patch_embed -> blocks -> merger), which returns merged tokens at text-hidden
width; `get_image_features` returns the PRE-merge backbone (wrong dim/count to splice) and is
only a fallback.
"""
from __future__ import annotations

import time

_TOWER_CACHE: dict = {}          # model_id -> (visual_module, device_str)


def _resolve_visual(model):
    """The tower submodule: Qwen2.5-Omni nests it under .thinker, everything else under
    .model (mirrors multimodal._resolve_visual)."""
    th = getattr(model, "thinker", None)
    if th is not None and getattr(th, "visual", None) is not None:
        return th.visual
    inner = getattr(model, "model", None)
    vis = getattr(inner, "visual", None) if inner is not None else None
    if vis is not None:
        return vis
    # Llava/Mistral3/Gemma-4 style: a separate vision_tower (+ projector applied by the model)
    vt = getattr(inner, "vision_tower", None) if inner is not None else None
    if vt is not None:
        return vt
    raise RuntimeError(f"no vision tower found on {type(model).__name__}")


def _apply_ckpt_renames(sd: dict, model_type: str) -> dict:
    """Checkpoint-key -> module-tree renames, mirroring multimodal._vision_ckpt_renames. Only
    gemma4_unified needs them; every other arch is a pass-through."""
    if model_type != "gemma4_unified":
        return sd
    for src, dst in (("model.vision_embedder.", "model.embed_vision."),
                     ("model.embed_vision.embedding_projection.",
                      "model.embed_vision.multimodal_embedder.embedding_projection.")):
        for k in [k for k in sd if k.startswith(src)]:
            sd[dst + k[len(src):]] = sd.pop(k)
    return sd


def _build_tower(model_id: str, weights_url: str):
    """Meta-build the multimodal model (ZERO memory — the text LM stays on meta) and
    materialize ONLY the tower from the controller-served safetensors. Cached per model_id."""
    import torch
    import urllib.request
    from safetensors.torch import load as st_load
    from transformers import AutoConfig

    hit = _TOWER_CACHE.get(model_id)
    if hit is not None:
        return hit
    t0 = time.time()
    cfg = AutoConfig.from_pretrained(model_id)
    is_omni = getattr(cfg, "thinker_config", None) is not None
    with torch.device("meta"):
        if is_omni:
            from transformers import AutoModelForTextToWaveform as _Auto
        else:
            from transformers import AutoModelForImageTextToText as _Auto
        model = _Auto.from_config(cfg)
    model.eval()
    with urllib.request.urlopen(weights_url, timeout=1800) as r:
        blob = r.read()
    sd = _apply_ckpt_renames(st_load(blob), getattr(cfg, "model_type", "") or "")
    visual = _resolve_visual(model)
    # Load RELATIVE to the tower module rather than through the whole model. The checkpoint's
    # names are NOT the module-tree names — Qwen2.5-VL stores the tower at top-level `visual.*`
    # while the built module is `model.visual.*` (transformers renames on load) — so a
    # whole-model load_state_dict would match nothing and leave every tensor on meta. Stripping
    # the leading namespace makes this rename-proof for every spelling the controller serves.
    rel: dict = {}
    for k, v in sd.items():
        for pfx in ("model.visual.", "visual.", "thinker.visual.",
                    "model.vision_tower.", "vision_tower."):
            if k.startswith(pfx):
                rel[k[len(pfx):]] = v
                break
        else:
            rel[k] = v                    # projector/embedder pieces: leave as-is
    # assign=True installs the served tensors straight onto the meta modules (no copy); the
    # text LM keeps its meta params and is never touched — we only ever call the tower.
    visual.load_state_dict(rel, strict=False, assign=True)
    # Some tower tensors are NOT in the checkpoint: a rotary module's `inv_freq` is a
    # NON-PERSISTENT buffer that transformers computes from config at __init__ — under a meta
    # build it is created on meta and no served tensor can fill it. Left alone, the subsequent
    # .to(device) dies with "Cannot copy out of meta tensor; no data!". (The controller's own
    # loader reports the same thing as "materialized 1 meta tensor(s)".) Rebuild every remaining
    # meta buffer on the real device before moving the tower.
    _rebuilt = 0
    for _mod in visual.modules():
        for _bname, _buf in list(_mod.named_buffers(recurse=False)):
            if not getattr(_buf, "is_meta", False):
                continue
            _new = None
            if _bname == "inv_freq":
                dim = int(getattr(_mod, "dim", 0) or 0) or int(_buf.shape[-1]) * 2
                theta = float(getattr(_mod, "theta", 0) or getattr(_mod, "base", 0) or 10000.0)
                _new = 1.0 / (theta ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim))
                _new = _new[: _buf.shape[-1]]
            if _new is None:
                _new = torch.zeros(_buf.shape, dtype=_buf.dtype)
            _mod.register_buffer(_bname, _new.to(_buf.dtype), persistent=False)
            _rebuilt += 1
    if _rebuilt:
        print(f"[vision-node] rebuilt {_rebuilt} non-persistent buffer(s) not in the checkpoint",
              flush=True)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    visual.to(dev)
    left = [n for n, p in visual.named_parameters() if getattr(p, "is_meta", False)]
    if left:
        raise RuntimeError(f"vision tower still has {len(left)} unmaterialized tensor(s) "
                           f"(e.g. {left[0]}) — served weights don't cover this arch")
    print(f"[vision-node] built {model_id} tower on {dev} in {time.time() - t0:.1f}s "
          f"({len(blob) / (1 << 20):.0f} MB)", flush=True)
    _TOWER_CACHE[model_id] = (visual, dev)
    return visual, dev


def encode(model_id: str, weights_url: str, payload: bytes) -> bytes:
    """Run the tower. `payload` is safetensors bytes carrying `pixel_values` and (optionally)
    `image_grid_thw`, exactly as the controller's image processor produced them. Returns
    safetensors bytes carrying `feats` — the LM-ready merged tokens the controller splices."""
    import torch
    from safetensors.torch import load as st_load, save as st_save

    visual, dev = _build_tower(model_id, weights_url)
    inp = st_load(payload)
    pv = inp["pixel_values"].to(dev)
    grid = inp.get("image_grid_thw")
    grid_dev = grid.to(dev) if grid is not None else None
    t0 = time.time()
    with torch.inference_mode():
        try:
            feats = visual(pv, grid_dev) if grid_dev is not None else visual(pv)
        except Exception as exc:                     # same fallback as the controller path
            print(f"[vision-node] tower forward failed ({type(exc).__name__}: "
                  f"{str(exc)[:160]}) -> get_image_features", flush=True)
            raise
    if not isinstance(feats, torch.Tensor):          # some towers return a ModelOutput
        feats = getattr(feats, "pooler_output", None) or getattr(feats, "last_hidden_state", None)
    out = feats.detach().to("cpu")
    print(f"[vision-node] {model_id}: forward={time.time() - t0:.1f}s on {dev} -> "
          f"{list(out.shape)}", flush=True)
    return st_save({"feats": out.contiguous()})
