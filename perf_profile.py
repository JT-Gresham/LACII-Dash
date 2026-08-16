"""perf_profile.py — #perf-auto: resolve every performance knob from the DETECTED setup.

A 4B dense model on a CPU node and a 30B MoE on a discrete GPU want opposite settings, and the
right answer also depends on the *kernel* available for the chosen quant on the *specific* silicon.
This module holds that decision table in one auditable place. It is PURE: no I/O, no globals, no
torch import — it takes facts in and returns (knobs, rationale) out, so it can be unit-tested and
its reasoning printed into the load log.

CONTRACT: the resolver only ever fills knobs the caller left UNSET. An explicit request always
wins — this never overrides an operator.

WHAT THE LOADER DOES WITH THE ANSWER. Resolving a knob is not the same as applying it. engine_load
APPLIES only the ones that cannot change the model's OUTPUT and cannot make a working load worse —
today kv_slots (section 3) and moe_offload (section 4), both pure placement. Everything else is
ADVICE: it goes into the load log and into the dashboard's ⚡ button for the operator to accept.
head_quant (4b), quant (1) and kv_quant (2) all move the logits and are therefore NEVER auto-applied
however good the speed case looks; attn, prefix_min and cuda_graph have no per-load knob to apply
them to. Anything added here should say which side of that line it falls on, and
_perf_auto_knobs' docstring in engine_load.py carries the reason per knob.

Everything below is grounded in measurements from the 2026-08-13 performance investigation or in
prior vault findings, and each rule carries its evidence. Rules asserted without evidence were
deliberately left out: three plausible-sounding optimizations (CUDA graphs, lm_head int8 on CUDA,
torch.compile) all measured as PESSIMIZATIONS, so "sounds faster" is not a reason to set a knob.

DEVICE CLASSES (the axis that matters most, because it selects the int4 kernel):
  cpu          - no GPU. int4 uses _weight_int4pack_mm_for_cpu; bf16 matmul is the fallback.
  cuda_modern  - CUDA sm80+. torch tinygemm int4 available (worker_quant.py:493).
  cuda_legacy  - CUDA sm<80 (Pascal/Turing-). FAILS the >=(8,0) gate -> int4 silently falls back to
                 the NAIVE path, which rematerializes the full bf16 weight EVERY forward. Measured
                 5-20x slower. On these cards int4 is a memory tier, never a speed tier.
  rocm         - HIP. Project's own Triton w4a16 kernel; measured 119-134 GB/s on gfx1151.
  unified      - APU/iGPU with unified memory (Strix Halo). "VRAM" is system RAM, so a CPU spill
                 costs bandwidth rather than a PCIe hop, but decode is still bandwidth-bound.
"""
from __future__ import annotations

# --- device classification -----------------------------------------------------------------

CPU = "cpu"
CUDA_MODERN = "cuda_modern"
CUDA_LEGACY = "cuda_legacy"
ROCM = "rocm"
UNIFIED = "unified"


def classify_device(*, has_gpu: bool, is_hip: bool = False, capability: tuple | None = None,
                    unified_memory: bool = False, device_name: str = "") -> str:
    """Bucket one node's compute into the class that determines which kernels it can reach."""
    if not has_gpu:
        return CPU
    if is_hip:
        return UNIFIED if unified_memory else ROCM
    if unified_memory:
        return UNIFIED
    if capability and tuple(capability) >= (8, 0):
        return CUDA_MODERN
    if capability:
        return CUDA_LEGACY
    return CUDA_MODERN          # unknown capability: assume modern (the common case)


# --- KV sizing -----------------------------------------------------------------------------

def looks_moe(*, meas_layer_w, hidden_size: int, num_heads: int, num_kv_heads: int,
              head_dim: int, intermediate_size: int, attn_bias: bool = True,
              weight_dtype_bytes: int = 2, ratio: float = 1.6) -> bool:
    """Is this a Mixture-of-Experts model? Inferred, because the controller cannot know directly.

    MoE-ness is only established WORKER-side (from the checkpoint's 3D expert tensors), so
    ModelSpec carries no is_moe flag — an earlier version of this module asked for one and got a
    silent False, which made the resolver confidently report "dense model" for a 35B-A3B. The
    signal that IS available on the controller: spec_with_measurements fills meas_layer_w from the
    REAL safetensors headers, while the dense formula models a single MLP per layer. A MoE layer
    holds N experts, so measured/dense runs far above 1; anything at/below ~1 is dense.

    Returns False when the model has not been downloaded (meas_layer_w is None) — unknown, not
    dense. The caller must treat that as "cannot tell" rather than a positive dense claim.
    """
    if not meas_layer_w or not hidden_size or not intermediate_size:
        return False
    h, qd = int(hidden_size), int(num_heads) * int(head_dim)
    kvd = int(num_kv_heads) * int(head_dim)
    attn = h * qd + 2 * h * kvd + qd * h
    bias = (qd + 2 * kvd) if attn_bias else 0
    dense = (attn + bias + 3 * h * int(intermediate_size) + 2 * h) * int(weight_dtype_bytes)
    return dense > 0 and (float(meas_layer_w) / dense) >= ratio


def kv_bytes_per_token(*, num_layers: int, num_kv_heads: int, head_dim: int,
                       full_attention_layers: int | None = None, bytes_per_elem: int = 2) -> int:
    """K+V bytes one token occupies across every full-attention layer.

    full_attention_layers lets a HYBRID arch (Kimi-Linear, Gemma-4 sliding, qwen3-next) pass the
    count of layers that actually grow a full-ctx KV — the linear-attention layers hold conv and
    recurrent state whose size does not scale with context, so charging them full KV over-reserves
    and can falsely refuse a placement that would fit.
    """
    n = int(full_attention_layers if full_attention_layers is not None else num_layers)
    return 2 * max(0, n) * max(0, int(num_kv_heads)) * max(0, int(head_dim)) * int(bytes_per_elem)


# --- the resolver --------------------------------------------------------------------------

def resolve(*,
            # --- model facts ---
            params_b: float,
            num_layers: int,
            num_kv_heads: int,
            head_dim: int,
            vocab_size: int = 0,
            is_moe: bool = False,
            active_params_b: float = 0.0,
            is_hybrid: bool = False,
            full_attention_layers: int | None = None,
            is_multimodal: bool = False,
            arch: str = "",
            tie_word_embeddings: bool = False,
            # --- placement facts ---
            device_class: str = CUDA_MODERN,
            free_vram_gb: float = 0.0,
            free_ram_gb: float = 0.0,
            node_count: int = 1,
            # --- request facts ---
            ctx: int = 4096,
            has_draft_model: bool = False,
            has_mtp_head: bool = False,
            # --- caller's explicit choices (None/"" = unset, resolver may fill) ---
            requested: dict | None = None) -> tuple[dict, list[str]]:
    """Return (knobs, rationale). `requested` entries that are set are passed through untouched.

    The rationale is a list of human-readable lines explaining every decision, so the load log can
    show WHY a knob was chosen — the thing that was missing when these settings had to be guessed.
    """
    req = dict(requested or {})
    out: dict = {}
    why: list[str] = []

    def take(name, value, reason):
        """Fill `name` unless the caller set it explicitly."""
        given = req.get(name)
        if given not in (None, "", -1):
            out[name] = given
            why.append(f"{name}={given} (explicit — resolver deferred)")
            return given
        out[name] = value
        why.append(f"{name}={value} — {reason}")
        return value

    gpu = device_class in (CUDA_MODERN, CUDA_LEGACY, ROCM, UNIFIED)

    # ---- 1. quant -------------------------------------------------------------------------
    # int4 is the default speed AND memory tier, but ONLY where a fused kernel exists. On
    # cuda_legacy the >=(8,0) gate fails and QuantLinear4 falls back to rematerializing the whole
    # bf16 weight per forward — measured 5-20x slower than just holding bf16. There, int4 is a
    # last resort for fitting, never for speed.
    if device_class == CUDA_LEGACY:
        bf16_gb, int8_gb = params_b * 2.0, params_b * 1.0
        vram = max(0.0, free_vram_gb)
        if bf16_gb <= vram * 0.80:
            quant = take("quant", "none",
                         f"sm<80 has no fused int4 kernel (naive remat is 5-20x slower) and bf16 "
                         f"({bf16_gb:.1f} GB) fits free VRAM ({vram:.1f} GB)")
        elif int8_gb <= vram * 0.80:
            quant = take("quant", "int8",
                         f"sm<80 has no fused int4 kernel; bf16 ({bf16_gb:.1f} GB) does not fit "
                         f"{vram:.1f} GB — int8 halves it with a cheaper dequant than int4")
        else:
            # Fitting beats kernel quality: a slow int4 that stays resident still beats spilling
            # layers to CPU, which is the single most destructive thing for decode.
            quant = take("quant", "int4",
                         f"sm<80 int4 has no fused kernel and is SLOW, but neither bf16 "
                         f"({bf16_gb:.1f} GB) nor int8 ({int8_gb:.1f} GB) fits {vram:.1f} GB — "
                         f"staying GPU-resident beats a CPU spill")
    elif not gpu:
        quant = take("quant", "int4",
                     "CPU node: int4 via _weight_int4pack_mm_for_cpu; decode is memory-bound so "
                     "the smaller weight wins")
    else:
        quant = take("quant", "int4",
                     f"{device_class} has a fused int4 kernel; int4-g128 is 4.25 bits/weight, "
                     f"leaner than llama.cpp Q4_K's 4.50")

    # ---- 2. KV placement knobs ------------------------------------------------------------
    # Both of these are measured LOSSES on a GPU and exist only for fitting.
    take("kv_offload", False,
         "KV in system RAM is catastrophic for GPU decode (per-layer prefetch on the critical "
         "path); only worth it to fit a model that otherwise cannot load")
    take("kv_quant", "none",
         "TurboQuant KV adds a per-layer bf16 dequant transient on the decode path and gates off "
         "other fast paths; quality/'fit' knob, not a speed knob")

    # ---- 3. kv_slots: the measured 3.9x, but ONLY when the KV genuinely fits ----------------
    # #prefix-kv keeps ONE prefix record per slot. At C=1 a single interleaved request (a second
    # client, an agent side-query, a keep-alive probe) evicts it and every later turn re-prefills
    # the whole prompt. Measured on an idle card, 2594-token prompt: C=1 interleaved 6506 ms
    # (MISS) vs C=3 1665 ms (HIT) = 3.9x. _SlotLease already routes a request to the free slot
    # with the longest common prefix, so C>1 turns a depth-1 record into an associative cache.
    #
    # HARD CONSTRAINT: C x full-ctx KV is reserved up front. Overshooting forces layers onto CPU,
    # and CPU layers are far more destructive than a prefix miss (two CPU layers took a 7B from
    # 27 to 9.7 tok/s). So C is sized against real headroom and clamped to 1 when it would spill.
    kv_pt = kv_bytes_per_token(num_layers=num_layers, num_kv_heads=num_kv_heads,
                               head_dim=head_dim, full_attention_layers=full_attention_layers)
    kv_per_slot_gb = kv_pt * max(1, int(ctx)) / (1024 ** 3)
    weights_gb = params_b * {"none": 2.0, "int8": 1.0, "int4": 0.53, "int2": 0.30}.get(quant, 0.53)
    budget_gb = max(0.0, free_vram_gb) if gpu else max(0.0, free_ram_gb)
    spare_gb = budget_gb - weights_gb
    if kv_per_slot_gb <= 0:
        slots = take("kv_slots", 1, "KV geometry unknown — cannot size slots safely")
    elif is_multimodal or is_hybrid:
        slots = take("kv_slots", 1,
                     "multimodal/hybrid arch: #prefix-kv is gated off for these, so extra slots "
                     "buy concurrency but no prefix reuse")
    else:
        # keep 25% of the post-weights headroom as slack for activations and fragmentation
        affordable = int(max(0.0, spare_gb * 0.75) // kv_per_slot_gb)
        slots = max(1, min(3, affordable))
        if slots > 1:
            take("kv_slots", slots,
                 f"{spare_gb:.2f} GB spare after weights fits {affordable} x {kv_per_slot_gb:.2f} GB "
                 f"KV — C={slots} keeps interleaved conversations' prefixes warm (measured 3.9x)")
        else:
            take("kv_slots", 1,
                 f"only {spare_gb:.2f} GB spare after weights vs {kv_per_slot_gb:.2f} GB per slot — "
                 f"a 2nd slot would force layers to CPU, which costs far more than a prefix miss")

    # ---- 4. MoE expert offload ------------------------------------------------------------
    # APPLIED by engine_load (the TRUE direction only). Safe to apply because it is placement, not
    # arithmetic: shard_build reads the flag ONLY on the spill path, so it is inert for a shard that
    # fits VRAM whole, and on a shard that does not fit it converts "whole MoE layer -> CPU" into
    # "attention+norms on GPU, routed experts in RAM" (falling back to the whole layer if even the
    # mixer misses the budget). Output is bit-identical either way. That inertness is what makes a
    # false positive harmless: this sizes against ONE node's free VRAM while the pipeline planner
    # may spread the model over several, so `not fits` is a "may spill", not a "will spill".
    if is_moe:
        fits = (weights_gb + slots * kv_per_slot_gb) <= budget_gb * 0.95
        take("moe_offload", not fits,
             ("routed experts stay on GPU — the model fits" if fits else
              f"model ({weights_gb:.1f} GB) + KV does not fit {budget_gb:.1f} GB; offloading routed "
              f"experts to RAM keeps attention on GPU, which is the latency-critical part"))
    else:
        take("moe_offload", False, "dense model — no expert tier to offload")

    # ---- 4b. lm_head precision (#head-quant) ----------------------------------------------
    # ADVICE ONLY, never applied: this is the one knob here that changes OUTPUT, so it stays an
    # explicit operator choice. The case for it is measured, not theoretical — on Qwen2.5-7B the
    # head is 22% of everything read per decoded token, and:
    #     int4 head  +0.0389 nats, 84.7% top-1  -> 92.5% of the damage the whole int4 body does.
    #                                              REJECTED; head_quant refuses this value.
    #     int8 head  +0.0050 nats, 98.4% top-1  -> ~8x cheaper than the body's own cost.
    # With the w8a16 kernel an int8 head measured +9.9% decode on gfx1151 (32.06 -> 35.22 tok/s)
    # with prefill unchanged and 0.5 GB LESS VRAM. Only worth suggesting where it can actually
    # pay: the head must be GPU-resident (the kernel is GPU-only; on CPU an int8 head is SLOWER
    # than bf16), the body must not already be int8, and a tied head is not a separate matrix.
    if quant in ("int4", "int2") and gpu and not tie_word_embeddings:
        take("head_quant", "",
             "int8 lm_head is available and measured +9.9% decode for +0.0050 nats (98.4% top-1) "
             "— NOT applied automatically because it changes output; pass head_quant=int8 to take it")
    else:
        take("head_quant", "",
             "int8 lm_head not applicable here (" + (
                 "tied embeddings — the head aliases embed_tokens" if tie_word_embeddings else
                 "body is not int4/int2" if quant not in ("int4", "int2") else
                 "no GPU — an int8 head is slower than bf16 on CPU") + ")")

    # ---- 5. speculative decoding ----------------------------------------------------------
    # Only a win when BOTH the verifier and the draft are GPU-resident; a CPU-resident draft makes
    # every verify round wait on the slow path and loses to plain decode. MoE decode is already
    # cheap per token (only active params are read), so the draft has less to hide behind.
    if has_draft_model or has_mtp_head:
        draft_gb = 0.0 if has_mtp_head else max(0.5, params_b * 0.10)
        room = spare_gb - slots * kv_per_slot_gb - draft_gb
        if gpu and room > 0.5:
            take("speculative", True,
                 f"draft fits GPU alongside the verifier ({room:.1f} GB spare) — "
                 f"{'MTP self-draft' if has_mtp_head else 'separate draft model'}")
            take("draft_gpu", True, "draft on GPU: the only configuration where spec-decode wins")
        else:
            take("speculative", False,
                 "no GPU room for the draft beside the verifier — a CPU-resident draft makes "
                 "speculative slower than plain decode")
            take("draft_gpu", False, "insufficient VRAM for a GPU-resident draft")
    else:
        take("speculative", False, "no draft model or MTP head available")
        take("draft_gpu", False, "no draft to place")

    # ---- 6. attention implementation ------------------------------------------------------
    if str(arch).lower() == "gpt_oss":
        take("attn", "eager",
             "gpt-oss attention SINKS are applied only by the eager kernel; SDPA silently drops "
             "s_aux and the sink mass is lost")
    else:
        take("attn", "sdpa", "SDPA picks the best available backend for this device")

    # ---- 7. prefix reuse ------------------------------------------------------------------
    take("prefix_min", 128,
         "1024 made #prefix-kv inert (0 hits in 1816 live requests, mean prompt 353 tokens); a "
         "miss costs at most one crop round-trip, so a low gate cannot regress")

    # ---- 8. things measured as pessimizations — pinned OFF --------------------------------
    # NOTE: head_quant is NOT taken here. It used to be, and because take() is last-wins
    # (`out[name] = value`), that call silently overwrote section 4b above — which is the whole
    # point of 4b. The overwrite predated the #8 measurement: it proposed int4 on ROCm back when
    # an int4 head still looked like the bandwidth win. #8 then measured that head at +0.0389 nats
    # (92.5% of the damage the entire int4 body does, 15.3% of greedy tokens flipped) and it is now
    # REFUSED by client.py's head_quant validator — so the stale line made ⚡ propose a value the
    # loader rejects on ROCm, and erased the useful int8 suggestion everywhere else.
    # Anything added to this section must not re-take a knob an earlier section already owns.
    take("cuda_graph", False,
         "graphed decode measured 0.11x-0.95x: its StaticCache mirror attends over the full "
         "maxlen every token, which swamps the dispatch saving")

    return out, why


def format_rationale(knobs: dict, why: list[str], *, prefix: str = "[perf-auto]") -> str:
    """One-line-per-decision block for the load log."""
    lines = [f"{prefix} resolved {len(knobs)} knobs from the detected setup:"]
    lines += [f"{prefix}   {w}" for w in why]
    return "\n".join(lines)
