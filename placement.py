"""Model-spec + layer-placement planner for InfiniteModel's controller (split out of server.py, #38).

Holds the pure, dependency-light "where does this model land" layer: ModelSpec (architecture +
byte-sizing, incl. per-quant footprint), the MODEL_SPECS table, the planner dataclasses (NodeMem,
StageAssign, PlanResult), and the placement functions (plan_pipeline / _plan_vram_first /
_describe_plan / _node_layer_capacity) plus the #76 pre-load guardrail (_assess_placement /
_round_ctx). Everything here is a PURE function of its inputs — no controller globals (registry,
engine, sockets), only stdlib — so it's cheap to import and easy to test in isolation.

Controller-only: listed in server.py's EXTRA_UPDATE_FILES so the multi-file self-update keeps it in
sync on the controller. server.py imports from here with a one-cycle convergence bridge (fetch-to-disk
then import) so a node that swapped in the new server.py before this file propagated still starts.

The sizing constants below MIRROR server.py's (they are universal, not config) so this module stays
self-contained and free of a circular import back into server.py.
"""

from dataclasses import dataclass, field, replace
from typing import Optional

GB = 1024 ** 3
FRAMEWORK_OVERHEAD_GB = 1.0
WEIGHT_DTYPE_BYTES = 2
KV_DTYPE_BYTES = 2
DEFAULT_CTX = 8192

def _is_kvless_layer_type(layer_type) -> bool:
    """Does this layer_types entry hold FIXED-SIZE state instead of a ctx-scaled K/V?

    ⚠⚠ This MUST stay character-identical to the worker's `shard_build._is_linear_attn_type`.
    The worker is the ENFORCER — it builds the real cache and its `kv_reserve_probe` fails a
    stage cleanly — so the controller may over-reserve relative to it but must NEVER
    under-reserve. A first draft of this used a frozenset naming `mamba`/`recurrent`/
    `gated_deltanet`/`short_conv`/`conv` as well; six of those seven strings do not contain
    "linear", so the planner would have funded such a layer at ZERO while the worker reserved a
    full ctx-scaled KV for it. That is a decode-time OOM, and it would have been introduced by
    the very change meant to stop this kind of drift.

    Substring, not a set, for the reason the worker documents: a future `linear_*` variant is
    then caught by both sniffs by default rather than by one of them. Deliberately NARROW —
    every other non-`full_attention` kind is still softmax attention holding K/V,
    `sliding_attention` above all (bounded by the window, not absent).
    """
    return "linear" in str(layer_type).lower()


@dataclass
class ModelSpec:
    name: str
    hidden_size: int
    num_layers: int
    num_heads: int
    num_kv_heads: int
    head_dim: int
    intermediate_size: int
    vocab_size: int
    tie_embeddings: bool
    max_ctx: int = 32768
    arch: str = "qwen2"
    attn_bias: bool = True
    weight_dtype_bytes: int = WEIGHT_DTYPE_BYTES
    # Measured overrides (bytes), filled from the model's real safetensors headers
    # by spec_with_measurements(). When set, the dense formula below is bypassed —
    # this is what makes the planner correct for MoE (N experts/layer) and exact for
    # every architecture, since the formula only models a single dense MLP per layer.
    meas_layer_w: Optional[int] = None    # average weight bytes per decoder layer
    meas_embed: Optional[int] = None
    meas_head: Optional[int] = None
    meas_norm: Optional[int] = None
    meas_params: Optional[int] = None     # REAL param count (sum prod(shape) from the safetensors
    #                                       headers) — exact + dtype-agnostic, unlike bytes/dtype_bytes
    src_dtype: Optional[str] = None       # on-disk weight dtype (F32/BF16/F16/...): drives the load
    #                                       dtype (fp32 stays fp32 at quant=none) + the card display
    is_embedding: bool = False            # encoder/sentence-embedding model (BERT-family): served by
    #                                       the single-node embedding path (no pipeline/TP/KV/lm_head)
    # #kv-slots: per-request KV slot count C this spec is being PLANNED for (set via
    # for_kv_slots at load; 1 = the classic single-stream figure, bit-identical). Baked into
    # kv_bytes_per_layer so EVERY controller-side consumer — plan_pipeline's required bytes,
    # _fit_ctx, _assess_placement, the co-located colo_need math, the /status kv_reserved
    # card — reserves C x per-stream KV automatically, exactly mirroring the worker's own
    # shard_build._kv_bytes_per_layer xC choke point.
    kv_slots: int = 1
    # #kimi-linear: fraction of decoder layers that actually grow a full-ctx KV cache. <1.0 only
    # for linear-attention hybrids, where the remaining layers hold a FIXED-SIZE recurrent +
    # short-conv state instead (Kimi-Linear: 7 of 27 full-attention -> 0.259). An average, applied
    # uniformly per stage: the planner is an ESTIMATOR and the worker's kv_reserve_probe is the
    # exact per-layer enforcer that fails a stage cleanly and triggers a replan. 1.0 = every
    # existing model, bit-identical.
    kv_layer_frac: float = 1.0
    # #perf-facts: two arch facts perf_profile.resolve() asks for on every load and every
    # /optimize_knobs. Until 2026-08-17 BOTH call sites read them off the spec with
    # getattr(spec, "layer_types", None) / getattr(spec, "is_multimodal", False) — and ModelSpec
    # declared NEITHER field, so both were structurally ALWAYS the default. The resolver's
    # "multimodal/hybrid arch -> clamp kv_slots to 1" branch (perf_profile.py:211) could
    # therefore never fire, and kv_slots is an APPLIED knob: a VLM could be auto-assigned C=2/3,
    # reserving 2-3x full-ctx KV for prefix reuse that #prefix-kv gates off for exactly those
    # arches. Carried as real fields so there is ONE place that knows, populated by
    # model_store.spec_from_config from the config it already parses.
    layer_types: Optional[tuple] = None   # transformers' per-layer attention kinds, as declared
    is_multimodal: bool = False           # checkpoint carries a vision/audio/talker tower

    @property
    def per_layer_weight_bytes(self) -> int:
        if self.meas_layer_w is not None:
            return self.meas_layer_w
        h = self.hidden_size
        qd = self.num_heads * self.head_dim
        kvd = self.num_kv_heads * self.head_dim
        attn = h * qd + h * kvd + h * kvd + qd * h
        bias = (qd + kvd + kvd) if self.attn_bias else 0
        mlp = 3 * h * self.intermediate_size
        norms = 2 * h
        return (attn + bias + mlp + norms) * self.weight_dtype_bytes

    @property
    def embed_bytes(self) -> int:
        if self.meas_embed is not None:
            return self.meas_embed
        return self.vocab_size * self.hidden_size * self.weight_dtype_bytes

    @property
    def head_bytes(self) -> int:
        if self.meas_head is not None:
            return self.meas_head
        return 0 if self.tie_embeddings else self.embed_bytes

    @property
    def final_norm_bytes(self) -> int:
        if self.meas_norm is not None:
            return self.meas_norm
        return self.hidden_size * self.weight_dtype_bytes

    def kv_bytes_per_layer(self, ctx_len: int) -> int:
        # #kv-slots: C independent per-request streams each grow their own full-ctx KV -> xC.
        return int(2 * self.num_kv_heads * self.head_dim * ctx_len * KV_DTYPE_BYTES
                   * max(1, int(self.kv_slots or 1)) * float(self.kv_layer_frac or 1.0))

    def per_layer_total_bytes(self, ctx_len: int) -> int:
        return self.per_layer_weight_bytes + self.kv_bytes_per_layer(ctx_len)

    @property
    def total_weight_bytes(self) -> int:
        return (self.num_layers * self.per_layer_weight_bytes
                + self.embed_bytes + self.head_bytes + self.final_norm_bytes)

    @property
    def param_count(self) -> int:
        # Prefer the MEASURED count (exact, from tensor shapes). The bytes/dtype fallback assumes
        # a 2-byte (bf16) source, so it doubles for an fp32 checkpoint and halves for fp8 — only
        # used when the model wasn't measured (no local safetensors yet).
        if self.meas_params is not None:
            return self.meas_params
        return self.total_weight_bytes // self.weight_dtype_bytes

    def for_quant(self, quant: str) -> "ModelSpec":
        """Return a spec whose WEIGHT bytes reflect `quant`, so the planner sizes (and
        packs) the quantized footprint. KV cache is unaffected (it stays bf16 activations).
        int8 weight-only halves the Linear weights (decoder layers + lm_head). int4 is
        group-wise ~4.25-bit (~0.27x bf16) on the decoder layers; the lm_head stays bf16
        (logit-sensitive). int2 (#int2) is group-wise ~2.5-bit / group 64 (~0.16x bf16) on the
        decoder layers, head bf16 like int4 (sized at 0.2x for planner headroom, mirroring
        int4's 0.27->0.3 cushion). Embeddings and norms are never quantized. Works on measured
        or formula-based specs alike."""
        if quant == "int8":
            return replace(self,
                           meas_layer_w=self.per_layer_weight_bytes // 2,
                           meas_head=self.head_bytes // 2,
                           meas_embed=self.embed_bytes,
                           meas_norm=self.final_norm_bytes)
        if quant == "int4":
            return replace(self,
                           meas_layer_w=self.per_layer_weight_bytes * 3 // 10,  # ~4.25-bit + group scale/zero
                           meas_head=self.head_bytes,        # head kept bf16
                           meas_embed=self.embed_bytes,
                           meas_norm=self.final_norm_bytes)
        if quant == "int2":
            return replace(self,
                           meas_layer_w=self.per_layer_weight_bytes * 2 // 10,  # ~2.5-bit + group scale/zero
                           meas_head=self.head_bytes,        # head kept bf16
                           meas_embed=self.embed_bytes,
                           meas_norm=self.final_norm_bytes)
        return self

    def for_kv_slots(self, slots: int) -> "ModelSpec":
        """#kv-slots: return a spec whose KV bytes reflect `slots` concurrent per-request KV
        streams (C x per-stream full-ctx KV — see kv_bytes_per_layer). Weights are unaffected.
        slots<=1 returns self unchanged (bit-identical planning for every legacy load)."""
        slots = max(1, int(slots or 1))
        if slots <= 1:
            return self
        return replace(self, kv_slots=slots)

    def for_kv_quant(self, kv_quant: str) -> tuple["ModelSpec", str]:
        """#172 TurboQuant: return (spec sized for a packed KV cache, note if it was REFUSED).

        Returns the ratio baked into `kv_layer_frac`, because that is the one knob scaling
        kv_bytes_per_layer for EVERY controller-side consumer at once — plan_pipeline's required
        bytes, _fit_ctx, _assess_placement's warnings and suggested_ctx, and the footprint table —
        so a fit and the figures printed beside it cannot disagree.

        WHY THIS IS A METHOD. It was written inline in /plan only, so the Preview sized KV the way
        the worker really reserves it while the LIVE load planner still sized the same load at
        bf16. In the narrow band where packed KV fits and bf16 does not, Preview said "fits" and
        /load then raised CapacityError on the same numbers — the two planners disagreeing about
        one model. Both now call this.

        The hybrid gate is INSIDE the helper on purpose. It mirrors shard_build's
        `_kvq_on = kv_quant != none and not self._hybrid`: a hybrid arch never builds a TurboQuant
        cache, so sizing one claims a fit the worker then reserves at bf16 and OOMs on. Leaving
        that gate to the callers is what produced the split above — the caller that has the rule
        shrinks and the caller that lacks it does not, which is worse than neither shrinking.

        `note` is non-empty exactly when a preset was asked for and refused, so the caller can say
        WHY rather than silently planning something other than what was requested.
        """
        kvq = (kv_quant or "none").strip().lower()
        if kvq not in ("turbo2", "turbo3", "turbo4"):
            return self, ""       # 'none' and every unrecognised preset -> bf16, never a smaller plan
        if self.is_hybrid:
            return self, ("kv_quant ignored — a hybrid / sliding-attention arch never builds a "
                          "TurboQuant cache, so the worker reserves KV at bf16")
        try:
            # Lazy: kv_quant.py is a controller leaf and placement.py is imported by the worker's
            # planner path too. Ask it for the STORED bytes rather than assuming bits/16 — the
            # indices are bit-packed to a byte boundary and each head carries an fp16 norm, so
            # turbo3 is not 3/16. Same function the worker reserves with (shard_build).
            import kv_quant as _kq
            pt = _kq.kv_quant_bytes_per_token_per_layer(kvq, self.num_kv_heads, self.head_dim)
        except Exception:   # noqa: BLE001 — sizing must never break planning; bf16 is conservative
            return self, "kv_quant ignored — its sizing helper could not be loaded; KV sized at bf16"
        bf = 2 * self.num_kv_heads * self.head_dim * KV_DTYPE_BYTES
        if pt <= 0 or bf <= 0:
            return self, "kv_quant ignored — sizing returned no packed figure; KV sized at bf16"
        # + one bf16 dequant transient amortized over the layers: the worker holds one full-ctx
        # bf16 layer per device owning any KV layer, and /status's #172 card counts it the same.
        r = pt / bf + 1.0 / max(1, self.num_layers)
        return replace(self, kv_layer_frac=float(self.kv_layer_frac or 1.0) * r), ""

    # NOTE: is_hybrid and full_attention_layers answer DIFFERENT questions and deliberately use
    # DIFFERENT predicates. Conflating them is the bug this pair exists to prevent — a
    # sliding-window layer is hybrid (no prefix reuse) but DOES grow a ctx-scaled KV.
    @property
    def is_hybrid(self) -> bool:
        """Does this arch deviate from uniform full-attention on every layer?

        Broad by design: ANY non-full_attention entry counts (sliding-window Gemma-4 / gpt-oss /
        Mistral just as much as linear-attention qwen3-next), plus the fla-style archs that
        declare their linear layers via linear_attn_config instead of layer_types — those reach
        us as kv_layer_frac < 1.0 (model_store.spec_from_config computes it from kda_layers).

        This is the predicate for "#prefix-kv is gated off here", which is why it must catch
        sliding-window: those arches are gated off prefix reuse too. It is NOT the predicate for
        "which layers grow a KV" — see full_attention_layers.
        """
        if any(str(t) != "full_attention" for t in (self.layer_types or ())):
            return True
        return float(self.kv_layer_frac or 1.0) < 1.0

    @property
    def full_attention_layers(self) -> int:
        """How many layers grow a KV cache that SCALES WITH CONTEXT.

        Narrow by design, and CONSERVATIVE in the only direction that is safe: a layer kind we
        do not recognise counts as full KV, so a new arch over-reserves (a refused placement)
        rather than under-reserves (a decode OOM). Sliding-window layers COUNT — their KV is
        bounded by the window, not zero, and charging them zero is precisely the mistake that
        funded gpt-oss's 12-of-24 and Gemma-4's 20-of-24 sliding layers at 0 bytes.

        Only genuinely KV-less layers are excluded, via the worker's own predicate — see
        _is_kvless_layer_type for why that identity is load-bearing. Falls back to kv_layer_frac
        (the fla/kda path, which has no layer_types at all) and to num_layers for every dense
        model, bit-identically.
        """
        lt = self.layer_types or ()
        if len(lt) == self.num_layers:   # trust it only when it describes THIS model's depth
            return sum(1 for t in lt if not _is_kvless_layer_type(t))
        return max(0, int(round(self.num_layers * float(self.kv_layer_frac or 1.0))))


MODEL_SPECS: dict[str, ModelSpec] = {
    "Qwen/Qwen2.5-0.5B-Instruct": ModelSpec(
        "Qwen2.5-0.5B", 896, 24, 14, 2, 64, 4864, 151936, tie_embeddings=True),
    "Qwen/Qwen2.5-1.5B-Instruct": ModelSpec(
        "Qwen2.5-1.5B", 1536, 28, 12, 2, 128, 8960, 151936, tie_embeddings=True),
    "Qwen/Qwen2.5-Coder-1.5B-Instruct": ModelSpec(
        "Qwen2.5-Coder-1.5B", 1536, 28, 12, 2, 128, 8960, 151936, tie_embeddings=True),
    "Qwen/Qwen2.5-7B-Instruct": ModelSpec(
        "Qwen2.5-7B", 3584, 28, 28, 4, 128, 18944, 152064, tie_embeddings=False),
    "Qwen/Qwen2.5-Coder-32B-Instruct": ModelSpec(
        "Qwen2.5-Coder-32B", 5120, 64, 40, 8, 128, 27648, 152064, tie_embeddings=False),
    # Llama-3.1-70B dims (Nemotron-70B is a fine-tune, same shape). No attn bias,
    # not tied. Used only by the planner's byte estimate — the loader builds from
    # the real config.json, so verify these against config.json before the load.
    "nvidia/Llama-3.1-Nemotron-70B-Instruct-HF": ModelSpec(
        "Llama-3.1-Nemotron-70B", 8192, 80, 64, 8, 128, 28672, 128256,
        tie_embeddings=False, arch="llama", attn_bias=False, max_ctx=131072),
    # DeepSeek-R1-Distill-Llama-70B = Llama-3.3-70B dims (same shape as Nemotron above).
    "deepseek-ai/DeepSeek-R1-Distill-Llama-70B": ModelSpec(
        "DeepSeek-R1-Distill-Llama-70B", 8192, 80, 64, 8, 128, 28672, 128256,
        tie_embeddings=False, arch="llama", attn_bias=False, max_ctx=131072),
    # MoE: intermediate_size is the dense-formula fallback only — the planner uses
    # MEASURED per-layer bytes (all 8 experts) at load. Attention dims + n_layers are
    # what matter here (KV math). Mixtral: 32 layers, GQA 32/8 heads, head_dim 128.
    "mistralai/Mixtral-8x7B-Instruct-v0.1": ModelSpec(
        "Mixtral-8x7B", 4096, 32, 32, 8, 128, 14336, 32000,
        tie_embeddings=False, arch="mixtral", attn_bias=False, max_ctx=32768),
    # OLMoE 1B-7B: 64 experts/layer (intermediate_size is per-expert; the planner MEASURES
    # the real per-layer bytes from the safetensors headers). 16 heads, no GQA, no attn bias.
    "allenai/OLMoE-1B-7B-0924-Instruct": ModelSpec(
        "OLMoE-1B-7B", 2048, 16, 16, 16, 128, 1024, 50304,
        tie_embeddings=False, arch="olmoe", attn_bias=False, max_ctx=4096),
    # Qwen3.6-35B-A3B (model_type qwen3_5_moe): 256 experts/8 active (moe_intermediate 512 is
    # the dense-formula fallback only — planner MEASURES real per-layer bytes). GQA 16/2,
    # head_dim 256 (independent of hidden), no attn bias, untied, native ctx 262144. The
    # checkpoint is multimodal; client loads only the text LM (language_model.* remap).
    # ⚠ A hard-coded entry SHORT-CIRCUITS the config: resolve_spec returns MODEL_SPECS[id] without
    # ever reading config.json, so any arch fact missing here is missing for good. This entry said
    # nothing about either, and both were wrong — READ FROM THE REAL CHECKPOINT 2026-08-17:
    # text_config.layer_types is 30 linear_attention + 10 full_attention of 40, and the top level
    # carries vision_config/image_token_id/video_token_id. So the planner charged KV on 4x the
    # layers that grow one, and perf_profile was told the model was neither hybrid nor multimodal.
    # kv_layer_frac (not layer_types) is the honest encoding: only the COUNTS are consumed, and
    # fabricating a per-layer order here would be data nobody measured.
    "Qwen/Qwen3.6-35B-A3B": ModelSpec(
        "Qwen3.6-35B-A3B", 2048, 40, 16, 2, 256, 512, 248320,
        tie_embeddings=False, arch="qwen3_5_moe", attn_bias=False, max_ctx=262144,
        kv_layer_frac=10 / 40, is_multimodal=True),
}


@dataclass
class NodeMem:
    node_id: str
    hostname: str
    usable_bytes: int          # total usable (RAM + usable VRAM)
    vram_bytes: int = 0        # usable VRAM portion (0 for CPU-only nodes)
    pref: int = 0              # placement preference (#50): higher = picked first when consolidating


def _mem_pref(node) -> int:
    """Placement-preference rank for picking/ordering nodes (CPU spill / consolidate): FASTER
    MEMORY FIRST, by DDR GENERATION — DDR5 > DDR4 > DDR3 > ... (LPDDR5 counts as gen 5, etc.).
    CPU decode is memory-bandwidth-bound, so a newer-DDR host runs CPU-resident layers faster.
    Generation is the primary key; beast (the controller box — no LAN hop to fetch its own
    weights) gets a small within-generation tiebreak so it leads among equal-DDR hosts. Used as
    the primary sort key (then usable_bytes), so a model lands on the fastest memory available."""
    import re
    r = (getattr(node, "ram", "") or "").upper()
    m = re.search(r"DDR(\d)", r)          # DDR5/LPDDR5->5, DDR4/LPDDR4->4, DDR3->3; unknown->0
    gen = int(m.group(1)) if m else 0
    pref = gen * 10                        # generation dominates: 50 > 40 > 30 > 0
    if (node.hostname or "").lower() == "beast":
        pref += 1                          # controller-local: edge out other same-generation hosts
    return pref


def _node_tp_bw(node, is_gpu: bool) -> float:
    """Rough memory/compute BANDWIDTH proxy (GB/s) for sizing a tensor-parallel rank's slice (#87).
    A lockstep TP mesh runs at its SLOWEST rank, so slices must be proportional to BANDWIDTH (not
    capacity / VRAM-RAM GB, #68's metric — that hands the slow CPU the biggest slice -> straggler).
    GPU ranks: coarse device-name tier (HBM/GDDR, hundreds of GB/s). CPU ranks: DDR generation
    (decode is RAM-bandwidth-bound). Tunable proxy; the #40 crossover bench refines the constants."""
    if is_gpu:
        name = (getattr(node, "device_name", "") or "").lower()
        for key, gbps in (("4090", 1000.0), ("4080", 720.0), ("4070", 600.0), ("3090", 900.0),
                          ("3080", 760.0), ("3070", 450.0), ("3060", 360.0), ("a100", 1500.0),
                          ("a6000", 770.0), ("v100", 900.0), ("2080", 450.0), ("2070", 400.0),
                          ("1080", 320.0), ("1070", 256.0)):
            if key in name:
                return gbps
        return 400.0   # unknown GPU: a sane mid default (still far above any CPU rank)
    import re
    m = re.search(r"DDR(\d)", (getattr(node, "ram", "") or "").upper())
    gen = int(m.group(1)) if m else 0
    return {5: 70.0, 4: 45.0, 3: 25.0}.get(gen, 35.0)   # rough dual-channel desktop GB/s


# --- #unified-mem (audit #17): APU shared-pool detection + plannable-budget clamp ------------
# amdgpu sizes the GTT pool at ~1/2 of system RAM, so an APU's mem_get_info "VRAM total" lands
# near 0.5x host RAM (om3nbox gfx1151: ~60/128 ≈ 0.47; steamdeck Van Gogh: ~8/16 ≈ 0.5). 0.35
# leaves slack for BIOS carve-outs / smaller GTT limits while staying far above any discrete
# card on this fleet (beast 4070TiS 16/128 ≈ 0.13).
UNIFIED_MEM_VRAM_RATIO = 0.35


def is_unified_mem_node(device_name: str, vram_total_gb: float, total_mem_gb: float,
                        explicit=None) -> bool:
    """True when a node's reported "VRAM" is a GTT slice CARVED FROM THE SAME physical RAM
    psutil reports (APUs: om3nbox's Strix Halo gfx1151, the steamdeck's Van Gogh). A planner
    that sums free RAM + free VRAM on such a node double-counts the one pool (an idle om3nbox
    reads ~165 GB "usable" on a 128 GB box) and over-commits into a cgroup OOM-kill mid-load.

    `explicit` is a worker-reported flag and WINS when present (bool, either way); no worker
    ships one yet (heartbeat has no unified_mem field), so detection falls to an HONEST
    HEURISTIC over registration data the controller already has:
      AMD device name ("amd"/"radeon"/"gfx" — an NVIDIA name never matches, so every CUDA
      node is exempt by construction) AND vram_total >= UNIFIED_MEM_VRAM_RATIO x host RAM
      (the GTT-pool signature above; a discrete card's VRAM sits far below it).
    KNOWN LIMIT, stated honestly: a big discrete Radeon beside little RAM (e.g. a 24 GB
    7900XTX on a 32 GB box) would false-positive — none exists on this fleet today, and the
    failure mode is UNDER-budgeting that node (conservative), never an OOM."""
    if explicit is not None:
        return bool(explicit)
    if not vram_total_gb or not total_mem_gb or vram_total_gb <= 0 or total_mem_gb <= 0:
        return False
    dn = (device_name or "").lower()
    if not ("amd" in dn or "radeon" in dn or "gfx" in dn):
        return False
    return vram_total_gb >= UNIFIED_MEM_VRAM_RATIO * total_mem_gb


def clamp_unified_usable_gb(usable_gb: float, free_mem_gb: float,
                            vram_reusable_gb: float = 0.0, reserve_gb: float = 0.0) -> float:
    """Clamp a unified-memory node's plannable budget (RAM share + "VRAM" share summed by the
    caller) to the LIVE physically-free bytes of the ONE pool: psutil free (GTT-pinned pages
    already depress it — which is exactly why the clamp keys off the LIVE figure, not a static
    total-minus-reserves) + reusable bytes held by the worker process (its vacant torch
    allocator pool, #vram-reusable — invisible to psutil-free yet a new load allocates from
    it; a re-place also passes its own reclaimable shard bytes here) - the same reserves the
    caller already charged its RAM/VRAM budgets. Pure + monotonic: never RAISES a budget
    (min against the caller's figure) and never returns < 0."""
    phys_free_gb = free_mem_gb + max(0.0, vram_reusable_gb) - max(0.0, reserve_gb)
    return max(0.0, min(usable_gb, phys_free_gb))


@dataclass
class StageAssign:
    node_id: str
    hostname: str
    layer_start: int
    layer_end: int
    has_embed: bool
    has_head: bool
    est_bytes: int
    usable_bytes: int
    gpu_bytes: int = 0          # worker-reported VRAM this stage actually placed (filled at load)
    gpu_kv_bytes: int = 0       # worker-reported full-ctx KV reserved on GPU (coexistence VRAM reserve)
    # #real-stats: worker-MEASURED total weight bytes of this stage (params+buffers, post-quant).
    # 0 = not reported (old worker) -> consumers fall back to the spec ESTIMATE. Needed because
    # spec.total_weight_bytes is a formulaic quant estimate that can overshoot the real packed
    # size by ~10% on MoE int4 — subtracting measured gpu_bytes from it fabricated a phantom
    # "weights on CPU" (1.9 GB / cpu_frac 0.106 on a fully-GPU-resident qwen3-30b-a3b).
    loaded_bytes: int = 0

    @property
    def num_layers(self) -> int:
        return self.layer_end - self.layer_start

    @property
    def est_gb(self) -> float:
        return self.est_bytes / GB

    @property
    def usable_gb(self) -> float:
        return self.usable_bytes / GB

    @property
    def headroom_gb(self) -> float:
        return (self.usable_bytes - self.est_bytes) / GB

    def to_dict(self) -> dict:
        return {
            "node_id": self.node_id, "hostname": self.hostname,
            "layer_start": self.layer_start, "layer_end": self.layer_end,
            "num_layers": self.num_layers, "has_embed": self.has_embed,
            "has_head": self.has_head, "est_gb": round(self.est_gb, 2),
            "usable_gb": round(self.usable_gb, 2), "headroom_gb": round(self.headroom_gb, 2),
            "gpu_gb": round(self.gpu_bytes / GB, 2),   # VRAM this stage actually placed on-GPU
        }


@dataclass
class PlanResult:
    ok: bool
    model: str
    ctx_len: int
    num_layers: int
    pool_usable_gb: float
    required_gb: float
    stages: list[StageAssign] = field(default_factory=list)
    error: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "ok": self.ok, "model": self.model, "ctx_len": self.ctx_len,
            "num_layers": self.num_layers,
            "pool_usable_gb": round(self.pool_usable_gb, 2),
            "required_gb": round(self.required_gb, 2),
            "stages": [s.to_dict() for s in self.stages], "error": self.error,
        }


def _node_layer_capacity(node: NodeMem, fixed_bytes: int, per_layer_bytes: int) -> int:
    avail = node.usable_bytes - fixed_bytes
    return 0 if avail <= 0 else avail // per_layer_bytes


def _plan_vram_first(spec: ModelSpec, nodes: list[NodeMem], ctx_len: int) -> PlanResult:
    """GPU-VRAM-first placement: keep as many layers as possible on the FAST tier
    (GPU VRAM, across every GPU node) and spill the remainder to CPU RAM. A layer
    on CPU is ~50-100x slower than on GPU, so minimizing CPU layers — even at the
    cost of an extra pipeline hop — wins. Three cases:
      1. model fits the GPU VRAM pool  -> use only GPU nodes, sized to their VRAM
         (whole model on GPU, multi-GPU pipeline).
      2. model exceeds GPU VRAM but fits GPU nodes' RAM+VRAM -> GPU nodes only,
         VRAM full + spill to their own RAM (worker handles the per-node split).
      3. bigger still -> GPU nodes (VRAM+RAM) + the fewest CPU nodes that fit."""
    overhead = int(FRAMEWORK_OVERHEAD_GB * GB)
    gpu = sorted((n for n in nodes if n.vram_bytes > 0),
                 key=lambda n: n.vram_bytes, reverse=True)
    cpu = sorted((n for n in nodes if n.vram_bytes <= 0),
                 key=lambda n: n.usable_bytes, reverse=True)
    if not gpu:
        return plan_pipeline(spec, nodes, ctx_len, consolidate=True)

    # Case 1/2: fit the whole model in GPU VRAM using the FEWEST GPU nodes (so a
    # model fitting one card stays on one card; a bigger one spans just enough
    # GPUs — all weights on the fast tier). Pass usable = vram + overhead so the
    # planner's framework-overhead subtraction doesn't dock VRAM (that overhead
    # is RAM, not VRAM); embed/head are real weights and still count.
    for k in range(1, len(gpu) + 1):
        vmems = [NodeMem(n.node_id, n.hostname, n.vram_bytes + overhead, n.vram_bytes)
                 for n in gpu[:k]]
        r = plan_pipeline(spec, vmems, ctx_len)
        if r.ok:
            return r

    # Case 3: too big for the GPU VRAM pool -> use ALL GPU nodes (max their VRAM)
    # plus the fewest CPU nodes needed; each worker spills its own overflow to RAM.
    # vram_first=True: fill every GPU's VRAM before spilling, so a small-VRAM/big-RAM
    # node (beast) can't hog layers and leave another GPU's VRAM half-empty.
    for k in range(0, len(cpu) + 1):
        r = plan_pipeline(spec, gpu + cpu[:k], ctx_len, vram_first=True)
        if r.ok:
            return r
    return plan_pipeline(spec, gpu + cpu, ctx_len, vram_first=True)  # none fit -> error


def _describe_plan(stages, node_by_id: dict, cpu_only: bool, prefer_vram: bool,
                   quant: str, gpu_spread: bool = False) -> str:
    """One-line, human-readable BASIS for a placement plan (#65): the strategy the planner
    actually used (auto GPU-first / CPU-only RAM / RAM-first), the shape (single node vs a
    distributed pipeline across N), and each stage's target tier + layer count. Surfaced in
    the activity feed and on the dashboard so a load explains its own placement instead of the
    user having to infer it from "handing out shards". Tier is the node's DIRECTED device (a
    cpu+gpu node loads GPU-first with RAM spill -> shown as GPU); the post-load shard log still
    reports the actual GPU/RAM byte split the worker chose."""
    n = len(stages)
    if cpu_only:
        strat = "CPU-only (RAM, no VRAM)"
    elif gpu_spread:
        strat = "all-GPU (every GPU, no CPU spill)"
    elif prefer_vram:
        strat = "auto / GPU-first (fill VRAM, spill to RAM)"
    else:
        strat = "RAM-first"
    parts = []
    for s in stages:
        nd = node_by_id.get(s.node_id)
        if cpu_only or nd is None:
            tier = "RAM"
        else:
            dev = nd.load_device()
            tier = "RAM" if dev == "cpu" else ("GPU" if (dev == "gpu" or nd.eff_vram_gb > 0)
                                               else "RAM")
        parts.append(f"{s.hostname}:{tier}({s.num_layers}L)")
    shape = "single node" if n == 1 else f"distributed pipeline, {n} nodes"
    q = "" if (quant in (None, "none", "")) else f", {quant}"
    return f"{strat} -> {shape}{q}: " + ", ".join(parts)


def _round_ctx(c: int) -> int:
    """Round a context length DOWN to a tidy value for an auto-cap suggestion."""
    for v in (524288, 262144, 131072, 65536, 32768, 16384, 8192, 4096, 2048, 1024):
        if c >= v:
            return v
    return int(max(0, c))


def _assess_placement(spec: ModelSpec, ctx: int, mems: list, stages: list,
                      cpu_only: bool = False) -> dict:
    """#76 PRE-LOAD GUARDRAIL. For the chosen placement, ESTIMATE how much of the weights and the
    full-ctx KV cache will sit in GPU VRAM vs spill to CPU RAM — BEFORE the load commits (workers
    only report the real gpu_bytes after loading). plan_pipeline only checks that weights+KV fit
    each node's TOTAL RAM+VRAM, so a successful plan still hides two failure modes:
      (A) huge KV (big ctx) that "fits" only by landing in RAM on GPU stages -> the GPU gathers KV
          from RAM every token -> enormous first-token latency / hang (deepseek-70b at 128K).
      (B) weights bigger than the fleet's VRAM -> most layers run on CPU -> unusably slow
          (deepseek-70b int4 put 57% of weights on CPU -> 0.11 tok/s).
    Mirrors the worker's fill order (VRAM holds weights first, then KV). Returns metrics + human
    warnings + suggested_ctx (largest tidy ctx that keeps KV in VRAM on the tightest GPU stage)."""
    vram_by_id = {m.node_id: m.vram_bytes for m in mems}
    kv_layer_tok = int(2 * spec.num_kv_heads * spec.head_dim * KV_DTYPE_BYTES
                       * float(spec.kv_layer_frac or 1.0))   # KV bytes / layer / token
    tot_w = tot_w_ram = tot_kv = tot_kv_ram = 0
    gpu_stage_max_ctx = []
    for s in stages:
        nl = s.num_layers
        w = nl * spec.per_layer_weight_bytes
        if s.has_embed:
            w += spec.embed_bytes
        if s.has_head:
            w += spec.head_bytes + spec.final_norm_bytes
        kv = nl * spec.kv_bytes_per_layer(ctx)
        vram = vram_by_id.get(s.node_id, 0)
        w_vram = min(w, vram)
        kv_vram = min(kv, max(0, vram - w_vram))     # VRAM fills weights-first, then KV
        tot_w += w
        tot_w_ram += (w - w_vram)
        tot_kv += kv
        tot_kv_ram += (kv - kv_vram)
        if vram > 0:                                 # a GPU stage with no KV headroom binds ctx to 0
            free_for_kv, denom = vram - w, nl * kv_layer_tok
            gpu_stage_max_ctx.append(free_for_kv // denom if (free_for_kv > 0 and denom > 0) else 0)
    cpu_w_frac = (tot_w_ram / tot_w) if tot_w else 0.0
    weight_bound = cpu_w_frac > 0.05                 # weights overflow VRAM -> capping ctx won't help
    suggested_ctx = _round_ctx(min(gpu_stage_max_ctx) if gpu_stage_max_ctx else 0)
    warnings = []
    # cpu_only is an INTENTIONAL RAM placement — plan_pipeline already guarantees the KV fits RAM
    # (and _fit_ctx caps ctx if not), and a CPU model reads KV from RAM normally, so there is no
    # GPU gather-from-RAM hang to warn about; it gets neither warning. weight_bound suppresses the
    # KV note because once weights overflow VRAM the honest fix is a smaller model/quant (the
    # weight-spill warning), not a lower ctx — there's no VRAM headroom to keep KV on the GPU.
    if weight_bound and not cpu_only:
        sev = " (SEVERE: most on CPU, <0.3 tok/s)" if cpu_w_frac > 0.5 else ""
        # Two SHORT warnings (each renders on its own ⚠ line) so the model card stays compact.
        warnings.append(
            f"{cpu_w_frac*100:.0f}% of weights (~{tot_w_ram/GB:.1f} GB) won't fit GPU VRAM "
            f"-> run on CPU, slow generation{sev}.")
        warnings.append(
            f"Needs ~{tot_w/GB:.0f} GB VRAM; use a smaller model/quant, or accept CPU speed.")
    elif (not weight_bound) and tot_kv_ram > 0.5 * GB:
        warnings.append(
            f"ctx={ctx}: KV cache is ~{tot_kv/GB:.1f} GB and ~{tot_kv_ram/GB:.1f} GB won't fit GPU "
            f"VRAM (spills to RAM) -> very high first-token latency / possible hang"
            + (f". Lower ctx to <= {suggested_ctx} to keep KV in VRAM." if suggested_ctx else "."))
    tier = "gpu" if cpu_w_frac <= 0.02 else ("mixed" if cpu_w_frac <= 0.5 else "cpu-bound")
    return {"warnings": warnings, "speed_tier": tier,
            "cpu_weight_frac": round(cpu_w_frac, 3), "cpu_weight_gb": round(tot_w_ram / GB, 2),
            "kv_total_gb": round(tot_kv / GB, 2), "kv_ram_gb": round(tot_kv_ram / GB, 2),
            "suggested_ctx": int(suggested_ctx), "weight_bound": weight_bound}


def plan_pipeline(spec: ModelSpec, nodes: list[NodeMem], ctx_len: int = DEFAULT_CTX,
                  consolidate: bool = False, prefer_vram: bool = False,
                  vram_first: bool = False, spread: bool = False,
                  proportional: bool = False, gpu_spread: bool = False) -> PlanResult:
    """Assign contiguous layer ranges across nodes proportional to usable memory.
    nodes[0] is stage 0 (embedding); the last node carries the final norm + LM head.

    prefer_vram=True (when any node has a GPU) keeps weights on GPU VRAM first,
    spilling to CPU RAM only for the overflow — see _plan_vram_first.

    consolidate=True picks the FEWEST, strongest nodes that still fit the model.
    Pipeline parallelism is sequential — a single-stream token must traverse every
    stage in series — so fewer stages on bigger boxes means lower per-token latency.
    When a model fits one box, this collapses the pipeline to a single stage.

    proportional=True (#78) is the BIG-MoE distributed mode: hand the layers to EVERY
    capable node in shares PROPORTIONAL to each node's layer-capacity, using largest-
    remainder (Hamilton) apportionment so a 30 GB box gets ~3x the layers of a 10 GB box
    and every node that can hold >=1 layer participates. Unlike plain `distribute` (whose
    floor-division drops small nodes to 0 when the pool dwarfs the model) and unlike
    `spread` (whose +1-round-robin remainder over-flattens, piling early layers onto tiny
    slow nodes), this keeps the split capacity-weighted AND fills the pool. Intended for a
    model too big for the GPU-first subset — e.g. MiniMax-M2 int4 across the whole fleet.
    Nodes are ordered biggest-capacity-first so the heaviest contiguous ranges + the embed
    (stage 0) and LM-head (last stage) land on the strongest boxes.

    gpu_spread=True (#all-gpu) is the "use EVERY GPU, nothing on CPU" mode: drop all CPU-only
    nodes, then lay the model out PROPORTIONALLY across the GPU subset so every GPU carries at
    least one layer (capacity-weighted — the biggest GPU still does the bulk). Unlike the
    `gpu-spread` mode (prefer_vram, which fills the biggest GPUs and spills the overflow to CPU),
    this never touches CPU RAM and guarantees a stage on each GPU. The trade-off is more pipeline
    hops (each adds per-token decode latency); its win is using all VRAM to avoid a CPU spill and
    to share prefill compute across cards. Fails cleanly if the model doesn't fit GPU VRAM alone."""
    if gpu_spread:
        nodes = [n for n in nodes if n.vram_bytes > 0]   # GPU nodes only — no CPU spill
        # Force the proportional (every-node, capacity-weighted) layout over the GPU subset and
        # disable the GPU-first-then-spill / fewest-nodes paths so the model spreads across cards.
        proportional, prefer_vram, consolidate = True, False, False
    if prefer_vram and any(n.vram_bytes > 0 for n in nodes):
        return _plan_vram_first(spec, nodes, ctx_len)
    if consolidate and nodes:
        # Prefer faster memory / beast first (#50), then by usable size — so a model that fits a
        # subset lands on the fastest hosts. pref defaults to 0 (size-only) for callers that don't set it.
        ordered = sorted(nodes, key=lambda n: (n.pref, n.usable_bytes), reverse=True)
        for k in range(1, len(ordered) + 1):
            r = plan_pipeline(spec, ordered[:k], ctx_len)  # try fewest nodes first
            if r.ok:
                return r
        return plan_pipeline(spec, ordered, ctx_len)  # none fit -> informative error
    overhead = int(FRAMEWORK_OVERHEAD_GB * GB)
    per_layer = spec.per_layer_total_bytes(ctx_len)
    L = spec.num_layers
    pool_usable = sum(n.usable_bytes for n in nodes)
    required = (spec.total_weight_bytes + L * spec.kv_bytes_per_layer(ctx_len)
                + len(nodes) * overhead)
    base = PlanResult(ok=False, model=spec.name, ctx_len=ctx_len, num_layers=L,
                      pool_usable_gb=pool_usable / GB, required_gb=required / GB)
    if not nodes:
        base.error = ("no GPU nodes available for all-GPU placement (every connected worker is "
                      "CPU-only — use a CPU-capable mode)") if gpu_spread else "no nodes connected"
        return base

    if spread and len(nodes) > 2:
        # Spread mode: a stage on EVERY node that can hold a layer (so even a tiny worker —
        # phone/tablet — joins the pipeline). Put the two BIGGEST nodes at the ends so stage 0
        # (embedding) and the last stage (LM head) — both heavy — never land on a tiny node;
        # the small nodes become cheap 1-layer middle stages.
        order = sorted(range(len(nodes)), key=lambda i: nodes[i].usable_bytes, reverse=True)
        nodes = [nodes[order[0]]] + [nodes[i] for i in order[2:]] + [nodes[order[1]]]
    elif proportional and len(nodes) > 1:
        # Proportional-spread (#78): biggest-capacity node first so the heaviest contiguous
        # range AND the embed (stage 0) land on the strongest box; the LM head (last stage)
        # then lands on the SMALLEST participating node. The head (untied) is heavier than the
        # embed for a tied model and equal otherwise, so swap the smallest end node out for the
        # 2nd-biggest when there are >=3 nodes — same end-protection idea as spread, but the
        # interior stays strictly capacity-ordered (descending) so the proportional shares below
        # map to contiguous ranges in size order.
        order = sorted(range(len(nodes)), key=lambda i: nodes[i].usable_bytes, reverse=True)
        if len(nodes) >= 3:
            nodes = ([nodes[order[0]]] + [nodes[i] for i in order[2:]] + [nodes[order[1]]])
        else:
            nodes = [nodes[i] for i in order]

    last = len(nodes) - 1
    fixed = []
    for i, _n in enumerate(nodes):
        f = overhead
        if i == 0:
            f += spec.embed_bytes
        if i == last:
            f += spec.head_bytes + spec.final_norm_bytes
        fixed.append(f)

    max_layers = [_node_layer_capacity(n, fixed[i], per_layer) for i, n in enumerate(nodes)]
    if sum(max_layers) < L:
        short = required - pool_usable
        base.error = (f"model needs ~{required/GB:.1f} GB but pool offers "
                      f"{pool_usable/GB:.1f} GB usable - short by ~{short/GB:.1f} GB "
                      f"(add nodes or lower context from {ctx_len})")
        return base

    if vram_first:
        # Fill every GPU's VRAM first (max layers on the fast tier), then spill the
        # remainder into nodes IN ORDER (GPU-by-VRAM, then CPU) so overflow lands on the
        # biggest-RAM node first — fewest pipeline hops, fastest CPU. Avoids the
        # capacity-proportional split under-filling a small-VRAM/big-RAM node's GPU.
        vcap = [(max(0, n.vram_bytes - fixed[i]) // per_layer) if n.vram_bytes > 0 else 0
                for i, n in enumerate(nodes)]
        counts = [min(vcap[i], max_layers[i]) for i in range(len(nodes))]
        leftover = L - sum(counts)
        for i in range(len(nodes)):
            if leftover <= 0:
                break
            add = min(leftover, max_layers[i] - counts[i])
            counts[i] += add
            leftover -= add
    elif spread:
        # one layer on every node that can hold one; the remainder is spread by the round-robin
        # below (so the biggest nodes carry the bulk, every node carries at least one).
        counts = [1 if max_layers[i] >= 1 else 0 for i in range(len(nodes))]
        if sum(counts) > L:        # more capable nodes than layers: keep the L biggest-capacity
            keep = set(sorted(range(len(nodes)),
                              key=lambda i: max_layers[i], reverse=True)[:L])
            counts = [1 if (i in keep and max_layers[i] >= 1) else 0 for i in range(len(nodes))]
        leftover = L - sum(counts)
    elif proportional:
        # Capacity-proportional shares via largest-remainder (Hamilton) apportionment (#78), but
        # FILL GPU VRAM FIRST so the fast tier is actually used and the CPU-RAM remainder spreads
        # across the fleet rather than under-filling GPUs. Phases: (0) keep the L biggest if there
        # are more capable nodes than layers; (1) GPU pre-fill — give each node as many layers as
        # its free VRAM holds (bounded by its total cap); (2) apportion the REMAINING layers by
        # leftover capacity via largest-remainder; (3) >=1 floor for every kept-capable node so the
        # whole pool participates; (4) shave any over-allocation (smallest remainder first, never
        # below 1); (5) hand any rounding leftover out by largest remainder. A node with 3x the
        # capacity still gets ~3x the layers, but GPUs fill before CPU RAM does.
        cap = [max_layers[i] for i in range(len(nodes))]
        capable = [i for i in range(len(nodes)) if cap[i] >= 1]
        if len(capable) > L:                 # more capable nodes than layers -> keep L biggest
            keep = set(sorted(capable, key=lambda i: cap[i], reverse=True)[:L])
            cap = [cap[i] if i in keep else 0 for i in range(len(nodes))]
            capable = [i for i in range(len(nodes)) if cap[i] >= 1]
        # (1) GPU pre-fill: free-VRAM layers per node (uses each node's free vram in the split).
        vcap = [(max(0, n.vram_bytes - fixed[i]) // per_layer) if n.vram_bytes > 0 else 0
                for i, n in enumerate(nodes)]
        counts = [min(vcap[i], cap[i]) for i in range(len(nodes))]
        # (2) proportional remainder over leftover (mostly CPU-RAM) capacity.
        rcap = [cap[i] - counts[i] for i in range(len(nodes))]
        Lrem = L - sum(counts)
        rsum = sum(rcap)
        ideal = [(Lrem * rcap[i] / rsum) if (Lrem > 0 and rsum > 0) else 0.0
                 for i in range(len(nodes))]
        if Lrem > 0 and rsum > 0:
            for i in range(len(nodes)):
                counts[i] += min(rcap[i], int(ideal[i]))
        # (3) >=1 floor for every kept-capable node (only safe because len(capable) <= L here).
        for i in capable:
            if counts[i] < 1:
                counts[i] = 1
        # (4) shave over-allocation (the >=1 floor or a GPU pre-fill bigger than L could exceed it)
        # by smallest remaining-share remainder, then SMALLEST capacity first — so tiny nodes empty
        # toward 1 before big GPU/RAM nodes do (keeps the strongest boxes full). Never below 1.
        over = sum(counts) - L
        if over > 0:
            for i in sorted(capable, key=lambda j: (ideal[j] - int(ideal[j]), cap[j])):
                if over <= 0:
                    break
                take = min(over, counts[i] - 1)
                counts[i] -= take
                over -= take
        leftover = L - sum(counts)
        if leftover > 0:                     # (5) hand remaining layers out by largest remainder
            for i in sorted(capable, key=lambda j: ideal[j] - int(ideal[j]), reverse=True):
                if leftover <= 0:
                    break
                add = min(leftover, cap[i] - counts[i])
                counts[i] += add
                leftover -= add
    else:
        cap_sum = sum(max_layers)
        counts = [min(max_layers[i], (L * max_layers[i]) // cap_sum) for i in range(len(nodes))]
        leftover = L - sum(counts)
    while leftover > 0:   # round-robin any remainder (proportional rounding / exact-fill slack)
        progressed = False
        for i in range(len(nodes)):
            if leftover == 0:
                break
            if counts[i] < max_layers[i]:
                counts[i] += 1
                leftover -= 1
                progressed = True
        if not progressed:
            break

    stages: list[StageAssign] = []
    cursor = 0
    for i, n in enumerate(nodes):
        c = counts[i]
        if c <= 0:
            continue                       # drop zero-layer nodes: a stage with no layers is just
                                           # a wasted pipeline hop (e.g. a GPU too small to hold even
                                           # one layer in vram-first mode got 0 in the split)
        stages.append(StageAssign(
            node_id=n.node_id, hostname=n.hostname,
            layer_start=cursor, layer_end=cursor + c,
            has_embed=False, has_head=False,
            est_bytes=c * per_layer + fixed[i], usable_bytes=n.usable_bytes))
        cursor += c
    if stages:                             # embed lives on the first REAL stage, head on the last
        stages[0].has_embed = True
        stages[-1].has_head = True

    base.ok = True
    base.stages = stages
    return base
