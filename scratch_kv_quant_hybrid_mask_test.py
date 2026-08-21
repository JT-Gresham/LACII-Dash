"""#172-hybrid — the WORKER's pack mask and the CONTROLLER's pack count must agree.

WHY THIS SHAPE. The worker decides which layers actually get a TurboQuant cache slot
(Shard._kv_quant_layer_mask); the controller decides how many bytes to reserve for them
(ModelSpec.turboquant_layers, via for_kv_quant). If those two ever disagree, the controller plans a
footprint the worker does not build — and in the dangerous direction that is an UNDER-reserve, i.e.
an OOM part-way through a decode rather than a clean refusal at load time.

This does not re-implement either rule. It imports the REAL worker method (shard_build imports
cleanly without torch) and calls it against a stub Shard, then compares with the REAL ModelSpec
property. A transcription of what I think each does would prove nothing — that is exactly how a
guard ships inert.

The third fixture is the one with teeth: a sliding-window hybrid. Sliding layers hold real K/V but
BOUNDED by their own transformers cache class, so they must be counted as KV-bearing yet NOT packed.
Getting that wrong is silent: the plan shrinks, the load succeeds, and the OOM arrives later.
"""
import sys
import types

from shard_build import ShardBuildMixin
from placement import ModelSpec

failures: list[str] = []


def _stub(layer_types, n_layers, layer_start=0, owned=None, linattn_flat=False):
    """A Shard-shaped object carrying only what _kv_quant_layer_mask actually reads."""
    s = types.SimpleNamespace()
    s.owned_layers = [object()] * (owned if owned is not None else n_layers)
    s.layer_start = layer_start
    s._linattn_flat = linattn_flat
    s.cfg = types.SimpleNamespace(layer_types=layer_types, num_hidden_layers=n_layers)
    s._hybrid = bool(layer_types and any(str(t) != "full_attention" for t in layer_types))
    return s


def _spec(layer_types, n_layers, kv_layer_frac=1.0):
    return ModelSpec(name="t", hidden_size=4096, num_layers=n_layers, num_heads=32, num_kv_heads=8,
                     head_dim=128, intermediate_size=11008, vocab_size=32000,
                     tie_embeddings=False,
                     layer_types=tuple(layer_types) if layer_types else None,
                     kv_layer_frac=kv_layer_frac)


QWEN35 = ["linear_attention"] * 3 + ["full_attention"]          # x16 -> 64 layers, 16 packed
CASES = [
    ("dense-32",           ["full_attention"] * 32,        32, 32),
    ("qwen3_5-64 (48L+16F)", QWEN35 * 16,                  64, 16),
    ("gpt-oss-24 (12S+12F)",
     ["sliding_attention" if i % 2 else "full_attention" for i in range(24)], 24, 12),
    ("gemma4-24 (5S:1F)",
     [("full_attention" if i % 6 == 5 else "sliding_attention") for i in range(24)], 24, 4),
]

for label, lts, n, want_packed in CASES:
    # --- worker: the real method, whole model owned by one shard
    mask = ShardBuildMixin._kv_quant_layer_mask(_stub(lts, n))
    got_worker = sum(1 for m in mask if m)
    if got_worker != want_packed:
        failures.append(f"{label}: WORKER mask packs {got_worker}, expected {want_packed}")

    # --- controller: the real property
    got_ctrl = _spec(lts, n).turboquant_layers
    if got_ctrl != want_packed:
        failures.append(f"{label}: CONTROLLER turboquant_layers={got_ctrl}, expected {want_packed}")
    if got_ctrl != got_worker:
        failures.append(f"{label}: DRIFT — worker packs {got_worker} layers, controller reserves "
                        f"for {got_ctrl}. Under-reserve = decode OOM.")

    # --- sliding layers must be KV-BEARING but UNPACKED (the trap)
    spec = _spec(lts, n)
    n_slide = sum(1 for t in lts if "sliding" in t)
    if n_slide:
        if spec.full_attention_layers != n:
            failures.append(f"{label}: sliding layers must COUNT as KV-bearing — "
                            f"full_attention_layers={spec.full_attention_layers}, expected {n}")
        if spec.turboquant_layers >= spec.full_attention_layers:
            failures.append(f"{label}: sliding layers were PACKED (packed={spec.turboquant_layers} "
                            f">= kv-bearing={spec.full_attention_layers}) — unbounds their window")

# --- sharded: a mid-stage must mask by GLOBAL index, not by its own 0-based offset
lts = QWEN35 * 16
for start, count in ((0, 16), (16, 16), (33, 15), (48, 16)):
    m = ShardBuildMixin._kv_quant_layer_mask(_stub(lts, 64, layer_start=start, owned=count))
    want = [str(lts[start + i]) == "full_attention" for i in range(count)]
    if m != want:
        failures.append(f"shard[{start}:{start+count}] mask misaligned — global indexing broken")

# --- fla/kda: worker packs nothing; controller must agree
m = ShardBuildMixin._kv_quant_layer_mask(_stub(None, 27, linattn_flat=True, owned=27))
if any(m):
    failures.append("fla/kda: worker must pack NOTHING (it builds _make_linattn_kv)")
if _spec(None, 27, kv_layer_frac=7 / 27).turboquant_layers != 0:
    failures.append("fla/kda: controller must reserve for 0 packed layers")

# --- unknown layer kind: both sides must be CONSERVATIVE (do not pack)
m = ShardBuildMixin._kv_quant_layer_mask(_stub(["full_attention", "some_new_thing"] * 8, 16))
if sum(1 for x in m if x) != 8:
    failures.append("unknown layer kind must NOT be packed (conservative = reserve bf16)")

if failures:
    print("FAIL — #172-hybrid pack-mask parity:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"PASS — worker mask == controller count across {len(CASES)} archs + 4 shard offsets; "
      "sliding counted as KV-bearing but never packed; fla/kda packs nothing; unknown kinds bf16")
