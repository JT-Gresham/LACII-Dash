# GGUF ingestion

InfiniteModel can run models that are published **only** as a llama.cpp **`.gguf`** file — the
large pool of community quants on Hugging Face that never shipped a safetensors checkpoint. It does
this by **normalizing the GGUF to a standard HF safetensors checkpoint once**, at add/download time.
After that one-time conversion the model is *ordinary* in the system: chunk-streamed to workers,
int4/int8 shard-cached, and run on the distributed pipeline with **no GGUF awareness anywhere
downstream** — exactly the same path a native safetensors model takes. (Same idea as the fp8 / nvfp4
source checkpoints: dequantize to bf16 once, then re-quantize to our own int4 for serving.)

Why not run GGUF directly? GGUF packs weights in GGML's k-quant / i-quant block layouts that the
GGML tensor library executes — not PyTorch. Porting those kernels would be large and pointless: the
engine already has a fast int4 decode path (`torch tinygemm` on NVIDIA/CPU, a Triton w4a16 kernel on
ROCm — see [ACCELERATION.md](ACCELERATION.md)). So the GGUF quantization is discarded; the value
GGUF ingestion unlocks is **access to the weights**, which are then served at the engine's own
quantization.

---

## Adding a GGUF model

You give the engine the **HF repo id** and the **`.gguf` filename** within it.

**Dashboard:** **+ Add model** → put the repo id in the model field, and the `.gguf` filename in the
**"GGUF file (optional)"** box → Add + download.

**API:**

```bash
# repo id + the quant you want (a single file, or any part of a split set)
curl -X POST "http://<controller>:21434/add_model?model=<hf-repo>&gguf_file=<file>.gguf"

# optionally give it a friendly name
curl -X POST "http://<controller>:21434/add_model?model=<hf-repo>&gguf_file=<file>.gguf&name=my-model"
```

The `gguf_file` must be a `.gguf` filename that exists in the repo (validated — a value not ending
in `.gguf` is rejected). It may be a **single-file quant** *or* **any part of a split
`NNNNN-of-NNNNN` set**: a part name is normalized to **part 1** and the whole 1..N series is checked
against the repo listing before anything is registered, so an incomplete set is a 400 at add time
rather than a 404 deep inside a multi-hour pull. (A single-file name is honoured verbatim with no
listing probe — power-user override. If the listing call itself fails for a split name, the add
still proceeds and says so: *"completeness NOT verified"*.)

Leaving the box **empty** on a GGUF-only repo lets the resolver pick for you: it prefers a
safetensors twin (`<repo>` minus its `-GGUF` tag) if one exists, else auto-selects a medium K-quant
— which may be a **complete split set**, returned as its part 1. A single file wins ties against a
split set of the same quant rank.

The repo is recorded as **GGUF-sourced** in `custom_gguf.json`, always as **one** filename (part 1);
the rest of the set is re-derived from that name, so nothing downstream or on disk changes shape.
From then on it's an ordinary registered model. Download (and thus conversion) starts like any other
model; the dashboard shows conversion progress on the model's row, with the part set counted in the
progress denominator.

After it's converted, use it exactly like any model — `/load`, the chat/completions APIs, shard
compile, etc. Nothing else in the workflow is GGUF-specific.

---

## What the conversion does (under the hood)

The heavy step runs in a **subprocess** (`gguf_convert.py`, driven by
`model_store.convert_gguf_to_model_dir`) so a large `from_pretrained` — which fully materializes the
model in RAM — can OOM the *subprocess* without taking down the controller box it co-hosts. The
subprocess:

1. **Derives the full part set from the filename.** `gguf-split` names its output
   `<base>-00001-of-0000N.gguf` (`llama_split_path`: `"%s-%05d-of-%05d.gguf"`), so both the index
   *and* the total live in the name — the whole series is derivable from any one part with no
   listing call. A single-file name yields a one-element set and takes exactly the path it always
   did. A name that is some *other* part (`-00003-of-00005`) is normalized to part 1, because only
   part 1 carries the full KV metadata (architecture, hyper-parameters, tokenizer); parts 2..N are a
   stub header plus their share of the tensors. ⚠ The parts are **not** a byte-split of one file —
   `cat`-ing them produces something that *opens* (part 1's header sits at offset 0) and is
   silently truncated garbage. The set is handed to the reader as part 1 with its siblings
   physically beside it in the same directory.
2. **Auto-installs its optional deps on demand** (the controller env has torch+transformers but may
   not have these extras, and the box may be SSH-less): `gguf` (parse the file), `accelerate`
   (the low-memory load path), and `sentencepiece` / `tiktoken` / `protobuf` (to build a tokenizer).
3. **Downloads every part** from the repo (one `hf_hub_download` each; HF token read from the
   `HF_TOKEN` env var, never a CLI arg — process listings leak args). A 404 on any part of a
   multi-part set **aborts** rather than dequantizing a model with whole layers missing. All parts
   must also land in one HF cache snapshot directory — two directories means the repo was
   re-committed mid-pull, so the set on disk mixes revisions, and that aborts too.
4. **Inventories the parts** (split sets only), by mmap-reading each part's GGUF *index* — the
   tensor data is never faulted in. This catches two things a filename cannot: a part that isn't a
   readable GGUF at all (a truncated pull, or an HTML error page saved under the right name), and a
   tensor **name present in two parts** — parts of one set partition the tensors, so an overlap
   means these files aren't one set. It also prints the tensor/parameter count and the bf16 RAM the
   load is about to need.
5. **Dequantizes to bf16** via the transformers GGUF loader and **saves a safetensors checkpoint**
   into `models/<name>/`. The load requests `output_loading_info`, and for a split set the
   non-benign `missing_keys` are a **gate** — this is the whole safety story. A transformers without
   sharded-GGUF support does not raise: it reads part 1, builds the full architecture, and
   **randomly initializes every later layer**, yielding a model that saves cleanly, is the right
   size, passes any structural check, and generates garbage. Non-empty ⇒ refuse, naming the unread
   tensors. A transformers too old to accept `output_loading_info` at all is also a refusal for a
   split set, since coverage then can't be proven either way. (`inv_freq` / rotary buffers and a
   tied `lm_head.weight` are excluded — legitimately absent from any checkpoint.) On the
   **single-file** path the same check only **warns**: that path ships and works today, and the
   partially-read-set failure mode doesn't exist for one file.
6. **Produces a fast tokenizer**, verified by reloading it (this is the fiddly part — see below).
7. Prints `GGUF_CONVERT_OK <dir>` on success; any non-zero exit fails the add with the captured
   error.

### The tokenizer step

The controller is a long-running process that caches "is sentencepiece/tiktoken available?" **at
startup** — so even though the subprocess just pip-installed them, the controller can't convert a
*slow* tokenizer to *fast* at serve time. The conversion therefore insists on saving a **fast
`tokenizer.json`** (which loads purely via the `tokenizers` Rust lib that transformers always has),
trying two sources in order, each verified by reloading from the saved dir:

1. the **GGUF-embedded** tokenizer (works when the slow→fast deps convert cleanly), then
2. the **base repo's native** tokenizer — most GGUF repos are named `<base>-GGUF`, and the base repo
   usually ships a ready `tokenizer.json` that loads with no extra deps.

If neither yields a reload-verified fast tokenizer, the conversion **aborts** (the model would save
but be unusable at serve time) rather than leaving a broken model registered.

---

## Coverage & limitations

- **Architectures:** whatever the transformers GGUF loader supports — Llama, Qwen2, Mistral, Gemma,
  and the other mainstream families. An unsupported arch fails the conversion with the loader's error.
- **Split `NNNNN-of-NNNNN` sets are accepted**, but only as a **complete contiguous 1..N series** —
  checked against the repo listing at add time, and again part-by-part during the pull. A gap fails
  loud at the earliest point that can see it, rather than converting a model with whole layers
  absent (which would save, be plausibly sized, and generate garbage). Name any part, or leave the
  GGUF field empty and let the resolver pick; part 1 is what gets registered and what the converter
  opens.
- ⚠ **A split conversion has never been run end-to-end.** Whether the transformers installed on the
  controller boxes can read a multi-part GGUF **at all** has not been determined. What *is* tested
  is offline and unit-level (`test_gguf_split.py` — no torch, no network, no fleet): part-name
  derivation, controller/converter agreement on those names, the incomplete-set refusals, and the
  `missing_keys` classifier on synthetic loading info. The dequantization step itself is untested on
  a real set. If the installed transformers can't read one, step 5's gate is designed to turn that
  into a **refusal naming the unread tensors** rather than a silently random-initialized
  checkpoint — but that is the design, not a measurement. Treat the first split conversion as an
  experiment, and check for the converter's `load verified: 0 unloaded tensors across N parts` line
  before trusting the output.
- **One quant per repo.** A repo is mapped to one chosen `.gguf`. Re-adding the same repo with a
  different `gguf_file` updates the choice (and re-converts on next download).
- **The GGUF quantization is not preserved.** The weights are dequantized to bf16 and then served at
  the engine's own int4/int8 (or bf16). So a `Q4_K_M` GGUF doesn't stay `Q4_K_M` — it becomes bf16
  on disk and is re-quantized to our int4 for serving. Pick a GGUF quant that's high enough quality
  to survive that round-trip (a very low-bit GGUF has already lost information the engine can't
  recover).
- **Conversion is a one-time heavy step.** It materializes the full model in RAM in the subprocess;
  size the controller box accordingly (it's the same memory profile as loading the bf16 model once).

---

## Lifecycle & troubleshooting

- The GGUF mapping persists in `custom_gguf.json` (kept in lockstep with the model registry). It's
  gitignored / per-controller, like other custom-model state.
- **`/delete`** (purge files) and **`/forget`** drop the repo's GGUF mark along with the model.
- **Conversion failed with a tokenizer error** → the GGUF-embedded tokenizer needed sentencepiece/
  tiktoken and the base repo had no fast tokenizer to fall back to; try a repo whose base model
  ships a `tokenizer.json`, or a different GGUF of the same model.
- **"split GGUF set is INCOMPLETE"** (at add time, from the repo listing) or **"split part … is not
  in `<repo>`"** (during the pull) → the repo doesn't actually hold every part of that series.
  Nothing was converted. Name a different quant, or a single-file one.
- **"N tensors were NOT read from the N-part GGUF set"** → the installed transformers read part 1
  and would have randomly initialized the rest; the conversion was refused and nothing saved.
  Upgrade transformers on the controller, or name a single-file quant. Same two options for **"this
  transformers does not accept output_loading_info"** (too old to prove a split read covered every
  part).
- **"tensor … appears in BOTH …"** → two different quants' parts collided under names that parse as
  one series; they are not one set. Pick an explicit quant.
- **"split parts landed in different cache snapshots"** → the repo was re-committed mid-download, so
  the parts on disk mix revisions. Retry the download.
- **OOM during conversion** → the box lacks RAM to materialize the full model; convert on a
  higher-RAM controller (the conversion is subprocess-isolated, so the controller itself survives
  the OOM and reports the failure).

See also: [OPERATIONS.md](OPERATIONS.md) (model lifecycle, shard-cache compile), and
[ACCELERATION.md](ACCELERATION.md) (the int4 path the converted model runs on).
