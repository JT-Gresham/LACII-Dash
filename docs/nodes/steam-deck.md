# Node: steamdeck (Steam Deck — CPU worker)

A Valve Steam Deck running as an InfiniteModel **CPU worker**. By default the GPU is **not**
used — the Deck contributes its CPU + RAM to the fleet's capacity pool. ROCm on the Deck's
iGPU is an unverified research item; see
[ROCm on the Deck](#rocm-on-the-deck--research-only-never-run-on-the-box).

See also: [../ACCELERATION.md](../ACCELERATION.md) (int4-decode kernel matrix, CPU path),
[../ROCM.md](../ROCM.md) (the validated RDNA ROCm recipe — for gfx1151, **not** the Deck).

## 1. Overview

| | |
|---|---|
| **Host** | `steamdeck` |
| **APU** | gfx1033 "Van Gogh" — RDNA2, 8 CU iGPU |
| **VRAM** | small UMA carve-out (shared with system RAM; APU unified memory) |
| **OS** | SteamOS (Arch-based, **immutable root filesystem**) |
| **Role** | **CPU worker** — GPU not used by default; capacity (not single-stream speed) |
| **Device flag** | `--device cpu` |

Per [../ACCELERATION.md](../ACCELERATION.md), CPU/RAM nodes exist for **capacity** — fitting
models too big for the GPU pool — **not** single-stream speed. CPU decode is bound by DDR
bandwidth (~50–90 GB/s) and is fundamentally ~5–10× slower than a GPU regardless of kernel.
None of the Triton fast-path kernels (fused MoE, split-K dense) run on a CPU worker.

## 2. Install

SteamOS's root fs is **read-only/immutable**, so do **not** try to `pacman -S` system
packages or write outside `$HOME`. Everything below lives in your home directory.

Python 3 is present on SteamOS. Create a venv under `$HOME` and install the **CPU** torch
build (no CUDA/ROCm wheel — plain CPU torch):

```bash
# venv in $HOME (writable on SteamOS)
python3 -m venv ~/imenv

# CPU-only PyTorch — pinned to the fleet's CPU worker version
~/imenv/bin/python -m pip install --upgrade pip
~/imenv/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# App deps — same pinned versions as the rest of the fleet
~/imenv/bin/python -m pip install \
    transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 \
    numpy==2.4.6 psutil==7.2.2 einops fastapi uvicorn
```

> The `torch ... /whl/cpu` index is the standard CPU wheel and `2.13.0` is the exact pin the
> fleet's own provisioner installs on a CPU worker
> ([`provision_worker.sh`](../../provision_worker.sh),
> [`install/requirements-client.txt`](../../install/requirements-client.txt)); the app-dep pins
> come from the same two files. The arch-specific ROCm indexes in [../ROCM.md](../ROCM.md)
> (e.g. `rocm.nightlies.amd.com/v2/gfx1151/`) are for RDNA3.5 GPU workers and do **not** apply
> to the Deck's CPU role.
>
> **Python version — unverified, needs the box.** Nothing in this repo pins a Python version:
> the only requirement is a `python3` that the pinned wheels publish for, and SteamOS's stock
> interpreter has been sufficient — the Deck has run as a fleet worker on it since 2026-07-07.
> Run `python3 --version` on the Deck (`ssh deck@<deck-ip>`) if you need the number,
> e.g. to hunt for a missing wheel. If the pinned torch has no wheel for that interpreter, that
> is the one case for relaxing the pin on this box rather than chasing a build.

Clone the repo into `$HOME` as well:

```bash
git clone https://github.com/SixOfFive/infiniteModel ~/infinitemodel
```

## 3. Run the worker

Launch `client.py` as a **CPU** worker pointed at the fleet controller (`:21434` —
control port `50100`). The Deck does **not** run a controller; it only joins an existing
fleet. The `--controller` flag overrides `config.json`'s `controller_host` (don't edit
`config.json` — it's in `EXTRA_UPDATE_FILES` and a self-update would revert it; per
[../ROCM.md](../ROCM.md)):

```bash
cd ~/infinitemodel
setsid ~/imenv/bin/python client.py --controller <controller-ip> --device cpu \
    >~/worker.log 2>&1 </dev/null &
```

`<controller-ip>` is the fleet controller's LAN address. `--device cpu` keeps everything
off the iGPU.

> **Neither of those two flags is strictly required, and the port needs no checking.** 50100 is
> `config.json`'s shipped `control_port`, and `client.py`'s `--control-port` default *is* that
> value (`default=load_config()["control_port"]`) — so the explicit
> `--control-port 50100` you see on the NVIDIA boxes in
> [../ACCELERATION.md](../ACCELERATION.md) is the default written out, not a Deck-specific
> requirement. Pass it only on a fleet that moved the port. Likewise `controller_host` ships as
> `"auto"`, so a worker with no `--controller` finds the controller by UDP broadcast on
> `discovery_port` 50099; name an IP when broadcast can't cross to it (subnet/VLAN/VPN), which
> is the case this doc assumes.

### Persistence

**As deployed (verified on the box 2026-07-07):** the worker runs as a **system** unit,
`/etc/systemd/system/infinitemodel-worker.service` (enabled, `WantedBy=multi-user.target`),
with the venv at `~/infinitemodel/.venv` — **not** `~/imenv`, and **not** a
`systemctl --user` unit:

```ini
[Unit]
Description=InfiniteModel worker -> controller <old-controller-ip>
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=deck
Environment=HOME=/home/deck
WorkingDirectory=/home/deck/infinitemodel
ExecStart=/home/deck/infinitemodel/.venv/bin/python /home/deck/infinitemodel/client.py --controller <controller-ip> --control-port 50100 --name steamdeck --ram 4x-LPDDR5-5500 --no-clean
Restart=always
RestartSec=5
OOMScoreAdjust=800
Nice=5

[Install]
WantedBy=multi-user.target
```

(The unit Description may still name a previous controller IP; the load-bearing
`--controller <controller-ip>` in ExecStart is what counts.) Restart with
`sudo systemctl restart infinitemodel-worker.service`.

Two things about that `ExecStart` that are easy to misread:

- **It does not pass `--device cpu`, and the Deck runs on CPU anyway.** `--device` defaults to
  `cpu+gpu`, which fills a VRAM budget on the GPU and spills the rest — but it *"falls back to
  CPU if there's no CUDA device"* (`client.py` argparse), and the Deck has no CUDA device. The
  controller confirms the outcome rather than the intent: `GET /status` reports this node's
  device as `x86_64 (no CUDA → CPU)`. Section 3 tells you to pass `--device cpu` explicitly
  because relying on that fallback means the box's placement depends on a negative — the day
  the Deck's iGPU *does* get a working runtime (see ROCm below), the same unit would silently
  start offering it.
- **`--no-clean` is a no-op.** Cache cleanup has been opt-*in* (`--clean`) for a while;
  `--no-clean` is retained only so old command lines keep parsing (`client.py` describes it as
  a "deprecated no-op"). Leaving it costs nothing; adding it to a new unit buys nothing. What
  you must **not** do is "fix" it to `--clean` — that would wipe this box's model cache on every
  restart.

`--ram 4x-LPDDR5-5500` is there because `dmidecode` needs a password the service user doesn't
have; it is a reporting override only, and it is what the dashboard shows for this node.

> A system unit under `/etc` survives SteamOS updates in practice (the `/etc` overlay is
> writable and persistent), but the original `systemctl --user` + `loginctl enable-linger`
> approach is the no-`sudo` alternative if this box is ever rebuilt.

## 4. Optimal settings

This is a **CPU/capacity** node — tune for fit, not speed.

- **Quant:** **int4** (smallest footprint; ¼ the bytes read per token). On CPU, dense int4
  uses torch's CPU tinygemm (`_weight_int4pack_mm_for_cpu`) when present, else a
  dequant→fp32 GEMM; MoE experts always bf16-rematerialize per expert
  ([../ACCELERATION.md](../ACCELERATION.md), CPU section).
- **Placement:** let the controller place stages here only when the GPU pool is full — the
  Deck is a **spillover/capacity** stage, not a primary compute stage. Don't pin a hot model
  to it.
- **Context:** keep modest — KV cache lives in the (small, shared) Deck RAM. Long contexts
  on a memory-tight box are a real risk; if a long prefill OOMs, lower
  `INFINITEMODEL_PREFILL_CHUNK` (default 2048 → e.g. 512) per
  [../ACCELERATION.md](../ACCELERATION.md). The chunked-prefill default is already ON.
- **Models that fit/run well:** small sparse **MoE with low active-param ratio** (~3B active)
  decode fastest per byte read — prefer those over big dense models
  ([../ACCELERATION.md](../ACCELERATION.md), "Choosing a fast model"). The Deck is best as a
  small slice of a larger pipeline.
- **Avoid:** running it as the only/primary node for a model; big dense models (every param
  read per token → very slow on DDR bandwidth); expecting GPU-class tok/s — none of the
  Triton kernels help a CPU worker.

## 5. Verify

1. **CPU torch sanity** (no GPU expected):
   ```bash
   ~/imenv/bin/python -c "import torch; print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available()); \
   a=torch.randn(2048,2048); (a@a).sum().item(); print('matmul OK')"
   ```
   Expect `cuda avail False` and `matmul OK` (CPU worker).
2. **Registration:** confirm the node appears in the fleet — controller dashboard at
   `http://<controller-ip>:21434`, or check the worker log on the controller:
   `GET /logs?node=steamdeck` (per [../ACCELERATION.md](../ACCELERATION.md) verify step).
3. **Gen test:** with the Deck placed in a pipeline stage, run a short generation against the
   controller API and confirm coherent output (e.g. the fleet's standard "capital of France
   is Paris" smoke test). Watch `~/worker.log` for errors.

## 6. Gotchas

- **Immutable root fs.** SteamOS's root is read-only — keep the venv, repo, and unit file in
  `$HOME`. Do not rely on system `pacman` packages; a SteamOS update can wipe anything
  written outside `$HOME` (and `pacman` modifications to the base image).
- **CPU kernels don't accelerate.** Per [../ACCELERATION.md](../ACCELERATION.md): the Triton
  fused-MoE and split-K dense kernels are **GPU-only**; `INFINITEMODEL_CUDA_FUSED_MOE` has no
  effect on a CPU worker. Don't set GPU env switches here.
- **Long-prompt prefill memory.** The Deck has little RAM headroom; a long prefill can blow
  memory. `INFINITEMODEL_PREFILL_CHUNK` (default 2048) caps it — lower to 512 on this box if
  needed ([../ACCELERATION.md](../ACCELERATION.md)).
- **One worker per box.** Run exactly one `client.py` per hostname — two workers sharing a
  hostname fight over the controller registration ([../ACCELERATION.md](../ACCELERATION.md)).
- **Capacity, not speed.** Expect single-digit tok/s at best for its stage; that's expected
  for a CPU node and is by design.

### ROCm on the Deck — research only, never run on the box

Using the gfx1033 iGPU is **experimental and unverified**. Everything below is reasoning from
the validated gfx1151 recipe in [../ROCM.md](../ROCM.md) plus what SteamOS is; **no step has
been executed on a Deck**, so treat each bullet as a hypothesis with the check that would settle
it. Nobody has spent the afternoon this needs — the last bullet is why.

- **gfx1033 is probably not a TheRock-supported target — unverified, needs a browser (not the
  box).** The validated RDNA ROCm recipe in [../ROCM.md](../ROCM.md) targets **gfx1151** (Strix
  Halo, RDNA3.5) via AMD's arch-specific "TheRock" wheel index
  (`https://rocm.nightlies.amd.com/v2/gfx1151/`). RDNA2 **gfx1033** (Van Gogh) was not published
  there when this doc was written. **To settle it:** list the arch directories under
  `https://rocm.nightlies.amd.com/v2/` and look for a `gfx1033` (or `gfx103x`) index — that is a
  one-minute check and needs no Deck. If it isn't there, stop: forcing
  `HSA_OVERRIDE_GFX_VERSION` to borrow another arch's binaries is explicitly discouraged in
  [../ROCM.md](../ROCM.md) (it pushes foreign code objects onto the GPU and tends to fail with
  "no kernel image").
- **The immutable root will fight the runtime — unverified, needs the box.**
  [../ROCM.md](../ROCM.md) relies on pip-installed userspace (TheRock `rocm-sdk-*` packages, all
  inside the venv) plus an in-tree amdgpu kernel driver and `render,video` group membership
  (`sudo usermod -aG render,video "$USER"`). On SteamOS's immutable root, both the group change
  and any system driver bits want a writable overlay or a **distrobox** container. **To settle
  it, on the Deck:** `ls -l /dev/kfd /dev/dri/renderD*` (the amdgpu compute nodes must exist),
  `id` (is `deck` already in `render`/`video`?), and `rpm-ostree`-style: check whether a
  `usermod` survives a SteamOS update. None of this is set up there today.
- **Triton needs a C toolchain SteamOS doesn't ship — unverified, needs the box.** The RDNA int4
  Triton kernel JIT-compiles launcher stubs and requires `gcc` + Python headers
  ([../ROCM.md](../ROCM.md)). **To settle it, on the Deck:** `command -v gcc` and
  `python3 -c "import sysconfig; print(sysconfig.get_paths()['include'])"`, then check that
  header directory actually exists. If either is missing, the toolchain has to come from a
  distrobox container, not from `pacman` on the read-only root.
- **Even if all three came good, the payoff is small — and that is the load-bearing point.**
  8 CU of RDNA2 sharing a tiny UMA carve-out is far weaker than the gfx1151 numbers in
  [../ROCM.md](../ROCM.md)/[../ACCELERATION.md](../ACCELERATION.md), and this node's job in the
  fleet is capacity, not speed (Section 4). The realistic outcome of the whole exercise is a few
  tok/s on a handful of layers. That is why the checks above have never been run: the Deck stays
  a **CPU worker** (Section 3) deliberately, not for want of a verification pass.
