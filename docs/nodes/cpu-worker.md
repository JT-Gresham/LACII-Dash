# CPU / RAM worker node

Pure-CPU workers in the InfiniteModel fleet. These nodes carry no GPU; they exist to add
**capacity** — fitting models too big for the GPU pool by holding pipeline stages in system
RAM — **not** single-stream speed. CPU decode stays slow (see [Optimal settings](#4-optimal-settings)).

> Cross-links: int4-on-CPU decode details and the CPU fp32-GEMM notes live in
> [../ACCELERATION.md](../ACCELERATION.md). The GPU kernel recipes are in
> [../ROCM.md](../ROCM.md) (ROCm) and [../ACCELERATION.md](../ACCELERATION.md) (NVIDIA) — neither
> applies here; the Triton kernels are GPU-only.

---

## 1. Overview

| | |
|---|---|
| **Hosts** | PVE `nuc01`–`nuc04`, `mini05`, `dell`, `prodesk` (Proxmox hosts/guests), `zippy`; plus `tablet` when the opt-in Android worker is running |
| **GPU / arch** | none — pure CPU (`x86_64` on every box above; `tablet` is ARM) |
| **VRAM** | n/a; capacity is bounded by the box's system RAM (and its `MemoryMax` cap) |
| **OS** | Linux on every one of them — confirmed from the controller's own `GET /status`, which reports each worker's kernel: `7.0.14-*-pve` on the Proxmox boxes (`nuc01`–`nuc04`, `mini05`, `dell`, `prodesk`) and stock Debian `6.1.0-*-amd64` on `zippy`. `tablet` is Android running a **proot Debian guest** (glibc — see [../../android/README.md](../../android/README.md)). |
| **Role in fleet** | Capacity tier — hold pipeline stages in RAM so the controller can place a model that won't fit the GPU pool. Distributed, the fleet's CPU/RAM pool fits much larger models than any single GPU. |

CPU workers are placed **faster-RAM-first**: the controller prefers DDR5 over DDR4 over DDR3,
since CPU decode is DDR-bandwidth-bound (~50–90 GB/s, vs 150–670 GB/s on a GPU — see
[../ACCELERATION.md](../ACCELERATION.md)). The live fleet spans that whole spread — the `nuc`s
report `4x LPDDR5-4800`, `dell` reports `4x DDR3-1333` — so `dell` is deliberately the last
resort: it is 31 GB of **capacity**, on a pre-AVX Xeon, and it will be the slowest link in any
pipeline that crosses it.

> **`mini05` is still a CPU worker.** What was retired on 2026-08-14 is its *other*, unrelated
> role: it was the controller's failover NFS **models source**, and that export is being deleted.
> `im-models-source mini05|auto` now hard-refuse; beast is the only source (see the 2026-08-14
> CHANGELOG entry). The hostname appearing in a retirement note is not a reason to drop it from
> this list — one host, two roles, only one of them retired.

---

## 2. Install

A Python venv + the **CPU** torch build + the worker deps. The torch CPU wheel index is the
key difference from a GPU node — do **not** install a CUDA/ROCm wheel here.

```bash
# 1) venv (use the box's python3)
python3 -m venv ~/imenv

# 2) PyTorch — CPU build (no CUDA/ROCm runtime pulled in). Pinned: this is the exact
#    version the fleet provisioner installs, so a new node matches the running ones.
~/imenv/bin/python -m pip install torch==2.13.0 --index-url https://download.pytorch.org/whl/cpu

# 3) App deps (same pinned versions as the rest of the fleet)
~/imenv/bin/python -m pip install \
    transformers==5.12.1 safetensors==0.8.0 huggingface_hub==1.19.0 \
    numpy==2.4.6 psutil==7.2.2 einops fastapi uvicorn
```

> **These pins are not folklore — they are what the repo installs.** `torch==2.13.0` from the
> `/whl/cpu` index and the five app pins are exactly the lines in
> [`provision_worker.sh`](../../provision_worker.sh) (the idempotent CPU-worker provisioner) and
> [`install/requirements-client.txt`](../../install/requirements-client.txt) (which the offline
> installer resolves against `install/wheels/<os>/`). Prefer running `provision_worker.sh` over
> typing the above; the manual form is here for boxes it can't run on.
>
> `einops`, `fastapi` and `uvicorn` are **not** in `requirements-client.txt`: `einops` is needed
> only by models whose `trust_remote_code` imports it (e.g. `nomic-embed-text`), and
> `fastapi`/`uvicorn` are the *controller's* deps — harmless on a worker, but you can drop them
> if the box will never run `server.py`. No build toolchain (`gcc` / `python3-dev`) is needed on
> a CPU worker: that requirement is for the Triton JIT, which is GPU-only and never taken here.

---

## 3. Run the worker

Launch `client.py` pointed at the controller, with `--device cpu`:

```bash
~/imenv/bin/python client.py --device cpu --controller <controller-ip>
```

`--controller` overrides `config.json`'s `controller_host`; prefer the flag over editing
`config.json`. The control port is **50100** — that is `config.json`'s shipped `control_port`,
and `client.py`'s `--control-port` simply defaults to it, so you only pass the flag on a fleet
that moved the port.

You can also omit `--controller` entirely: the shipped `controller_host` is `"auto"`, which
makes the worker find its controller by UDP broadcast on `discovery_port` 50099 (a static host
that turns out to be unreachable falls back to the same discovery). The fleet provisioner takes
exactly that route — [`provision_worker.sh`](../../provision_worker.sh) installs an `ExecStart`
of `client.py --device cpu --attn sdpa` with **no** `--controller` at all. Name the IP only when
broadcast can't reach it (different subnet/VLAN/VPN).

### Persistence — systemd unit with a memory cap

CPU workers run under a **systemd unit** named **`infinitemodel-worker.service`** — that is the
name [`provision_worker.sh`](../../provision_worker.sh) writes to `/etc/systemd/system/`, so it
is the name `systemctl restart` and the fleet-wide `POST /restart?workers=1` path expect to find
on a provisioned box. The unit below is that shipped unit **plus the memory cap**, which the
provisioner does not set: `MemoryMax=<cap>` and `MemorySwapMax=0` turn an over-budget load into a
**clean OOM-kill** (the controller sees the worker drop and replans) instead of the host freezing
on swap thrash. Size `<cap>` to leave headroom for the OS.

```ini
# /etc/systemd/system/infinitemodel-worker.service
[Unit]
Description=InfiniteModel worker
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=<user>
# Absolute paths, NOT %h: in a SYSTEM unit %h resolves to /root regardless of User=
# (systemd.unit(5): "in case of the system manager this resolves to /root"), so a unit
# written with %h under User=node silently looks for /root/imenv and fails to start.
WorkingDirectory=/home/<user>/infinitemodel
ExecStart=/home/<user>/imenv/bin/python /home/<user>/infinitemodel/client.py --device cpu --attn sdpa
Restart=always
RestartSec=5
# Make this worker the preferred OOM victim so a memory crunch on a shared box never takes
# the production VM/CT with it, and keep it off the CPU that interactive work needs.
OOMScoreAdjust=800
Nice=5
# Memory safety — clean OOM-kill -> controller replan, never a host freeze:
MemoryMax=<cap>
MemorySwapMax=0

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl daemon-reload
sudo systemctl enable --now infinitemodel-worker.service
```

No `--controller` appears above on purpose (see "Run the worker": `controller_host: "auto"`
discovers it). Add `--controller <controller-ip>` to `ExecStart` when broadcast can't reach the
controller, and `--name <host>` when the box's own hostname isn't what you want on the dashboard.

**A box without root or systemd** — the `dell` node is one (unprivileged user, `sudo` not even
installed) — can be kept alive by a flock-guarded supervisor loop plus two crontab lines: an
`@reboot` start and a per-minute `pgrep -f client.py || restart` keepalive. Take the lock
**outside** the worker and close the fd when launching it (`9>&-`), or an orphaned worker
inherits the lock and blocks every future restart. For a `systemctl --user` unit instead, run
`loginctl enable-linger "$USER"` once so it starts at boot without an active login (per
[../ROCM.md](../ROCM.md)); note `MemoryMax=` in a user unit needs cgroup delegation, which is why
the system unit above is the default.

---

## 4. Optimal settings

- **Quant: int4.** Smallest footprint → most capacity per box, and at batch-1 the int4 unpack
  is paid regardless, so the fp32 weight is effectively free. Dense int4 on CPU uses torch's
  CPU tinygemm (`_weight_int4pack_mm_for_cpu`) when present, otherwise a dequant→fp32 GEMM
  (the faster CPU path at batch-1). MoE experts always bf16-rematerialize per expert. See
  [../ACCELERATION.md](../ACCELERATION.md).
- **Context:** keep modest — CPU RAM holds both weights and the KV cache, and longer context
  multiplies KV. There is no per-tier "best" ctx documented here; size it to the box's RAM and
  the model (verify against your `MemoryMax`).
- **Placement mode:** CPU workers participate as **pipeline stages** in a distributed model;
  they are the spill/capacity tier behind the GPU nodes. CPU tensor-parallel is **not** worth
  it here — per [../ACCELERATION.md](../ACCELERATION.md), CPU-TP never beats pipelining onto a
  GPU on this fleet.
- **Models that fit/run well:** the value case is a **big model that won't fit the GPU pool**,
  placed across CPU RAM. For tolerable tok/s prefer a **sparse MoE with a low active-param
  ratio** (≈3B active) over a big dense model of similar quality — decode reads only the active
  bytes per token (see "Choosing a fast model" in [../ACCELERATION.md](../ACCELERATION.md)).
- **What to avoid:**
  - Don't expect speed — a dense model decodes ~all params/token on slow DDR; a CPU worker
    adds capacity, not single-stream throughput (~5–10× slower than a GPU regardless of kernel).
  - The GPU kernel work (fused MoE, split-K dense, autotune) does **nothing** on CPU — none of
    those Triton kernels run off-`cuda` (the self-check returns False). Don't set
    `INFINITEMODEL_CUDA_FUSED_MOE` on a CPU box; it has no effect.

---

## 5. Verify

1. **torch is the CPU build, no GPU expected:**
   ```bash
   ~/imenv/bin/python -c "import torch; print('torch', torch.__version__, '| cuda avail', torch.cuda.is_available())"
   ```
   `cuda avail` should be `False` on a pure-CPU node, and a matmul must run:
   ```bash
   ~/imenv/bin/python -c "import torch; a=torch.randn(2048,2048); (a@a).sum().item(); print('cpu matmul OK')"
   ```
   (Adapted from the ROCm sanity check in [../ROCM.md](../ROCM.md), with the device left on CPU.)
2. **Worker registered with the controller:** check the controller dashboard / `GET /status`
   on `<controller-ip>:21434` and confirm the node's hostname appears with its CPU/RAM.
3. **Logs:** `GET /logs?node=<host>` on the controller streams this worker's log on its
   heartbeat (per [../ACCELERATION.md](../ACCELERATION.md)'s log reference).
4. **Gen test:** place/load a model that spans this node and run one short generation through
   the controller API; confirm coherent output (it will be slow — that's expected for CPU).

---

## 6. Gotchas

- **Set `MemoryMax` + `MemorySwapMax=0`.** Without the cap (or with swap enabled) an
  over-budget load thrashes swap and **freezes the host**; with the cap the kernel OOM-kills
  the worker cleanly and the controller **replans**. This is the whole reason CPU workers run
  under systemd here.
- **One worker per box.** Two workers sharing a hostname fight over the controller
  registration (noted for GPU boxes in [../ACCELERATION.md](../ACCELERATION.md); same applies).
- **CPU is capacity, not speed.** Don't troubleshoot "slow decode" on a CPU node — it's
  bandwidth-bound by design. If you need speed, place the model on a GPU.
- **The GPU acceleration env switches are inert here.** `INFINITEMODEL_CUDA_FUSED_MOE`,
  split-K, CUDA-graph, etc. are GPU-only ([../ACCELERATION.md](../ACCELERATION.md)). The one
  switch that *is* CPU-relevant is `INFINITEMODEL_PREFILL_CHUNK` (default 2048): it chunks long
  prefill so the SDPA **math** backend — which the CPU path uses — doesn't materialize the full
  `[1, H, q, total]` score tensor and OOM on long prompts. Standard-attention models only;
  lower it (512–1024) on a memory-tight box, `0` to disable.
- **Faster RAM first.** Placement prefers DDR5 > DDR4 > DDR3; a DDR3 box will be the slowest
  link in a pipeline that spans it.
