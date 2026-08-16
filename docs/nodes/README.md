# Per-device node setup

Practical, per-device setup guides for joining a machine to an InfiniteModel fleet as a
**worker**. (The controller is a separate role — any reachable host running `server.py`;
it need not be one of these worker boxes.) Each guide covers install → run
the worker → persistence → optimal quant/ctx/placement for that VRAM tier → verify →
gotchas, grounded in the cross-platform references:

- [../ACCELERATION.md](../ACCELERATION.md) — int4 decode kernel matrix, the Windows/Linux
  NVIDIA + ROCm setup, the `INFINITEMODEL_CUDA_FUSED_MOE` MoE opt-in, prefill chunking.
- [../ROCM.md](../ROCM.md) — the full gfx1151 Strix Halo (AMD ROCm) recipe.

## Pick your node

| Doc | Host(s) | GPU / SoC | Arch | VRAM | OS | Role |
|---|---|---|---|---|---|---|
| [4070-ti-super.md](4070-ti-super.md) | `beast` | RTX 4070 Ti SUPER | Ada `sm_89` | 16 GB | Linux (Proxmox VE — see [../PROXMOX9_NVIDIA.md](../PROXMOX9_NVIDIA.md)) | GPU worker |
| [3060.md](3060.md) | `amdcomp`, `mobile` | RTX 3060 / 3060 Laptop | Ampere `sm_86` | 12 GB / 6 GB | Linux (both) — see note | GPU worker |
| [p620.md](p620.md) | `work` | Quadro P620 | Pascal `sm_61` | ~4 GB | Linux | small GPU helper |
| [strix-halo.md](strix-halo.md) | `om3nbox` | Ryzen AI Max+ 395 (gfx1151) | RDNA3.5 (ROCm) | ~60 GB unified | Ubuntu | standalone controller+worker |
| [steam-deck.md](steam-deck.md) | `steamdeck` | Van Gogh (gfx1033) | RDNA2 | small UMA | SteamOS | CPU worker (ROCm experimental) |
| [cpu-worker.md](cpu-worker.md) | `nuc01`–`nuc04`, `mini05`, `dell`, `prodesk`, `zippy` (+ `tablet`, opt-in) | — (CPU/RAM) | — | — | Linux | CPU worker |

### Inventory notes (the two rows above that go stale most often)

- **`mobile` is Linux now.** The laptop was reinstalled from Windows 11 to Debian 13 on
  2026-08-10 — same box, same RTX 3060 Laptop 6 GB, same LAN address. This column read
  *"Linux / Windows"* until 2026-08-16, which is how a reader ended up at `C:\Python314` and
  `client.bat` months after those paths stopped existing. See [3060.md](3060.md) §0 for what
  survived the reinstall and what is unknown. One thing the driver decides for you: `nvidia-smi`
  on that box reports **550.163.01 / CUDA 12.4** (probed 2026-08-16), and per
  [../PROXMOX9_NVIDIA.md](../PROXMOX9_NVIDIA.md) a 550 driver takes a **`cu124`** torch wheel —
  *not* the `cu128` the rest of the CUDA fleet installs.
- **`mini05` was retired as the controller's NFS *models source* on 2026-08-14, not as a
  worker.** It is still a registered CPU worker and belongs in the cpu-worker row; what went
  away is the models-mount failover (`im-models-source mini05|auto` now hard-refuse — beast is
  the only source; see the 2026-08-14 CHANGELOG entry). Two different roles for one hostname is
  exactly why this row read as stale — do not delete the node, and do not re-point the models
  mount at it.
- **`dell` was missing** — it joined as a CPU worker on 2026-07-11 (31 GB DDR3, no-AVX Xeon;
  a capacity node, deliberately slow). **`tablet`** is the opt-in Android worker
  ([../../android/README.md](../../android/README.md)) and is only in the fleet while someone
  starts it by hand, so it is marked as such rather than listed as fleet furniture.

## Fleet shape, in one line

A **controller** (any host running `server.py`, `:21434`) splits one model's transformer
layers across worker nodes over plain TCP and serves the Ollama + OpenAI + Anthropic APIs. GPU workers
contribute VRAM (run layers fast); CPU/RAM workers add **capacity** (fit bigger models when
distributed), not single-stream speed. `om3nbox` is a separate, self-contained
controller+worker on its own ~60 GB APU. Decode speed tracks a model's **active** params
(favor low-active-ratio MoE); total params track what **fits**.

> Compute-capability rule of thumb: the fused Triton MoE kernel needs **NVIDIA Ampere
> (`sm_80`) or newer** — it's an opt-in win on the 3060/4070, auto-disabled (safe fallback)
> on the Pascal P620, and the gfx1151 APU uses its own ROCm Triton path. See the per-device
> docs + ../ACCELERATION.md.
