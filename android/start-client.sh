#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel Android worker — launcher.
#  Run INSIDE the proot guest, from this android/ folder (after setup.sh):
#      bash start-client.sh --name tablet
#
#  Defaults: controller "auto" = find it by UDP-broadcast discovery (retry forever),
#  --device cpu (no CUDA on Android), --ram "android-tablet" (dmidecode/root aren't
#  available in proot — harmless). Pass an explicit --controller to pin a static IP.
#  Extra flags pass straight through, e.g.:
#      bash start-client.sh --name tablet --controller 192.168.1.50
#      bash start-client.sh --name tablet --os-reserve-gb 3
#
#  Tip: run under tmux so it survives the terminal closing:
#      tmux new -s im   ->   bash start-client.sh --name tablet   ->   Ctrl-b d
#  And in TERMUX (host) keep the CPU awake:   termux-wake-lock
# ===========================================================================
HERE="$(cd "$(dirname "$0")" && pwd)"
cd "$HERE"
if [ ! -x .venv/bin/python ]; then
  echo "[!] worker env not built yet — run:  bash setup.sh"
  exit 1
fi
PY="$HERE/.venv/bin/python"
# Clean interactive stop: Ctrl-C / SIGTERM breaks the self-heal loop below.
trap 'echo "[stop] signalled - exiting."; exit 0' INT TERM
code=0
while true; do
  set +e
  "$PY" client.py --controller auto --control-port 50100 \
       --device cpu --ram "android-tablet" "$@"
  code=$?
  set -e
  if [ "$code" = "42" ]; then
    echo "[update] new code pulled - relaunching ..."
    continue
  fi
  # ANY other exit (crash / dropped control link / flaky-Wi-Fi blip) self-heals by
  # restarting. Critical on the tablet: if this script exits, the tmux 'wrk' pane drops
  # to a shell and the bandwidth panel that tails that pane FREEZES. Back off so a
  # hard-failing worker can't hot-loop.
  echo "[restart] worker exited (code $code) - restarting in 3s ..."
  sleep 3
done
