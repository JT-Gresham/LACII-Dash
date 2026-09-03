#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel CONTROLLER (Linux)  -  starts a controller
# ===========================================================================
set -e
cd "$(dirname "$0")"
PY=./InfMdlCtl_env/bin/python
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
exec "$PY" server.py "$@"   # controller host/port default from config.json (override: --controller HOST)
