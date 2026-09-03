#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel CONTROLLER (Linux)  -  starts a controller
# ===========================================================================
set -e
cd "$(dirname "$0")"
PY=/root/InfMdlCtl/InfMdlCtl_env/bin/python
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"
export PATH="$(pwd)/InfMdlCtl_env/bin:$PATH"
exec "$PY" server.py "$@" &
