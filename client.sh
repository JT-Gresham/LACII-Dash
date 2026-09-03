#!/usr/bin/env bash
# ===========================================================================
#  InfiniteModel WORKER (Linux)  -  connects to the BEAST controller
#
#  Usage:
#    ./client.sh                    CPU worker (uses ./InfMdlWrk_env/bin/python if present)
#    ./client.sh --device cpu+gpu   use a local GPU, spill overflow to CPU
#    ./client.sh --name work        override the reported hostname
#
#  Cleanup is OFF by default; pass --clean only to purge cached models/chunks.
#  Extra client.py flags pass through ("$@"): --data-port, --ram, --name ...
#  To run detached on a worker:
#    setsid ./client.sh </dev/null >client.log 2>&1 &
# ===========================================================================

# Check if installed to a master or worker
if [ -d /root/InfMdlCtl ]
    then
        PY=/root/InfMdlCtl/InfMdlCtl_env/bin/python
    else
        PY=/root/InfMdlWrk/InfMdlWrk_env/bin/python
fi

set -e
cd "$(dirname "$0")"
[ -x "$PY" ] || PY="$(command -v python3 || command -v python)"

if [ -d /root/InfMdlCtl ]
    then
        export PATH="$(pwd)/InfMdlCtl_env/bin:$PATH"
    else
        export PATH="$(pwd)/InfMdlWrk_env/bin:$PATH"
fi
exec "$PY" client.py "$@" &


# controller host/port default from config.json (override: --controller HOST)
