"""#mixed-set — a forced /update must never RESTART onto a primary whose VERSION did not move.

WHY. The raw CDN propagates per file, so a fetch minutes after a push can return a NEW extra beside
a STALE primary — a set that is not any commit that ever existed. Every guard already in
_self_update_check passed that case (the primary was not OLDER, it was EQUAL; the bytes parsed; the
staging read back clean) and `force` then restarted into it. Live hit 2026-08-18: om3nbox took the
0.3.29 routes_dashboard.py and engine_load.py — both calling ModelSpec.for_kv_quant — beside the
0.3.28 placement.py that does not define it, and came up 500-ing AttributeError on every /plan.

This drives the REAL _self_update_check against a throwaway copy of the tree (so `here` /
os.path.dirname(__file__) can never be the live repo), with _fetch_repo_file monkeypatched to serve
whatever the scenario needs, and reads the restart decision off the process exit code (42 = restart).

The second scenario is the one that keeps the fix honest: staging must still HAPPEN on an unbumped
primary, because writing the newer extras is how a half-updated box CONVERGES when the lagging
files arrive. A "fix" that refuses to write them would strand the box in exactly the inconsistent
state being escaped, needing a fresh VERSION bump to get out.
"""
import os
import shutil
import subprocess
import sys
import tempfile

REPO = os.path.dirname(os.path.abspath(__file__))

DRIVER = r'''
import os, sys, types
sys.path.insert(0, %(tmp)r)
os.environ.setdefault("INFINITEMODEL_NO_AUTOSTART", "1")
import server

PRIMARY = %(primary)r
EXTRA   = %(extra)r
server.VERSION = %(running)r

FAKE_SHA = "b" * 40
server._resolve_repo_sha = lambda: FAKE_SHA
REFS = []

def fake_fetch(fn, ref=""):
    # #sha-pin: record the ref every fetch used — the whole point is that ONE cycle pins ONE commit.
    # Written EAGERLY on each call, not in a finally: the restart path ends in os._exit(42), which
    # runs no finally blocks and no atexit hooks, so a deferred write is simply lost in exactly the
    # scenario that matters most (the successful deploy).
    REFS.append((fn, ref))
    bad = sum(1 for _, r in REFS if r != FAKE_SHA)
    with open(os.path.join(%(tmp)r, "_refs.txt"), "w") as _fh:
        _fh.write(f"{len(REFS)} {bad}\n")
    # Every OTHER file in EXTRA_UPDATE_FILES must fetch successfully and unchanged, or the cycle
    # aborts on a fetch failure (4 tries x backoff) instead of exercising the decision under test.
    if fn == "server.py":
        return PRIMARY
    if fn == "placement.py":
        return EXTRA
    try:
        with open(os.path.join(%(tmp)r, fn), "rb") as fh:
            return fh.read()
    except Exception:
        return b"# absent locally\n"
server._fetch_repo_file = fake_fetch
server._self_update_check("server.py", lambda: True, force=%(force)r)
# routes_lifecycle's forced path skips its unconditional os._exit(42) ONLY when it finds an
# "update REFUSED: " entry in ACTIVITY. Record whether one is there: a decision this function
# makes but does not ANNOUNCE is a decision the caller overrides.
_ref = any(str((e or {}).get("msg") or "").startswith("update REFUSED: ")
           for e in list(getattr(server, "ACTIVITY", [])))
with open(os.path.join(%(tmp)r, "_refused.txt"), "w") as _fh:
    _fh.write("1" if _ref else "0")
print("NO_RESTART")
sys.exit(0)
'''

def run(name, running, primary_ver, extra_body, force, expect_restart, expect_staged,
        expect_route_told=None):
    tmp = tempfile.mkdtemp(prefix="imupd_")
    try:
        for fn in os.listdir(REPO):
            if fn.endswith((".py", ".txt", ".json")):
                shutil.copy2(os.path.join(REPO, fn), os.path.join(tmp, fn))
        # A minimal but REAL primary: it must parse, carry a VERSION, and declare the fetch list.
        primary = (f'VERSION = "{primary_ver}"\n'
                   'EXTRA_UPDATE_FILES: list[str] = ["placement.py"]\n').encode()
        with open(os.path.join(tmp, "placement.py"), "rb") as fh:
            local_extra = fh.read()
        extra = extra_body if extra_body is not None else local_extra

        drv = DRIVER % {"tmp": tmp, "primary": primary, "extra": extra,
                        "running": running, "force": force}
        p = subprocess.run([sys.executable, "-c", drv], capture_output=True, text=True, timeout=180)
        restarted = (p.returncode == 42)

        staged_now = open(os.path.join(tmp, "placement.py"), "rb").read()
        staged = (staged_now == extra) and (extra != local_extra)

        # #sha-pin: every fetch of the cycle must have carried the SAME resolved commit.
        pinned_ok = True
        try:
            n, bad = open(os.path.join(tmp, "_refs.txt")).read().split()
            pinned_ok = int(n) > 0 and int(bad) == 0
        except Exception:
            pinned_ok = False
        # Did the updater ANNOUNCE its no-restart decision loudly enough for the forced-update
        # route to see it? Without this the route restarts anyway and the gate is inert.
        route_told = None
        try:
            route_told = open(os.path.join(tmp, "_refused.txt")).read().strip() == "1"
        except Exception:
            route_told = None
        told_ok = (expect_route_told is None) or (route_told == expect_route_told)
        ok = ((restarted == expect_restart) and (staged == expect_staged)
              and pinned_ok and told_ok)
        print(f"{'PASS' if ok else 'FAIL'}  {name}")
        if not ok:
            print(f"        restart: got {restarted} want {expect_restart}")
            print(f"        staged : got {staged} want {expect_staged}")
            print(f"        route_told: got {route_told} want {expect_route_told}")
            print(f"        pinned : {pinned_ok} (every fetch used the resolved SHA)")
            print(f"        rc={p.returncode} out={p.stdout.strip()[-300:]!r} err={p.stderr.strip()[-300:]!r}")
        return ok
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


NEW_EXTRA = b"# propagated ahead of the primary\nX = 1\n"
results = []

# 1. THE BUG: stale primary (same VERSION) + a newer extra, forced.
#    Must NOT restart — but must still stage, so the box converges when the primary lands.
results.append(run("mixed set, forced -> stage but DO NOT restart",
                   running="0.3.28", primary_ver="0.3.28", extra_body=NEW_EXTRA,
                   force=True, expect_restart=False, expect_staged=True,
                   expect_route_told=True))

# 2. Same, unforced (the pre-existing #4 path) — unchanged behaviour.
results.append(run("mixed set, unforced -> stage, no restart",
                   running="0.3.28", primary_ver="0.3.28", extra_body=NEW_EXTRA,
                   force=False, expect_restart=False, expect_staged=True,
                   expect_route_told=True))

# 3. A REAL deploy: VERSION moved. Must still restart, or the fleet can never update.
results.append(run("clean deploy (VERSION bumped) -> RESTART",
                   running="0.3.28", primary_ver="0.3.29", extra_body=NEW_EXTRA,
                   force=True, expect_restart=True, expect_staged=True))

# 4. Downgrade must still be refused outright: nothing staged, no restart.
results.append(run("older primary -> refuse, nothing written",
                   running="0.3.29", primary_ver="0.3.28", extra_body=NEW_EXTRA,
                   force=True, expect_restart=False, expect_staged=False,
                   expect_route_told=True))

# 5. Extra unchanged, primary content differs but VERSION does not move (the fake primary always
#    differs from the real local server.py, so this is "primary churn without a bump", NOT "nothing
#    changed"). Must stage nothing new and must not restart.
results.append(run("primary differs but VERSION static -> no restart",
                   running="0.3.28", primary_ver="0.3.28", extra_body=None,
                   force=True, expect_restart=False, expect_staged=False))

if not all(results):
    print("\nFAIL — #mixed-set self-update behaviour")
    sys.exit(1)
print("\nPASS — forced update restarts ONLY on a VERSION bump; staging still converges a "
      "half-updated box; downgrade still refused; every fetch of a cycle used ONE pinned commit")
