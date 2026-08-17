"""#node-optout PARITY test — the rule must have exactly ONE definition, and every placement
path must route through it.

WHY THIS SHAPE. The obvious test ("t2a refuses an opted-out node") passes happily while the other
five placement paths disagree with it — which is exactly the state this codebase was in: six inline
copies of `(n.ram_enabled or n.vram_enabled)` in the media leaves, and NO copy at all in the
vision-tower encode picker or the /compile_shards candidate list, both of which were written later.
A per-path test cannot see that; it tests the path you remembered to write a test for. So this
asserts the STRUCTURAL property instead: the condition is spelled once, and nothing re-spells it.

This is deliberately a source-level test. It has to run on the controller box, which has no torch,
so importing engine_load is not an option — and a behavioural test built from a hand-written fake
Node would only re-assert my own transcription of the rule, which is how a guard ships inert.
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

# The rule, spelled the way it must NOT be spelled again anywhere but its single definition.
RESPELLINGS = [
    re.compile(r"\.ram_enabled\s+or\s+\w*\.?vram_enabled"),
    re.compile(r"\.vram_enabled\s+or\s+\w*\.?ram_enabled"),
    re.compile(r"not\s+\w+\.ram_enabled\s+and\s+not\s+\w+\.vram_enabled"),
]

failures: list[str] = []


def _src(name: str) -> str:
    with open(os.path.join(ROOT, name), encoding="utf-8") as fh:
        return fh.read()


def _strip_strings_and_comments(src: str) -> str:
    """Docstrings legitimately QUOTE the old spelling to explain the history; only live code counts."""
    tree = ast.parse(src)
    spans: list[tuple[int, int]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Constant) and isinstance(node.value, str):
            if node.end_lineno is not None:
                spans.append((node.lineno, node.end_lineno))
    lines = src.splitlines()
    out = []
    for i, line in enumerate(lines, start=1):
        if any(a <= i <= b for a, b in spans):
            out.append("")
            continue
        out.append(line.split("#", 1)[0])
    return "\n".join(out)


# ---------------------------------------------------------------- 1. exactly one definition
server = _src("server.py")
if "def placement_enabled" not in server:
    failures.append("server.py: Node.placement_enabled is GONE — the single definition of the rule")

n_defs = sum(_src(f).count("def placement_enabled")
             for f in os.listdir(ROOT) if f.endswith(".py") and not f.startswith("scratch_"))
if n_defs != 1:
    failures.append(f"placement_enabled is defined {n_defs} times across the tree; must be exactly 1")

# ---------------------------------------------------------------- 2. nothing re-spells it
for fn in sorted(f for f in os.listdir(ROOT) if f.endswith(".py")):
    if fn.startswith("scratch_") or fn == os.path.basename(__file__):
        continue
    try:
        code = _strip_strings_and_comments(_src(fn))
    except SyntaxError as exc:
        failures.append(f"{fn}: does not parse ({exc})")
        continue
    for pat in RESPELLINGS:
        for m in pat.finditer(code):
            line = code[:m.start()].count("\n") + 1
            # server.py's own property body IS the definition; that one is allowed.
            if fn == "server.py" and "return self.ram_enabled or self.vram_enabled" in \
                    code.splitlines()[line - 1]:
                continue
            failures.append(
                f"{fn}:{line}: re-spells the opt-out rule inline ({m.group(0)!r}) — "
                f"use `.placement_enabled` so there stays exactly one definition")

# ------------------------------------------- 2b. _place_filter must actually APPLY the rule
# The leaves no longer pre-filter on placement_enabled (so that an opted-out node survives long
# enough for _place_filter to name it as the reason a pin failed). That makes _place_filter the
# ONLY thing standing between a media load and an off-limits node — if this call is ever dropped
# during a refactor, all five media leaves silently lose the rule at once.
_pf = re.search(r"def _place_filter\(.*?(?=\n    def )", _src("engine_load.py"), re.S)
if _pf is None:
    failures.append("engine_load.py: _place_filter is gone — the media leaves depend on it")
elif "placement_enabled" not in _pf.group(0):
    failures.append("engine_load.py: _place_filter no longer applies placement_enabled — every "
                    "media leaf lost the opt-out rule with it (they do not check it themselves)")

# ---------------------------------------------------------------- 3. every load dispatch is gated
# Each load in engine_load.py sends {"type": "load", ...} on a link. That link must come from
# _load_link (which checks the opt-out), never from a bare links.get.
el = _src("engine_load.py")
el_lines = el.splitlines()
load_sends = [i + 1 for i, l in enumerate(el_lines) if re.search(r'link\.send\(\{\s*$', l)
              or re.search(r'link\.send\(\{"type":\s*"load"', l)]
send_load_lines = []
for i, line in enumerate(el_lines, start=1):
    if '"type": "load"' in line:
        send_load_lines.append(i)

for ln in send_load_lines:
    # Walk back to the nearest link binding above this dispatch.
    bind = None
    for j in range(ln - 1, max(0, ln - 60), -1):
        if re.search(r"^\s*link\s*=\s*", el_lines[j - 1]):
            bind = (j, el_lines[j - 1].strip())
            break
    if bind is None:
        failures.append(f"engine_load.py:{ln}: sends a load but no `link =` binding found above it")
    elif "_load_link(" not in bind[1]:
        failures.append(
            f"engine_load.py:{ln}: sends a load using a link bound at :{bind[0]} "
            f"({bind[1]!r}) that did NOT go through _load_link — an opted-out node "
            f"would be dispatched to")

# ---------------------------------------------------------------- 4. later-written pickers gated
for fn, needle in (("media_encode.py", "placement_enabled"),
                   ("routes_shards.py", "placement_enabled")):
    if needle not in _src(fn):
        failures.append(f"{fn}: selects nodes for work but never consults placement_enabled — "
                        f"this is the exact gap that let an off-limits node keep receiving work")

# ---------------------------------------------------------------- 5. teardown must NOT be gated
# Disabling a node has to let it DRAIN. If unload/teardown ever started refusing, a node switched
# off would strand whatever is resident on it.
for i, line in enumerate(el_lines, start=1):
    if '"type": "unload"' in line:
        for j in range(i - 1, max(0, i - 30), -1):
            if re.search(r"^\s*(_?ln|link)\s*=\s*", el_lines[j - 1]):
                if "_load_link(" in el_lines[j - 1]:
                    failures.append(
                        f"engine_load.py:{i}: UNLOAD is going through _load_link — an opted-out "
                        f"node could not be drained, which inverts the intent")
                break

if failures:
    print("FAIL — #node-optout parity:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print(f"PASS — opt-out rule has exactly 1 definition; {len(send_load_lines)} load dispatches gated; "
       "teardown ungated; later-written pickers consult it")
