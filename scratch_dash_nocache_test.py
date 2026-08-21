"""#dash-nocache — every dashboard HTML page must be served no-store.

WHY THIS SHAPE. The bug was not a wrong header, it was a MISSING one: pages were returned as bare
strings, so the response carried no Cache-Control, no ETag and no Last-Modified, and browsers
heuristically cached them forever. That failure is invisible from the server side — /chat on the
wire serves the new page while an open tab runs the old JS — so it reads as "the fix did not work".

A live header check would only cover pages that exist today. This asserts the STRUCTURAL property:
every route declaring response_class=HTMLResponse returns through the shared header set, and that
set is defined exactly once. A page added later as a bare `return SOME_HTML` fails here.
"""
import ast
import os
import re
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))
failures: list[str] = []


def src(n):
    with open(os.path.join(ROOT, n), encoding="utf-8") as fh:
        return fh.read()


# ---- 1. exactly one definition of the header set
defs = [f for f in os.listdir(ROOT)
        if f.endswith(".py") and not f.startswith("scratch_")
        and re.search(r"^NOCACHE_HEADERS\s*=", src(f), re.M)]
if defs != ["dashboard_html.py"]:
    failures.append(f"NOCACHE_HEADERS must be defined exactly once, in dashboard_html.py; found {defs}")

# ---- 2. it must actually disable caching
blob = src("dashboard_html.py")
m = re.search(r"NOCACHE_HEADERS\s*=\s*\{(.*?)\}", blob, re.S)
if not m:
    failures.append("dashboard_html.py: NOCACHE_HEADERS not parseable")
elif "no-store" not in m.group(1):
    failures.append("NOCACHE_HEADERS no longer contains no-store — heuristic caching returns")

# ---- 3. every HTML page route returns through it
for fn in ("routes_dashboard.py", "routes_peers.py"):
    text = src(fn)
    tree = ast.parse(text)
    lines = text.splitlines()
    for node in ast.walk(tree):
        if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            continue
        decos = "\n".join(lines[d.lineno - 1] for d in node.decorator_list
                          if getattr(d, "lineno", None))
        if "response_class=HTMLResponse" not in decos:
            continue
        # Inspect the RETURN EXPRESSION via AST, not the function's raw text. A first version of
        # this test substring-matched the body, and the explanatory COMMENT above the return
        # contained "NOCACHE_HEADERS" — so reverting the return to a bare string still passed. The
        # negative control below caught it; without one this test would have shipped inert, which
        # is the same class of defect it exists to prevent.
        ok = False
        for r in ast.walk(node):
            if not isinstance(r, ast.Return) or r.value is None:
                continue
            v = r.value
            if isinstance(v, ast.Call):
                f = v.func
                name = getattr(f, "id", None) or getattr(f, "attr", None)
                if name == "_page":
                    ok = True
                elif name == "HTMLResponse" and any(k.arg == "headers" for k in v.keywords):
                    ok = True
            if not ok:
                failures.append(
                    f"{fn}:{r.lineno} route '{node.name}' returns HTML without cache headers "
                    f"(bare value or HTMLResponse with no headers=) — no Cache-Control at all, "
                    f"which browsers cache heuristically")
                break

# ---- 4. the helper it funnels through must really apply them
pg = re.search(r"def _page\(.*?(?=\n    @app\.get)", src("routes_dashboard.py"), re.S)
if pg is None:
    failures.append("routes_dashboard._page is gone — the five pages route through it")
elif "NOCACHE_HEADERS" not in pg.group(0):
    failures.append("routes_dashboard._page no longer applies NOCACHE_HEADERS — all five pages "
                    "silently lose the header together")

if failures:
    print("FAIL — #dash-nocache:")
    for f in failures:
        print("  -", f)
    sys.exit(1)
print("PASS — one header definition, contains no-store, every HTMLResponse route funnels through it")
