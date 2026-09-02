"""Diff the per-call LLM traces of one instance across arms.

usage: python trace_diff.py <results/run-id prefix> <instance_id> [arms...]
e.g.   python trace_diff.py results/glm-8-aligned3 django__django-16255 jac langgraph openai

Reads `<prefix>-<arm>/logs/<instance>/llm_trace.jsonl` (written when the driver
sets $CODEAGENT_TRACE) and prints: per-arm totals; a unified diff of the first
call's messages, tool schemas and response_format against the first arm; the
newest user message of every phase/route start; and a per-call table of
context size, prompt tokens and the tool the model chose.
"""
import json, sys, difflib, os

prefix, inst = sys.argv[1], sys.argv[2]
arms = sys.argv[3:] or ["jac", "langgraph", "openai"]
traces = {}
for a in arms:
    p = f"{prefix}-{a}/logs/{inst}/llm_trace.jsonl"
    if not os.path.exists(p):
        print(f"!! missing {p}"); continue
    traces[a] = [json.loads(l) for l in open(p)]

def content(m):
    c = m.get("content")
    if isinstance(c, list):
        return "".join(x.get("text", "") if isinstance(x, dict) else str(x) for x in c)
    return c or ""

def show_diff(label, a, b, ta, tb):
    if ta == tb:
        print(f"  {label}: identical between {a} and {b}")
        return
    print(f"  {label}: DIFFERS between {a} and {b}")
    for l in difflib.unified_diff(ta.splitlines(), tb.splitlines(), a, b, lineterm="", n=1):
        print("    " + l)

print("== per-arm summary ==")
for a, rows in traces.items():
    pt = sum((r["usage"] or {}).get("prompt_tokens", 0) for r in rows)
    ct = sum((r["usage"] or {}).get("completion_tokens", 0) for r in rows)
    opens = [r["call"] for r in rows if r["messages"] is not None]
    extras = sorted({k for r in rows for k in r["extra"]})
    temps = sorted({str(r["temperature"]) for r in rows})
    print(f"{a:10} calls={len(rows):3} prompt={pt:8,} completion={ct:7,} phase/route starts={opens} temperature={temps} extra_keys={extras}")

print("\n== first call: scaffold comparison ==")
base = arms[0]
b0 = traces[base][0]
for a in arms[1:]:
    if a not in traces: continue
    r0 = traces[a][0]
    bm, am = b0["messages"], r0["messages"]
    print(f"[{base} vs {a}] n_messages {len(bm)} vs {len(am)}; tools {b0['tools']} vs {r0['tools']}")
    for i in range(max(len(bm), len(am))):
        x = content(bm[i]) if i < len(bm) else "<absent>"
        y = content(am[i]) if i < len(am) else "<absent>"
        show_diff(f"message[{i}] role={bm[i].get('role') if i < len(bm) else '?'}", base, a, x, y)
    show_diff("tool_schemas", base, a,
              json.dumps(b0["tool_schemas"], indent=1, sort_keys=True),
              json.dumps(r0["tool_schemas"], indent=1, sort_keys=True))
    show_diff("response_format", base, a, json.dumps(b0["response_format"], sort_keys=True), json.dumps(r0["response_format"], sort_keys=True))

print("\n== every phase/route start: the newest user message, per arm ==")
for a, rows in traces.items():
    print(f"-- {a}")
    for r in rows:
        if r["messages"] is None: continue
        u = r["messages"][-1]
        print(f"  call {r['call']:3} n_msg={r['n_messages']:3} prompt={(r['usage'] or {}).get('prompt_tokens',0):6,} tools={r['tools']}")
        print("     " + content(u).replace("\n", "\n     ")[:1500])

print("\n== per-call table ==")
for a, rows in traces.items():
    print(f"-- {a}")
    for r in rows:
        u = r["usage"] or {}
        rep = r["reply"] or {}
        tcs = [t["function"]["name"] for t in (rep.get("tool_calls") or [])]
        last = r["last"] or (r["messages"] or [{}])[-1]
        lc = len(content(last))
        print(f"  {r['call']:3} msgs={r['n_messages']:3} prompt={u.get('prompt_tokens',0):7,} compl={u.get('completion_tokens',0):5,} last={last.get('role','?'):9} last_len={lc:6} -> {tcs or (content(rep)[:60]+'...' if content(rep) else '')}")
