"""loc_resync.py: re-align manifest line ranges to current sources by matching 'kind name' in order.
Usage: resync.py [--write]. Prints unmatched current symbols (need manual tag)."""
import json, sys, re
from pathlib import Path
from pathlib import Path
HERE = Path(__file__).resolve().parent; sys.path.insert(0, str(HERE)); import os; os.chdir(HERE.parent)
import loc_symbols as ls
write = "--write" in sys.argv
def label(s): return f"{s.kind} {s.name}".strip()
for mf in sorted(Path("loc_manifests").glob("*.json")):
    man = json.loads(mf.read_text()); changed = False
    for rel, entries in man.items():
        p = Path(rel)
        if not p.exists() or not entries: continue
        src = p.read_text(); nlines = len(src.splitlines())
        if p.suffix == ".py": syms = ls.python_symbols(src)
        elif p.suffix == ".jac": syms = ls.jac_symbols(src)
        else:
            # yaml etc: single/whole-file entries; clamp ends to file length
            new = [[e[0], min(e[1], nlines), *e[2:]] for e in entries]
            if new != entries: man[rel] = new; changed = True
            continue
        cur = [(s.start, s.end) for s in syms]; old = [(e[0], e[1]) for e in entries]
        if set(cur) == set(old): continue
        if len(entries) > len(syms) and max(e[1] for e in entries) == nlines and all(e[1] <= nlines for e in entries):
            print(f"  keep hand-split manifest for {rel}"); continue
        changed = True
        # sequence-match by label
        import difflib
        a = [e[2].strip() for e in entries]; b = [label(s) for s in syms]
        sm = difflib.SequenceMatcher(a=a, b=b, autojunk=False)
        new = []; unmatched = []
        for tag, i1, i2, j1, j2 in sm.get_opcodes():
            if tag == "equal":
                for k in range(i2 - i1):
                    e = entries[i1 + k]; s = syms[j1 + k]
                    new.append([s.start, s.end, e[2], e[3], e[4] if len(e) > 4 else ""])
            elif tag == "replace" and (i2 - i1) == (j2 - j1):
                # same count: likely renamed/regrouped; carry tag but flag
                for k in range(i2 - i1):
                    e = entries[i1 + k]; s = syms[j1 + k]
                    new.append([s.start, s.end, label(s), e[3], f"{e[4] if len(e)>4 else ''} [resync: was '{e[2]}'] ?"])
                    unmatched.append((s, f"replaced '{e[2]}' tag={e[3]}"))
            else:
                for s in syms[j1:j2]:
                    if re.match(r"(glob|assign) (LLM_CALL_PARAMS|SEND_TEMPERATURE|REASONING_EFFORT|REASONING_MODEL|LLM_TEMP)$", label(s)) or (s.kind=="stmt" and s.start==s.end and any(n.start==s.start and n.kind=="glob" for n in syms)):
                        new.append([s.start, s.end, label(s), "adapter", "model/env config constant (added after initial tagging)"])
                    else:
                        new.append([s.start, s.end, label(s), "TODO", "? new symbol, untagged"])
                        unmatched.append((s, "NEW"))
                for e in entries[i1:i2]:
                    print(f"  dropped {rel}: {e[2]} ({e[3]})")
        man[rel] = new
        for s, why in unmatched:
            print(f"  {rel}:{s.start}-{s.end} {label(s)!r} -> {why}")
    if changed and write:
        mf.write_text(json.dumps(man, indent=1) + "\n"); print(f"wrote {mf}")
