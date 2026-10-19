#!/usr/bin/env python3
"""Re-align manifest line ranges to the current sources.

Each manifest entry's "kind name" label is matched, in order, against the
symbols loc_symbols.py finds now.  Hand-split files (more entries than
symbols, last entry ending at EOF) are left alone.  Guard files are skipped.

Usage: loc_resync.py [--write]   (default: dry run; prints what needs a manual tag)
"""
from __future__ import annotations

import difflib
import json
import re
import sys
from pathlib import Path

import loc_symbols as ls

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent

CONFIG_CONST = re.compile(
    r"(glob|assign) (LLM_CALL_PARAMS|SEND_TEMPERATURE|REASONING_EFFORT|REASONING_MODEL|LLM_TEMP)$")


def label(s: ls.Symbol) -> str:
    return f"{s.kind} {s.name}".strip()


def is_config_const(s: ls.Symbol, syms: list[ls.Symbol]) -> bool:
    if CONFIG_CONST.match(label(s)):
        return True
    return s.kind == "stmt" and s.start == s.end and any(
        n.start == s.start and n.kind == "glob" for n in syms)


def resync_file(rel: str, entries: list) -> tuple[list | None, list[str]]:
    """Return (new entries or None if unchanged, notes to print)."""
    p = ROOT / rel
    if not p.exists() or not entries:
        return None, []
    src = p.read_text()
    nlines = len(src.splitlines())
    if p.suffix == ".py":
        syms = ls.python_symbols(src)
    elif p.suffix == ".jac":
        syms = ls.jac_symbols(src)
    else:  # yaml etc.: whole-file entries, clamp to file length
        new = [[e[0], min(e[1], nlines), *e[2:]] for e in entries]
        return (new if new != entries else None), []

    if {(s.start, s.end) for s in syms} == {(e[0], e[1]) for e in entries}:
        return None, []
    if len(entries) > len(syms) and max(e[1] for e in entries) == nlines:
        return None, [f"keep hand-split manifest for {rel}"]

    notes: list[str] = []
    new: list = []
    old_labels = [e[2].strip() for e in entries]
    cur_labels = [label(s) for s in syms]
    sm = difflib.SequenceMatcher(a=old_labels, b=cur_labels, autojunk=False)
    for op, i1, i2, j1, j2 in sm.get_opcodes():
        if op == "equal":
            for e, s in zip(entries[i1:i2], syms[j1:j2]):
                new.append([s.start, s.end, e[2], e[3], e[4] if len(e) > 4 else ""])
        elif op == "replace" and (i2 - i1) == (j2 - j1):
            # same count: renamed or regrouped; carry the tag, flag for review
            for e, s in zip(entries[i1:i2], syms[j1:j2]):
                reason = e[4] if len(e) > 4 else ""
                new.append([s.start, s.end, label(s), e[3], f"{reason} [resync: was '{e[2]}'] ?"])
                notes.append(f"{rel}:{s.start}-{s.end} {label(s)!r} -> replaced '{e[2]}' tag={e[3]}")
        else:
            for s in syms[j1:j2]:
                if is_config_const(s, syms):
                    new.append([s.start, s.end, label(s), "adapter",
                                "model/env config constant (added after initial tagging)"])
                else:
                    new.append([s.start, s.end, label(s), "TODO", "? new symbol, untagged"])
                    notes.append(f"{rel}:{s.start}-{s.end} {label(s)!r} -> NEW")
            for e in entries[i1:i2]:
                notes.append(f"dropped {rel}: {e[2]} ({e[3]})")
    return new, notes


def main() -> None:
    write = "--write" in sys.argv
    for mf in sorted(HERE.glob("*.json")):
        if mf.name.endswith(".guard.json"):
            continue
        man = json.loads(mf.read_text())
        changed = False
        for rel, entries in man.items():
            new, notes = resync_file(rel, entries)
            for n in notes:
                print(f"  {n}")
            if new is not None:
                man[rel] = new
                changed = True
        if changed and write:
            mf.write_text(json.dumps(man, indent=1) + "\n")
            print(f"wrote {mf}")


if __name__ == "__main__":
    main()
