#!/usr/bin/env python3
"""Split each source file into top-level symbols with line ranges.

Python: via `ast` (functions, classes, assignments, if __main__ blocks...).
Jac:    brace/semicolon scanner at depth 0 (obj/node/walker/def/impl/sem/glob/
        with entry/import ...).

Used by loc_count.py --core to tag symbols as core vs adapter.
"""
from __future__ import annotations

import ast
import re
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class Symbol:
    kind: str
    name: str
    start: int   # 1-based inclusive
    end: int     # 1-based inclusive


def python_symbols(src: str) -> list[Symbol]:
    tree = ast.parse(src)
    out: list[Symbol] = []
    for node in tree.body:
        end = getattr(node, "end_lineno", node.lineno)
        # include decorators
        start = node.lineno
        if hasattr(node, "decorator_list") and node.decorator_list:
            start = min(d.lineno for d in node.decorator_list)
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            out.append(Symbol("def", node.name, start, end))
        elif isinstance(node, ast.ClassDef):
            out.append(Symbol("class", node.name, start, end))
        elif isinstance(node, (ast.Import, ast.ImportFrom)):
            out.append(Symbol("import", "", start, end))
        elif isinstance(node, (ast.Assign, ast.AnnAssign, ast.AugAssign)):
            tgt = node.targets[0] if isinstance(node, ast.Assign) else node.target
            name = ast.unparse(tgt) if hasattr(ast, "unparse") else "?"
            out.append(Symbol("assign", name, start, end))
        elif isinstance(node, ast.If):
            test = ast.unparse(node.test) if hasattr(ast, "unparse") else "if"
            out.append(Symbol("if", test[:40], start, end))
        elif isinstance(node, ast.Expr) and isinstance(getattr(node.value, "value", None), str):
            out.append(Symbol("docstring", "", start, end))
        else:
            out.append(Symbol(type(node).__name__.lower(), "", start, end))
    return out


_JAC_HEAD = re.compile(
    r"^\s*(?:@\w+(?:\([^)]*\))?\s*)*"          # decorators
    r"(?:(?:async|static|override|abstract|pub|priv|prot|prev|cap)\s+)*"
    r"(?P<kw>import|include|glob|let|sem|obj|node|edge|walker|enum|class|def|can|impl|with|test|has)\b"
    r"\s*(?:entry\s*)?(?P<name>[\w.]+)?"
)


def jac_symbols(src: str) -> list[Symbol]:
    """Statements at brace depth 0. A statement ends at `;` (depth 0) or at the
    `}` that closes a `{` opened at depth 0."""
    out: list[Symbol] = []
    i, n = 0, len(src)
    line = 1
    depth = 0
    stmt_start_line = None
    stmt_start_idx = None

    def close(end_line: int) -> None:
        nonlocal stmt_start_line, stmt_start_idx
        head = src[stmt_start_idx:stmt_start_idx + 200]
        m = _JAC_HEAD.match(head)
        kw = m.group("kw") if m else "stmt"
        name = (m.group("name") or "") if m else ""
        if kw == "with":
            name = "entry"
        out.append(Symbol(kw, name, stmt_start_line, end_line))
        stmt_start_line = None
        stmt_start_idx = None

    while i < n:
        ch = src[i]
        if ch == "\n":
            line += 1; i += 1; continue
        if src.startswith("#*", i):
            j = src.find("*#", i + 2); j = n if j < 0 else j + 2
            line += src.count("\n", i, j); i = j; continue
        if ch == "#":
            j = src.find("\n", i); i = n if j < 0 else j; continue
        if ch in "\"'":
            q = src[i:i + 3] if src.startswith(ch * 3, i) else ch
            j = i + len(q)
            while j < n:
                if src[j] == "\\": j += 2; continue
                if src.startswith(q, j): j += len(q); break
                if len(q) == 1 and src[j] == "\n": break
                j += 1
            if stmt_start_line is None:
                stmt_start_line, stmt_start_idx = line, i
            line += src.count("\n", i, j); i = j; continue
        if ch.isspace():
            i += 1; continue
        if stmt_start_line is None:
            stmt_start_line, stmt_start_idx = line, i
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                close(line)
        elif ch == ";" and depth == 0:
            close(line)
        i += 1
    if stmt_start_line is not None:
        close(line)
    return out


def symbols_for(p: Path) -> list[Symbol]:
    src = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix == ".py":
        return python_symbols(src)
    if p.suffix == ".jac":
        return jac_symbols(src)
    return [Symbol("file", p.name, 1, max(1, src.count("\n") + 1))]


if __name__ == "__main__":
    for arg in sys.argv[1:]:
        p = Path(arg)
        src = p.read_text(encoding="utf-8", errors="replace").splitlines()
        print(f"### {p}")
        for s in symbols_for(p):
            body = src[s.start - 1:s.end]
            sloc = sum(1 for l in body if l.strip() and not l.strip().startswith("#"))
            first = src[s.start - 1].strip()[:90]
            print(f"  {s.start:4d}-{s.end:4d} {sloc:4d}  {s.kind:9s} {s.name:40s} {first}")
