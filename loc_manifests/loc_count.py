#!/usr/bin/env python3
"""Count developer-written source per agent arm (Jac vs. framework baselines).

Metrics per file (all exclude .venv, caches, tests, eval harnesses, datasets):

  phys     physical lines
  blank    whitespace-only lines
  comment  lines that are only a comment
  sloc     phys - blank - comment  (what people usually call "LOC")
  prompt   sloc lines that lie entirely inside a string literal spanning >1 line
           (docstrings, `sem` strings, multi-line prompt templates). These are
           natural-language spec, not glue.
  logic    sloc - prompt  (orchestration / glue / plumbing the developer wrote)
  codechars non-whitespace chars outside string literals (formatting-insensitive glue)
  strchars  non-whitespace chars inside string literals (prompts, sem, docstrings)

Usage:
  python loc_count.py            # markdown table, agent-core scope
  python loc_count.py --scope all
  python loc_count.py --files    # per-file breakdown
  python loc_count.py --csv out.csv
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import sys
import tokenize
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MANIFEST_DIR = ROOT / "loc_manifests"   # <app>.json: {file: [[start,end,"kind name",tag,reason],...]}

# --------------------------------------------------------------------------
# Scope manifest.  "core" = the agent itself: orchestration, nodes, tools,
# prompts, entry point.  "all" = every first-party source file in the arm
# (still excluding tests, eval harnesses, vendored deps, datasets, web UI).
# Paths are relative to the arm directory; globs allowed.
# --------------------------------------------------------------------------
ARMS: dict[str, dict[str, dict[str, list[str]]]] = {
    "FactCheck": {
        "Jac":       {"core": ["fact_check.jac"], "all": ["**/*.jac", "**/*.py"]},
        "LangGraph": {"core": ["fact_check.py"], "all": ["**/*.py"]},
        "OpenaiSDK": {"core": ["fact_check.py"], "all": ["**/*.py"]},
    },
    "Email-Auto-response": {
        "byLLM":            {"core": ["*.jac"], "all": ["**/*.jac", "**/*.py"]},
        "CrewAI-LangGraph": {"core": ["main.py", "src/**/*.py", "src/**/*.yaml"],
                             "all": ["**/*.py", "**/*.yaml"]},
        "openai_sdk":       {"core": ["*.py"], "all": ["**/*.py"]},
    },
    "meeting-assistant": {
        "byLLM":      {"core": ["*.jac"], "all": ["**/*.jac", "**/*.py"]},
        "CrewAI":     {"core": ["src/**/*.py", "src/**/*.yaml"],
                       "all": ["src/**/*.py", "src/**/*.yaml"]},
        "openai_sdk": {"core": ["*.py"], "all": ["**/*.py"]},
    },
    "YTNavigator": {
        "byLLM":      {"core": ["*.jac"], "all": ["**/*.jac", "**/*.py"]},
        # Django app: count only the agent layer, not views/models/scraping/UI.
        "langchain":  {"core": ["app/services/agent/*.py",
                                "app/services/vector_database/tools/*.py",
                                "app/schemas/agent.py", "app/schemas/tools.py"],
                       "all": ["app/**/*.py", "yt_navigator/**/*.py", "manage.py"]},
        "openai_sdk": {"core": ["*.py"], "all": ["**/*.py"]},
    },
    "CodeAgent": {
        "Jac":        {"core": ["*.jac"], "all": ["**/*.jac", "**/*.py"]},
        "langgraph":  {"core": ["*.py"], "all": ["**/*.py"]},
        "openai_sdk": {"core": ["*.py"], "all": ["**/*.py"]},
        "NOOA":       {"core": ["*.py", "tools/*.py"], "all": ["**/*.py"]},
    },
}

EXCLUDE_PARTS = {".venv", "venv", "__pycache__", ".jac", "tests", "test",
                 "migrations", "node_modules", "benchmark", "eval", "datasets",
                 "mock_output", "logs"}


@dataclass
class Counts:
    phys: int = 0
    blank: int = 0
    comment: int = 0
    prompt: int = 0
    nwchars: int = 0
    strchars: int = 0   # non-ws chars inside string literals (prompt/spec text)
    files: int = 0
    paths: list[str] = field(default_factory=list)

    @property
    def sloc(self) -> int:
        return self.phys - self.blank - self.comment

    @property
    def logic(self) -> int:
        return self.sloc - self.prompt

    @property
    def codechars(self) -> int:
        return self.nwchars - self.strchars

    def add(self, o: "Counts") -> None:
        self.phys += o.phys; self.blank += o.blank; self.comment += o.comment
        self.prompt += o.prompt; self.nwchars += o.nwchars; self.strchars += o.strchars; self.files += o.files
        self.paths += o.paths


def _finish(lines: list[str], comment_only: set[int], in_str: set[int],
            str_by_line: dict[int, int] | None = None,
            keep: set[int] | None = None) -> Counts:
    """Given per-line classification, produce Counts. `keep` restricts to those lines."""
    str_by_line = str_by_line or {}
    if keep is not None:
        lines = [ln if i in keep else "" for i, ln in enumerate(lines, 1)]
    c = Counts(phys=len(lines) if keep is None else len(keep & set(range(1, len(lines) + 1))),
               files=1)
    for i, ln in enumerate(lines, 1):
        if keep is not None and i not in keep:
            continue
        c.strchars += str_by_line.get(i, 0)
        if not ln.strip():
            c.blank += 1
        elif i in comment_only:
            c.comment += 1
        else:
            c.nwchars += sum(1 for ch in ln if not ch.isspace())
            if i in in_str:
                c.prompt += 1
    return c


def count_python(src: str, keep: set[int] | None = None) -> Counts:
    lines = src.splitlines()
    comment_only: set[int] = set()
    in_str: set[int] = set()
    code_on_line: set[int] = set()
    try:
        toks = list(tokenize.generate_tokens(io.StringIO(src).readline))
    except (tokenize.TokenError, SyntaxError):
        return count_generic(src, line_comment="#", keep=keep)
    str_by_line: dict[int, int] = {}
    for tok in toks:
        (sl, _), (el, _) = tok.start, tok.end
        if tok.type == tokenize.COMMENT:
            continue
        if tok.type == tokenize.STRING:
            for k, part in enumerate(tok.string.split("\n")):
                str_by_line[sl + k] = str_by_line.get(sl + k, 0) + sum(1 for ch in part if not ch.isspace())
        if tok.type in (tokenize.NL, tokenize.NEWLINE, tokenize.INDENT,
                        tokenize.DEDENT, tokenize.ENDMARKER, tokenize.ENCODING):
            continue
        if tok.type == tokenize.STRING and el > sl:
            # interior lines (and the closing line) are pure string text
            for l in range(sl + 1, el + 1):
                in_str.add(l)
        for l in range(sl, el + 1):
            code_on_line.add(l)
    for i, ln in enumerate(lines, 1):
        s = ln.strip()
        if s.startswith("#") and i not in code_on_line:
            comment_only.add(i)
    return _finish(lines, comment_only, in_str, str_by_line, keep)


def count_jac(src: str, keep: set[int] | None = None) -> Counts:
    """Jac: `#` line comments, `#* ... *#` block comments, Python-style strings
    (incl. triple-quoted docstrings and `sem` strings)."""
    lines = src.splitlines()
    comment_only: set[int] = set()
    in_str: set[int] = set()
    code_on_line: set[int] = set()
    i, n = 0, len(src)
    line = 1
    str_by_line: dict[int, int] = {}

    def mark_code(l: int) -> None:
        code_on_line.add(l)

    while i < n:
        ch = src[i]
        if ch == "\n":
            line += 1; i += 1; continue
        if src.startswith("#*", i):
            j = src.find("*#", i + 2)
            j = n if j < 0 else j + 2
            line += src.count("\n", i, j); i = j; continue
        if ch == "#":
            j = src.find("\n", i)
            i = n if j < 0 else j; continue
        if ch in "\"'":
            q = src[i:i + 3] if src.startswith(ch * 3, i) else ch
            j = i + len(q)
            while j < n:
                if src[j] == "\\":
                    j += 2; continue
                if src.startswith(q, j):
                    j += len(q); break
                if len(q) == 1 and src[j] == "\n":
                    break
                j += 1
            for k, part in enumerate(src[i:j].split("\n")):
                str_by_line[line + k] = str_by_line.get(line + k, 0) + sum(1 for ch in part if not ch.isspace())
            start_line = line
            end_line = line + src.count("\n", i, j)
            if end_line > start_line:
                for l in range(start_line + 1, end_line + 1):
                    in_str.add(l)
            for l in range(start_line, end_line + 1):
                mark_code(l)
            line = end_line; i = j; continue
        if not ch.isspace():
            mark_code(line)
        i += 1
    for k, ln in enumerate(lines, 1):
        if ln.strip() and k not in code_on_line:
            comment_only.add(k)
    return _finish(lines, comment_only, in_str, str_by_line, keep)


def count_generic(src: str, line_comment: str = "#", keep: set[int] | None = None) -> Counts:
    """YAML etc.: `#` comments; block scalars (`|`, `>`) count as prompt text."""
    lines = src.splitlines()
    comment_only = {i for i, ln in enumerate(lines, 1)
                    if ln.strip().startswith(line_comment)}
    in_str: set[int] = set()
    block_indent = None
    for i, ln in enumerate(lines, 1):
        if not ln.strip() or i in comment_only:
            continue
        ind = len(ln) - len(ln.lstrip())
        if block_indent is not None:
            if ind > block_indent:
                in_str.add(i); continue
            block_indent = None
        s = ln.rstrip()
        if s.endswith(("|", ">", "|-", ">-", "|+", ">+")) and ":" in s:
            block_indent = ind
    str_by_line = {i: sum(1 for ch in lines[i - 1] if not ch.isspace()) for i in in_str}
    return _finish(lines, comment_only, in_str, str_by_line, keep)


def count_file(p: Path, keep: set[int] | None = None) -> Counts:
    src = p.read_text(encoding="utf-8", errors="replace")
    if p.suffix == ".py":
        c = count_python(src, keep)
    elif p.suffix == ".jac":
        c = count_jac(src, keep)
    else:
        c = count_generic(src, keep=keep)
    c.paths = [str(p.relative_to(ROOT))]
    return c


def load_manifest(app: str) -> dict[str, list]:
    f = MANIFEST_DIR / f"{app}.json"
    return json.loads(f.read_text()) if f.exists() else {}


def core_lines(manifest: dict[str, list], rel: str) -> set[int] | None:
    """Line numbers tagged core for this file, or None if the file is untagged."""
    if rel not in manifest:
        return None
    keep: set[int] = set()
    for start, end, _name, tag, *_ in manifest[rel]:
        if tag == "core":
            keep.update(range(start, end + 1))
    return keep


def collect(arm_dir: Path, globs: list[str]) -> list[Path]:
    out: set[Path] = set()
    for g in globs:
        for p in arm_dir.glob(g):
            if not p.is_file():
                continue
            rel = p.relative_to(arm_dir)
            if any(part in EXCLUDE_PARTS for part in rel.parts[:-1]):
                continue
            if p.name.startswith("test_") or p.name == "conftest.py":
                continue
            out.add(p)
    return sorted(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scope", choices=["core", "all"], default="core",
                    help="core: agent-layer files; all: every first-party file")
    ap.add_argument("--core", action="store_true",
                    help="count only symbols tagged core in loc_manifests/<app>.json "
                         "(execution logic + state; drops CLI/harness/telemetry adapters)")
    ap.add_argument("--files", action="store_true", help="per-file rows")
    ap.add_argument("--csv", type=Path)
    args = ap.parse_args()

    rows = []
    for app, arms in ARMS.items():
        for arm, scopes in arms.items():
            arm_dir = ROOT / app / arm
            if not arm_dir.is_dir():
                print(f"!! missing {arm_dir}", file=sys.stderr); continue
            total = Counts()
            per_file = []
            manifest = load_manifest(app) if args.core else {}
            for p in collect(arm_dir, scopes[args.scope]):
                rel = str(p.relative_to(ROOT))
                keep = core_lines(manifest, rel) if args.core else None
                if args.core and keep is None:
                    print(f"!! {rel}: no manifest entry, counting whole file", file=sys.stderr)
                c = count_file(p, keep)
                per_file.append(c)
                total.add(c)
            rows.append((app, arm, total, per_file))

    hdr = ["app", "arm", "files", "phys", "sloc", "logic", "prompt", "comment", "codechars", "strchars"]
    print(f"scope = {args.scope}{' + core-only symbols' if args.core else ''}\n")
    print("| " + " | ".join(hdr) + " | vs Jac (logic) |")
    print("|" + "---|" * (len(hdr) + 1))
    jac_logic: dict[str, int] = {}
    for app, arm, t, _ in rows:
        if arm.lower() in ("jac", "byllm"):
            jac_logic[app] = t.logic
    for app, arm, t, per_file in rows:
        ratio = f"{t.logic / jac_logic[app]:.2f}x" if app in jac_logic and jac_logic[app] else ""
        print(f"| {app} | {arm} | {t.files} | {t.phys} | {t.sloc} | {t.logic} | "
              f"{t.prompt} | {t.comment} | {t.codechars} | {t.strchars} | {ratio} |")
        if args.files:
            for c in per_file:
                print(f"|  | `{c.paths[0]}` | | {c.phys} | {c.sloc} | {c.logic} | "
                      f"{c.prompt} | {c.comment} | {c.codechars} | {c.strchars} | |")

    if args.csv:
        with args.csv.open("w", newline="") as f:
            w = csv.writer(f)
            w.writerow(hdr + ["path"])
            for app, arm, t, per_file in rows:
                w.writerow([app, arm, t.files, t.phys, t.sloc, t.logic, t.prompt,
                            t.comment, t.codechars, t.strchars, "(total)"])
                for c in per_file:
                    w.writerow([app, arm, 1, c.phys, c.sloc, c.logic, c.prompt,
                                c.comment, c.codechars, c.strchars, c.paths[0]])
        print(f"\nwrote {args.csv}")


if __name__ == "__main__":
    main()
