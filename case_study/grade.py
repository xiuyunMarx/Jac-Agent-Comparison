#!/usr/bin/env python3
"""Grade a captured test log against a case's FAIL_TO_PASS / PASS_TO_PASS sets.

    python3 grade.py <case_dir> <test_output.txt>

Uses swebench's own log parser for the repo, so the verdict is the harness's verdict
for the same log. Exit status is 0 when the case is resolved, 1 when it is not.
"""
import json
import sys
from pathlib import Path


def main(argv):
    if len(argv) != 3:
        print(__doc__.strip())
        return 2
    case, log = Path(argv[1]), Path(argv[2])
    meta = json.loads((case / "meta.json").read_text())
    text = log.read_text(errors="replace")

    from swebench.harness.log_parsers import PARSER_REGISTRY
    parser = PARSER_REGISTRY.get(meta["log_parser"]) or PARSER_REGISTRY["parse_log_pytest"]
    try:
        status = parser(text, None)
    except TypeError:
        status = parser(text)

    def split(names):
        ok = [n for n in names if status.get(n) in ("PASSED", "XFAIL")]
        bad = [n for n in names if n not in ok]
        return ok, bad

    f2p_ok, f2p_bad = split(meta["FAIL_TO_PASS"])
    p2p_ok, p2p_bad = split(meta["PASS_TO_PASS"])
    resolved = not f2p_bad and not p2p_bad

    print(f"FAIL_TO_PASS  {len(f2p_ok)}/{len(meta['FAIL_TO_PASS'])} passed")
    for n in f2p_bad:
        print(f"  FAILED  {n}  [{status.get(n, 'not run')}]")
    print(f"PASS_TO_PASS  {len(p2p_ok)}/{len(meta['PASS_TO_PASS'])} passed")
    for n in p2p_bad[:20]:
        print(f"  FAILED  {n}  [{status.get(n, 'not run')}]")
    if len(p2p_bad) > 20:
        print(f"  ... and {len(p2p_bad) - 20} more")
    print()
    print("RESOLVED" if resolved else "NOT RESOLVED")
    return 0 if resolved else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv))
