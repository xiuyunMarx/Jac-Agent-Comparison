#!/usr/bin/env python3
"""Score YT-Navigator benchmark result files.

Thin wrapper over the shared evaluator (YT-Navigator/benchmark/evaluate.py)
so scoring lives next to run.py like in the sibling comparison projects.
Identical CLI:

    python score.py out/results_byllm.jsonl out/results_langgraph.jsonl \
        --questions ../YT-Navigator/benchmark/questions.jsonl [--judge] [--report out/report.json]
"""

import runpy
import sys
from pathlib import Path

EVALUATE = Path(__file__).resolve().parent.parent / "YT-Navigator" / "benchmark" / "evaluate.py"

if __name__ == "__main__":
    sys.argv[0] = str(EVALUATE)
    runpy.run_path(str(EVALUATE), run_name="__main__")
