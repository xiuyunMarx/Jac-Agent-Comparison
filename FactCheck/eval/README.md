# FactCheck 3-way evaluation

Runs the Jac (byLLM), OpenAI SDK and LangGraph fact checkers over
`../hover_claims.tsv` (HoVer dev claims: `label<TAB>hops<TAB>claim`) against
GLM-5.2 on ollama.com and compares task performance and token usage.

```sh
cd FactCheck
set -a; . ../.env; set +a                 # OLLAMA_API_KEY
python eval/run.py --workers 2            # all arms, all claims -> eval/out/<arm>_<idx>.json
python eval/run.py --arms jac --limit 5   # subset; existing results are skipped unless --force
python eval/score.py                      # -> eval/out/report.md (also printed)
```

`run.py` starts one subprocess per (arm, claim) with the arm's own entry point
and the default knobs (`FC_SCOUTS=2`, `FC_ROUNDS=FC_MIN_ROUNDS=6`), the shared
`../wiki_cache`, and a per-call token log: `FC_TRACE` for the Python arms,
`BYLLM_USAGE_LOG` for the Jac arm (needs the jac-main build,
`~/miniconda3/envs/jac-main/bin/jac`; override with `FC_JAC`; Python via `FC_PY`,
default the `jaseci` conda env). Each result records verdict, rounds, findings,
calls, prompt/completion tokens, retries, wall time and the exit status; stdout
and the call log sit next to it.

`score.py` reports accuracy (binary HoVer: NEED_MORE and failed runs count as
wrong), accuracy on decided claims, abstain rate, accuracy by hop count, the
verdict distribution, per-claim mean/median calls and tokens, and a per-claim
verdict table.
