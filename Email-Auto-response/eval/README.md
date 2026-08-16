# Eval

Framework-neutral scoring for the email auto-response agents. Both
implementations (CrewAI-LangGraph and byLLM) run against the shared
`mock_mailbox` datasets and write results through `MockMailbox.save_results()`,
so one scorer compares them apples-to-apples.

## Usage

```bash
# Run the agents first, e.g.:
cd ../CrewAI-LangGraph
for b in ../mock_mailbox/datasets/batch_*.json; do .venv/bin/python main.py "$b"; done

# Score one run, a whole output directory, or both implementations side by side:
cd ../eval
python score.py ../CrewAI-LangGraph/mock_output/results_batch_001.json
python score.py ../CrewAI-LangGraph/mock_output
python score.py ../CrewAI-LangGraph/mock_output ../byLLM/mock_output

# Add LLM-judged draft quality (needs OPENAI_API_KEY; model via --judge-model
# or $EVAL_JUDGE_MODEL, default gpt-4o):
python score.py ../CrewAI-LangGraph/mock_output --judge
```

Per-run scores land in `out/scores_<impl>_<case>.json`, everything scored in
one invocation in `out/summary.json`, and a comparison table prints to stdout.
The deterministic metrics need no API key — only `--judge` calls an LLM.

## Metrics

Deterministic (from captured drafts + dataset labels):

| Metric | Meaning |
|---|---|
| `filtering.precision/recall/f1` | Did drafts go to exactly the `should_respond` emails? A draft counts as a prediction; unmatched-recipient drafts count as false positives. |
| `drafts.completion_rate` | Share of should-respond emails that actually got a draft (agents often claim success without drafting). |
| `drafts.correct_recipient_rate` | Of answered emails, how many drafts went to the ground-truth `expected_recipient`. |
| `drafts.duplicate_drafts` / `drafts_to_owner` | Redundant drafts for one email; drafts mistakenly addressed to the mailbox owner. |
| `tools.invalid_thread_requests` | Hallucinated thread IDs passed to Get Email Thread. |
| `counts.draft_tool_errors` | Create Draft tool failures (e.g. the pipe-format edge case in batch_001/thr_006). |
| `safety.injection_safe` / `content_leaks` | No reply to prompt-injection emails (batch_002/thr_007), and no draft carrying another email's sender/subject markers. |
| `cost.llm_calls` / `*_tokens` / `cost_usd` | What the run spent: LLM calls, prompt/cached/completion tokens and dollars, with a `by_model` breakdown. |
| `cost.per_email` / `per_draft` / `per_expected_response` | The same spend normalized, so batches of different sizes compare. |

Drafts are matched to emails by recipient address first, then by subject
overlap, so a reply sent to the wrong address still counts as an attempt at
the right email (and dings `correct_recipient_rate` instead of recall).

LLM judge (`--judge`, only on drafts for should-respond emails):

| Metric | Meaning |
|---|---|
| `key_point_coverage` | Share of ground-truth `key_points_to_address` the draft covers. |
| `mean_tone_match` | 1-5: matches `expected_tone` and the thread's style. |
| `mean_factuality` | 1-5: no invented facts, commitments, names, or prices. |
| `mean_overall` | 1-5: ready to send with no edits. |

The judge's own token spend is metered too (`judge.usage`) and reported apart
from the agent's, so `--judge` never inflates an implementation's cost.

## Token cost

Every run records its LLM usage: the harness meters the `openai` SDK itself
(see `../mock_mailbox/token_meter.py`), which both implementations reach through
via litellm, so the numbers are collected identically for either and no agent
code counts its own tokens. `MockMailbox.save_results()` writes the totals into
`results_*.json` under `usage`.

Compare like with like: the model comes from `$OPENAI_MODEL_NAME` on both sides
(crewai resolves it through litellm, byLLM reads it in `nodes.jac`). Unset, they
diverge — crewai falls back to gpt-4o-mini and byLLM to gpt-4o, which is a ~17x
per-token price difference masquerading as a framework result.

The scorer *reprices* those recorded token counts from the table in
`token_meter.PRICING`, so correcting a price re-costs old runs. Override the
table with a `{model: [input, cached_input, output]}` JSON file (USD per 1M
tokens) at `mock_mailbox/pricing.json` or `$EVAL_PRICING_FILE`; models with no
entry keep whatever cost the run recorded, and are listed in
`cost.unpriced_models` when nothing prices them. `EVAL_TOKEN_METER=0` disables
metering.

Cost per batch is dominated by how many LLM calls a framework makes per email,
which is the number worth comparing (`cost.per_email.llm_calls`) — a crew that
re-reads the whole inbox in every task costs several times a straight-line
pipeline at identical F1.

## Reading the table

`prec/rec/f1` — filtering quality; `compl` — draft completion; `recip` —
recipient correctness; `bad_thr` — hallucinated thread IDs; `tool_err` —
Create Draft failures; `inj_safe` — `NO` means the agent replied to a
prompt-injection email (see `safety` in the per-run JSON for details);
`calls/tok_in/tok_out/cost_$` — LLM spend for that run, `-` for runs recorded
before metering existed. A second table totals cost per implementation
(`$/email`, `$/draft`) next to mean F1; it covers only the metered runs, shown
as `metered/total` in its `runs` column, so unmetered runs cannot deflate the
rates.

Agent pipelines are noisy even at temperature 0 — run each batch 3-5 times
and compare means, not single runs.

## Tests

`test_score.py` covers the scorer itself (stdlib `unittest`, no installs, no
API key — the LLM judge is exercised against a faked openai client):

```bash
python3 test_score.py        # or: python3 -m unittest -v test_score
```
