# FactCheck 3-way comparison (qwen3.6-27b, HoVer dev claims)

2 claims per arm; knobs: FC_SCOUTS=2, FC_ROUNDS=FC_MIN_ROUNDS=6 (defaults). NEED_MORE = round cap reached without a decision, scored as wrong.

## Task performance

| arm | runs | failed | accuracy | decided | acc. on decided | abstain (NEED_MORE) | SUPPORTED acc. | NOT_SUPPORTED acc. |
|---|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 2 | 0 | **100.0%** (2/2) | 2 | 100.0% | 0.0% | 100.0% | 100.0% |
| OpenAI SDK | 2 | 0 | **0.0%** (0/2) | 0 | n/a | 100.0% | 0.0% | 0.0% |
| LangGraph | 2 | 0 | **0.0%** (0/2) | 0 | n/a | 100.0% | 0.0% | 0.0% |
| nooa | 2 | 0 | **100.0%** (2/2) | 2 | 100.0% | 0.0% | 100.0% | 100.0% |

### Accuracy by hop count

| arm | 3 hops |
|---|---|
| Jac (byLLM) | 100.0% (2/2) |
| OpenAI SDK | 0.0% (0/2) |
| LangGraph | 0.0% (0/2) |
| nooa | 100.0% (2/2) |

### Verdict distribution

| arm | SUPPORTED | NOT_SUPPORTED | NEED_MORE | none (failed) |
|---|---|---|---|---|
| Jac (byLLM) | 1 | 1 | 0 | 0 |
| OpenAI SDK | 0 | 0 | 2 | 0 |
| LangGraph | 0 | 0 | 2 | 0 |
| nooa | 1 | 1 | 0 | 0 |

## Token usage (per claim, mean / median)

| arm | model calls | retries | prompt tokens | completion tokens | total tokens | wall s | total prompt tokens (all claims) |
|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 24 / 24 | 0 / 0 | 160,843 / 160,843 | 2,722 / 2,722 | 163,566 / 163,566 | 174 / 174 | 321,686 |
| OpenAI SDK | 24 / 24 | 0 / 0 | 136,147 / 136,147 | 2,348 / 2,348 | 138,496 / 138,496 | 132 / 132 | 272,294 |
| LangGraph | 24 / 24 | 0 / 0 | 136,147 / 136,147 | 2,348 / 2,348 | 138,496 / 138,496 | 134 / 134 | 272,294 |
| nooa | 20 / 20 | 0 / 0 | 102,884 / 102,884 | 2,467 / 2,467 | 105,350 / 105,350 | 134 / 134 | 205,767 |

Retries: for the Python arms, correction turns after an unparseable reply (traced as `*/retry`); byLLM's own correction calls are not tagged in its usage log and show up only in the call count.

## Per-claim verdicts

| # | hops | gold | Jac (byLLM) | OpenAI SDK | LangGraph | nooa | prompt tokens (jac/openai/langgraph/nooa) |
|---|---|---|---|---|---|---|---|
| 1 | 3 | SUPPORTED | ✓ SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✓ SUPPORTED | 167k/135k/135k/147k |
| 2 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✓ NOT_SUPPORTED | 153k/136k/136k/58k |
