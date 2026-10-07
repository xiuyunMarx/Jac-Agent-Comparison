# FactCheck 3-way comparison (GLM-5.2, HoVer dev claims)

2 claims per arm; knobs: FC_SCOUTS=2, FC_ROUNDS=FC_MIN_ROUNDS=6 (defaults). NEED_MORE = round cap reached without a decision, scored as wrong.

## Task performance

| arm | runs | failed | accuracy | decided | acc. on decided | abstain (NEED_MORE) | SUPPORTED acc. | NOT_SUPPORTED acc. |
|---|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 2 | 0 | **50.0%** (1/2) | 1 | 100.0% | 50.0% | 0.0% | 100.0% |
| OpenAI SDK | 2 | 0 | **50.0%** (1/2) | 1 | 100.0% | 50.0% | 0.0% | 100.0% |
| LangGraph | 2 | 0 | **50.0%** (1/2) | 1 | 100.0% | 50.0% | 0.0% | 100.0% |

### Accuracy by hop count

| arm | 3 hops |
|---|---|
| Jac (byLLM) | 50.0% (1/2) |
| OpenAI SDK | 50.0% (1/2) |
| LangGraph | 50.0% (1/2) |

### Verdict distribution

| arm | SUPPORTED | NOT_SUPPORTED | NEED_MORE | none (failed) |
|---|---|---|---|---|
| Jac (byLLM) | 0 | 1 | 1 | 0 |
| OpenAI SDK | 0 | 1 | 1 | 0 |
| LangGraph | 0 | 1 | 1 | 0 |

## Token usage (per claim, mean / median)

| arm | model calls | retries | prompt tokens | completion tokens | total tokens | wall s | total prompt tokens (all claims) |
|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 23 / 23 | 0 / 0 | 94,759 / 94,759 | 26,126 / 26,126 | 120,884 / 120,884 | 284 / 284 | 189,518 |
| OpenAI SDK | 22 / 22 | 0 / 0 | 87,638 / 87,638 | 21,582 / 21,582 | 109,220 / 109,220 | 247 / 247 | 175,277 |
| LangGraph | 22 / 22 | 0 / 0 | 88,954 / 88,954 | 23,152 / 23,152 | 112,106 / 112,106 | 243 / 243 | 177,909 |

Retries: for the Python arms, correction turns after an unparseable reply (traced as `*/retry`); byLLM's own correction calls are not tagged in its usage log and show up only in the call count.

## Per-claim verdicts

| # | hops | gold | Jac (byLLM) | OpenAI SDK | LangGraph | prompt tokens (jac/openai/langgraph) |
|---|---|---|---|---|---|---|
| 1 | 3 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 85k/121k/127k |
| 2 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 103k/53k/50k |
