# FactCheck 3-way comparison (GLM-5.2, HoVer dev claims)

3 claims per arm; knobs: FC_SCOUTS=2, FC_ROUNDS=FC_MIN_ROUNDS=6 (defaults). NEED_MORE = round cap reached without a decision, scored as wrong.

## Task performance

| arm | runs | failed | accuracy | decided | acc. on decided | abstain (NEED_MORE) | SUPPORTED acc. | NOT_SUPPORTED acc. |
|---|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 3 | 0 | **100.0%** (3/3) | 3 | 100.0% | 0.0% | 100.0% | 100.0% |

### Accuracy by hop count

| arm | 2 hops | 3 hops | 4 hops |
|---|---|---|---|
| Jac (byLLM) | 100.0% (1/1) | 100.0% (1/1) | 100.0% (1/1) |

### Verdict distribution

| arm | SUPPORTED | NOT_SUPPORTED | NEED_MORE | none (failed) |
|---|---|---|---|---|
| Jac (byLLM) | 2 | 1 | 0 | 0 |

## Token usage (per claim, mean / median)

| arm | model calls | retries | prompt tokens | completion tokens | total tokens | wall s | total prompt tokens (all claims) |
|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 19 / 14 | 0 / 0 | 60,504 / 31,796 | 3,967 / 4,138 | 64,471 / 36,104 | 102 / 93 | 181,513 |

Retries: for the Python arms, correction turns after an unparseable reply (traced as `*/retry`); byLLM's own correction calls are not tagged in its usage log and show up only in the call count.

## Per-claim verdicts

| # | hops | gold | Jac (byLLM) | prompt tokens (jac) |
|---|---|---|---|---|
| 1 | 2 | SUPPORTED | ✓ SUPPORTED | 30k |
| 2 | 3 | SUPPORTED | ✓ SUPPORTED | 31k |
| 3 | 4 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | 118k |
