# FactCheck 3-way comparison (gpt-4.1-mini, HoVer dev claims)

29 claims per arm; knobs: FC_SCOUTS=2, FC_ROUNDS=FC_MIN_ROUNDS=6 (defaults). NEED_MORE = round cap reached without a decision, scored as wrong.

## Task performance

| arm | runs | failed | accuracy | decided | acc. on decided | abstain (NEED_MORE) | SUPPORTED acc. | NOT_SUPPORTED acc. |
|---|---|---|---|---|---|---|---|---|
| nooa | 29 | 0 | **27.6%** (8/29) | 10 | 80.0% | 65.5% | 25.0% | 30.8% |

### Accuracy by hop count

| arm | 2 hops | 3 hops | 4 hops |
|---|---|---|---|
| nooa | 30.0% (3/10) | 44.4% (4/9) | 10.0% (1/10) |

### Verdict distribution

| arm | SUPPORTED | NOT_SUPPORTED | NEED_MORE | none (failed) |
|---|---|---|---|---|
| nooa | 5 | 5 | 19 | 0 |

## Token usage (per claim, mean / median)

| arm | model calls | retries | prompt tokens | completion tokens | total tokens | wall s | total prompt tokens (all claims) |
|---|---|---|---|---|---|---|---|
| nooa | 23 / 24 | 0 / 0 | 131,679 / 127,676 | 1,883 / 1,782 | 133,562 / 129,328 | 29 / 28 | 3,818,692 |

Retries: for the Python arms, correction turns after an unparseable reply (traced as `*/retry`); byLLM's own correction calls are not tagged in its usage log and show up only in the call count.

## Per-claim verdicts

| # | hops | gold | nooa | prompt tokens (nooa) |
|---|---|---|---|---|
| 1 | 3 | SUPPORTED | ✗ NEED_MORE | 158k |
| 2 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | 190k |
| 3 | 2 | SUPPORTED | ✗ NEED_MORE | 127k |
| 4 | 2 | SUPPORTED | ✓ SUPPORTED | 45k |
| 5 | 3 | SUPPORTED | ✓ SUPPORTED | 107k |
| 6 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | 42k |
| 7 | 4 | SUPPORTED | ✗ NEED_MORE | 48k |
| 8 | 4 | SUPPORTED | ✗ NEED_MORE | 185k |
| 9 | 3 | SUPPORTED | ✗ NEED_MORE | 107k |
| 10 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | 49k |
| 11 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | 113k |
| 12 | 4 | SUPPORTED | ✓ SUPPORTED | 55k |
| 13 | 3 | SUPPORTED | ✗ NEED_MORE | 197k |
| 14 | 2 | NOT_SUPPORTED | ✗ NEED_MORE | 117k |
| 15 | 2 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | 104k |
| 16 | 3 | SUPPORTED | ✓ SUPPORTED | 156k |
| 17 | 3 | NOT_SUPPORTED | ✗ NEED_MORE | 203k |
| 18 | 2 | SUPPORTED | ✗ NEED_MORE | 179k |
| 19 | 2 | SUPPORTED | ✗ NOT_SUPPORTED | 215k |
| 20 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | 92k |
| 21 | 4 | SUPPORTED | ✗ NEED_MORE | 78k |
| 22 | 2 | NOT_SUPPORTED | ✗ SUPPORTED | 103k |
| 23 | 2 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | 133k |
| 24 | 3 | NOT_SUPPORTED | ✗ NEED_MORE | 181k |
| 25 | 4 | SUPPORTED | ✗ NEED_MORE | 181k |
| 26 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | 185k |
| 27 | 2 | SUPPORTED | ✗ NEED_MORE | 186k |
| 28 | 2 | SUPPORTED | ✗ NEED_MORE | 176k |
| 29 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | 95k |
