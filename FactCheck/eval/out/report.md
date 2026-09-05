# FactCheck 3-way comparison (GLM-5.2, HoVer dev claims)

29 claims per arm; knobs: FC_SCOUTS=2, FC_ROUNDS=FC_MIN_ROUNDS=6 (defaults). NEED_MORE = round cap reached without a decision, scored as wrong.

## Task performance

| arm | runs | failed | accuracy | decided | acc. on decided | abstain (NEED_MORE) | SUPPORTED acc. | NOT_SUPPORTED acc. |
|---|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 29 | 0 | **69.0%** (20/29) | 23 | 87.0% | 20.7% | 68.8% | 69.2% |
| OpenAI SDK | 29 | 0 | **72.4%** (21/29) | 22 | 95.5% | 24.1% | 68.8% | 76.9% |
| LangGraph | 29 | 0 | **72.4%** (21/29) | 23 | 91.3% | 20.7% | 75.0% | 69.2% |

### Accuracy by hop count

| arm | 2 hops | 3 hops | 4 hops |
|---|---|---|---|
| Jac (byLLM) | 50.0% (5/10) | 88.9% (8/9) | 70.0% (7/10) |
| OpenAI SDK | 60.0% (6/10) | 100.0% (9/9) | 60.0% (6/10) |
| LangGraph | 70.0% (7/10) | 88.9% (8/9) | 60.0% (6/10) |

### Verdict distribution

| arm | SUPPORTED | NOT_SUPPORTED | NEED_MORE | none (failed) |
|---|---|---|---|---|
| Jac (byLLM) | 13 | 10 | 6 | 0 |
| OpenAI SDK | 12 | 10 | 7 | 0 |
| LangGraph | 13 | 10 | 6 | 0 |

## Token usage (per claim, mean / median)

| arm | model calls | retries | prompt tokens | completion tokens | total tokens | wall s | total prompt tokens (all claims) |
|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 18 / 16 | 0 / 0 | 78,356 / 54,307 | 9,782 / 9,319 | 88,138 / 61,392 | 155 / 139 | 2,272,320 |
| OpenAI SDK | 20 / 22 | 0 / 0 | 94,628 / 93,725 | 6,791 / 6,091 | 101,419 / 102,270 | 84 / 75 | 2,744,220 |
| LangGraph | 20 / 20 | 0 / 0 | 94,396 / 89,746 | 6,770 / 6,014 | 101,166 / 99,158 | 271 / 129 | 2,737,479 |

Retries: for the Python arms, correction turns after an unparseable reply (traced as `*/retry`); byLLM's own correction calls are not tagged in its usage log and show up only in the call count.

## Per-claim verdicts

| # | hops | gold | Jac (byLLM) | OpenAI SDK | LangGraph | prompt tokens (jac/openai/langgraph) |
|---|---|---|---|---|---|---|
| 1 | 3 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 56k/59k/83k |
| 2 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 29k/55k/89k |
| 3 | 2 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NOT_SUPPORTED | 181k/154k/164k |
| 4 | 2 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 31k/29k/29k |
| 5 | 3 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 31k/78k/74k |
| 6 | 4 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 117k/51k/41k |
| 7 | 4 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 54k/46k/46k |
| 8 | 4 | SUPPORTED | ✓ SUPPORTED | ✗ NEED_MORE | ✓ SUPPORTED | 28k/153k/125k |
| 9 | 3 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 26k/62k/62k |
| 10 | 4 | NOT_SUPPORTED | ✗ SUPPORTED | ✗ NEED_MORE | ✗ SUPPORTED | 96k/29k/96k |
| 11 | 4 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✗ NEED_MORE | 173k/160k/153k |
| 12 | 4 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 31k/34k/37k |
| 13 | 3 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 61k/153k/97k |
| 14 | 2 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 125k/96k/93k |
| 15 | 2 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 38k/171k/61k |
| 16 | 3 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 35k/119k/94k |
| 17 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 62k/106k/167k |
| 18 | 2 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 143k/140k/142k |
| 19 | 2 | SUPPORTED | ✗ NOT_SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 35k/87k/65k |
| 20 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 130k/110k/113k |
| 21 | 4 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 51k/93k/88k |
| 22 | 2 | NOT_SUPPORTED | ✗ SUPPORTED | ✗ SUPPORTED | ✓ NOT_SUPPORTED | 32k/66k/89k |
| 23 | 2 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 32k/56k/56k |
| 24 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 33k/52k/72k |
| 25 | 4 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 115k/111k/123k |
| 26 | 4 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 156k/140k/167k |
| 27 | 2 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 163k/161k/147k |
| 28 | 2 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 28k/58k/59k |
| 29 | 3 | NOT_SUPPORTED | ✗ NEED_MORE | ✓ NOT_SUPPORTED | ✗ NEED_MORE | 164k/101k/91k |
