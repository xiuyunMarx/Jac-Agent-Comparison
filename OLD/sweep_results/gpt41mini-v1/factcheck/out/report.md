# FactCheck 3-way comparison (gpt-4.1-mini, HoVer dev claims)

29 claims per arm; knobs: FC_SCOUTS=2, FC_ROUNDS=FC_MIN_ROUNDS=6 (defaults). NEED_MORE = round cap reached without a decision, scored as wrong.

## Task performance

| arm | runs | failed | accuracy | decided | acc. on decided | abstain (NEED_MORE) | SUPPORTED acc. | NOT_SUPPORTED acc. |
|---|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 29 | 0 | **27.6%** (8/29) | 10 | 80.0% | 65.5% | 43.8% | 7.7% |
| OpenAI SDK | 29 | 0 | **20.7%** (6/29) | 9 | 66.7% | 69.0% | 18.8% | 23.1% |
| LangGraph | 29 | 1 | **10.3%** (3/29) | 4 | 75.0% | 82.8% | 12.5% | 7.7% |

### Accuracy by hop count

| arm | 2 hops | 3 hops | 4 hops |
|---|---|---|---|
| Jac (byLLM) | 20.0% (2/10) | 44.4% (4/9) | 20.0% (2/10) |
| OpenAI SDK | 30.0% (3/10) | 11.1% (1/9) | 20.0% (2/10) |
| LangGraph | 20.0% (2/10) | 0.0% (0/9) | 10.0% (1/10) |

### Verdict distribution

| arm | SUPPORTED | NOT_SUPPORTED | NEED_MORE | none (failed) |
|---|---|---|---|---|
| Jac (byLLM) | 8 | 2 | 19 | 0 |
| OpenAI SDK | 4 | 5 | 20 | 0 |
| LangGraph | 3 | 1 | 24 | 1 |

## Token usage (per claim, mean / median)

| arm | model calls | retries | prompt tokens | completion tokens | total tokens | wall s | total prompt tokens (all claims) |
|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 24 / 24 | 0 / 0 | 128,008 / 142,783 | 2,760 / 2,404 | 130,768 / 145,634 | 54 / 50 | 3,712,229 |
| OpenAI SDK | 22 / 24 | 0 / 0 | 92,165 / 86,961 | 1,906 / 1,709 | 94,071 / 88,568 | 32 / 31 | 2,672,786 |
| LangGraph | 22 / 24 | 0 / 0 | 92,345 / 87,440 | 1,892 / 1,766 | 94,237 / 89,253 | 33 / 32 | 2,677,994 |

Retries: for the Python arms, correction turns after an unparseable reply (traced as `*/retry`); byLLM's own correction calls are not tagged in its usage log and show up only in the call count.

## Per-claim verdicts

| # | hops | gold | Jac (byLLM) | OpenAI SDK | LangGraph | prompt tokens (jac/openai/langgraph) |
|---|---|---|---|---|---|---|
| 1 | 3 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 122k/86k/86k |
| 2 | 3 | NOT_SUPPORTED | ✗ NEED_MORE | ✓ NOT_SUPPORTED | FAILED | 84k/85k/4k |
| 3 | 2 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 142k/47k/41k |
| 4 | 2 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 175k/42k/43k |
| 5 | 3 | SUPPORTED | ✓ SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | 154k/81k/81k |
| 6 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 23k/20k/20k |
| 7 | 4 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 20k/33k/32k |
| 8 | 4 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 172k/91k/44k |
| 9 | 3 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 47k/28k/28k |
| 10 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 27k/51k/54k |
| 11 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 167k/182k/182k |
| 12 | 4 | SUPPORTED | ✗ NEED_MORE | ✓ SUPPORTED | ✗ NEED_MORE | 84k/54k/37k |
| 13 | 3 | SUPPORTED | ✓ SUPPORTED | ✗ NOT_SUPPORTED | ✗ NEED_MORE | 179k/97k/186k |
| 14 | 2 | NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 94k/99k/96k |
| 15 | 2 | NOT_SUPPORTED | ✗ NEED_MORE | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 96k/94k/125k |
| 16 | 3 | SUPPORTED | ✓ SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | 167k/86k/86k |
| 17 | 3 | NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 191k/170k/170k |
| 18 | 2 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 130k/160k/139k |
| 19 | 2 | SUPPORTED | ✗ NOT_SUPPORTED | ✗ NOT_SUPPORTED | ✗ NEED_MORE | 196k/159k/116k |
| 20 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 162k/98k/95k |
| 21 | 4 | SUPPORTED | ✓ SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | 104k/39k/39k |
| 22 | 2 | NOT_SUPPORTED | ✗ SUPPORTED | ✗ SUPPORTED | ✗ SUPPORTED | 182k/43k/91k |
| 23 | 2 | NOT_SUPPORTED | ✗ NEED_MORE | ✓ NOT_SUPPORTED | ✗ NEED_MORE | 109k/153k/151k |
| 24 | 3 | NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 176k/163k/160k |
| 25 | 4 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 160k/155k/156k |
| 26 | 4 | NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 166k/88k/119k |
| 27 | 2 | SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | ✗ NEED_MORE | 166k/160k/160k |
| 28 | 2 | SUPPORTED | ✓ SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | 128k/58k/87k |
| 29 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✗ NEED_MORE | ✗ NEED_MORE | 76k/36k/35k |

## Failed runs

- LangGraph #002 (exit 1): During task with name 'scout' and id '353ade61-2bad-fdac-d450-84e5bbdcb674'
