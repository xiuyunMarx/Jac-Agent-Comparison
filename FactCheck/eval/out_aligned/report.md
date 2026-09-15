# FactCheck 3-way comparison (unknown model, HoVer dev claims)

29 claims per arm; knobs: FC_SCOUTS=2, FC_ROUNDS=FC_MIN_ROUNDS=6 (defaults). NEED_MORE = round cap reached without a decision, scored as wrong.

## Task performance

| arm | runs | failed | accuracy | decided | acc. on decided | abstain (NEED_MORE) | SUPPORTED acc. | NOT_SUPPORTED acc. |
|---|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 29 | 0 | **86.2%** (25/29) | 29 | 86.2% | 0.0% | 81.2% | 92.3% |
| OpenAI SDK | 29 | 0 | **79.3%** (23/29) | 27 | 85.2% | 6.9% | 81.2% | 76.9% |
| LangGraph | 29 | 0 | **79.3%** (23/29) | 28 | 82.1% | 3.4% | 75.0% | 84.6% |
| nooa | 29 | 0 | **82.8%** (24/29) | 29 | 82.8% | 0.0% | 81.2% | 84.6% |

### Accuracy by hop count

| arm | 2 hops | 3 hops | 4 hops |
|---|---|---|---|
| Jac (byLLM) | 70.0% (7/10) | 100.0% (9/9) | 90.0% (9/10) |
| OpenAI SDK | 60.0% (6/10) | 88.9% (8/9) | 90.0% (9/10) |
| LangGraph | 60.0% (6/10) | 88.9% (8/9) | 90.0% (9/10) |
| nooa | 60.0% (6/10) | 100.0% (9/9) | 90.0% (9/10) |

### Verdict distribution

| arm | SUPPORTED | NOT_SUPPORTED | NEED_MORE | none (failed) |
|---|---|---|---|---|
| Jac (byLLM) | 14 | 15 | 0 | 0 |
| OpenAI SDK | 15 | 12 | 2 | 0 |
| LangGraph | 14 | 14 | 1 | 0 |
| nooa | 15 | 14 | 0 | 0 |

## Token usage (per claim, mean / median)

| arm | model calls | retries | prompt tokens | completion tokens | total tokens | wall s | total prompt tokens (all claims) |
|---|---|---|---|---|---|---|---|
| Jac (byLLM) | 19 / 16 | 0 / 0 | 79,574 / 57,629 | 11,409 / 9,452 | 90,983 / 68,711 | 138 / 112 | 2,307,641 |
| OpenAI SDK | 21 / 20 | 0 / 0 | 97,054 / 91,167 | 8,432 / 7,312 | 105,486 / 98,479 | 99 / 90 | 2,814,567 |
| LangGraph | 21 / 22 | 0 / 0 | 97,062 / 92,412 | 7,855 / 6,863 | 104,917 / 99,345 | 102 / 102 | 2,814,795 |
| nooa | 18 / 16 | 0 / 0 | 86,738 / 72,765 | 11,632 / 11,256 | 98,370 / 78,089 | 135 / 119 | 2,515,413 |

Retries: for the Python arms, correction turns after an unparseable reply (traced as `*/retry`); byLLM's own correction calls are not tagged in its usage log and show up only in the call count.

## Per-claim verdicts

| # | hops | gold | Jac (byLLM) | OpenAI SDK | LangGraph | nooa | prompt tokens (jac/openai/langgraph/nooa) |
|---|---|---|---|---|---|---|---|
| 1 | 3 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 57k/87k/109k/76k |
| 2 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 39k/90k/91k/48k |
| 3 | 2 | SUPPORTED | ✗ NOT_SUPPORTED | ✗ NOT_SUPPORTED | ✗ NOT_SUPPORTED | ✗ NOT_SUPPORTED | 178k/154k/132k/190k |
| 4 | 2 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 30k/30k/30k/43k |
| 5 | 3 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 56k/53k/54k/44k |
| 6 | 4 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 121k/93k/99k/114k |
| 7 | 4 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 51k/52k/45k/54k |
| 8 | 4 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 29k/168k/147k/64k |
| 9 | 3 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 27k/69k/56k/40k |
| 10 | 4 | NOT_SUPPORTED | ✗ SUPPORTED | ✗ SUPPORTED | ✗ SUPPORTED | ✗ SUPPORTED | 98k/63k/48k/77k |
| 11 | 4 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 174k/167k/176k/158k |
| 12 | 4 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 44k/39k/46k/55k |
| 13 | 3 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✗ NEED_MORE | ✓ SUPPORTED | 77k/137k/107k/75k |
| 14 | 2 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 125k/106k/92k/116k |
| 15 | 2 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 30k/91k/154k/66k |
| 16 | 3 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 37k/98k/135k/68k |
| 17 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✗ NEED_MORE | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 67k/108k/89k/80k |
| 18 | 2 | SUPPORTED | ✗ NOT_SUPPORTED | ✗ NOT_SUPPORTED | ✗ NOT_SUPPORTED | ✗ NOT_SUPPORTED | 151k/146k/135k/171k |
| 19 | 2 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 35k/88k/88k/50k |
| 20 | 4 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 103k/106k/100k/123k |
| 21 | 4 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 126k/84k/82k/63k |
| 22 | 2 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✗ SUPPORTED | ✗ SUPPORTED | ✗ SUPPORTED | 31k/64k/31k/45k |
| 23 | 2 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 33k/58k/57k/72k |
| 24 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 28k/85k/88k/46k |
| 25 | 4 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 164k/124k/107k/118k |
| 26 | 4 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 174k/113k/154k/81k |
| 27 | 2 | SUPPORTED | ✗ NOT_SUPPORTED | ✗ NEED_MORE | ✗ NOT_SUPPORTED | ✗ NOT_SUPPORTED | 64k/146k/152k/170k |
| 28 | 2 | SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | ✓ SUPPORTED | 31k/83k/112k/60k |
| 29 | 3 | NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | ✓ NOT_SUPPORTED | 112k/99k/86k/133k |
