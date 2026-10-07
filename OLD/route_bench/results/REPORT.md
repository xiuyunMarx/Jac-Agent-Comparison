# Learnt model routing inside byLLM -- Email arm, 2026-09-08

Setup: byLLM runtime (jac PR #8922 line + `routing-bench` branch) asks LLMRouter's KNN router, over HTTP,
which of {gemma4:31b, glm-5.2} on ollama.com should serve each `by llm` call. Agent code unchanged; routing is
switched on with three environment variables (or a `[byllm.routing]` table). Router trained by replaying the
strong model's logged requests on both candidates and labelling with the call's own type contract
(valid JSON for the declared return type / well-formed call to a declared tool, agreement with glm on the typed
skeleton, minus a cost term). Two cross splits so the router never sees its evaluation batches.
Prices: glm-5.2 $0.42/$1.32, gemma4:31b $0.09/$0.34 per 1M tokens (OpenRouter list). Judge: glm-5.2.

Headline: routed keeps the strong arm's task metrics on both splits at 24% (split B) to 80% (split A) lower cost.
The cheap model alone is *more* expensive than the strong one: gemma loops on web_search at the tool sites
(132 and 224 calls vs 46 and 55), which is exactly what the router learns to avoid.

Split B caveat: the router sent every tool-site call to glm-5.2, so strong and routed produced drafts with the same
model; the judge gap there (overall 4.12 vs 3.50 on 8 drafts) is run-to-run and judge noise, not a routing effect.

## Split A: router trained on batches 1-3, evaluated on 4-6

Router: LLMRouter KNNRouter (k=5, Longformer embeddings), trained on 44 requests replayed on gemma4:31b, glm-5.2; held-out oracle accuracy 64% on 11 requests (lambda=0.2).

| mode | runs | expected replies found (tp/fp/fn) | filtering P / R / F1 | drafts done / correct recipient | LLM calls | prompt tok | completion tok | cost USD |
|---|---:|---|---|---|---:|---:|---:|---:|
| strong | 3 | 5/0/0 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 | 46 | 36,054 | 5,508 | 0.0224 |
| cheap | 3 | 5/0/0 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 | 132 | 226,925 | 20,495 | 0.0274 |
| routed | 3 | 5/0/0 | 1.00 / 1.00 / 1.00 | 1.00 / 1.00 | 26 | 14,637 | 2,514 | 0.0044 |

LLM judge (glm-5.2 via ollama.com; 1-5 scales, key-point coverage as a fraction):

| mode | drafts judged | key points covered | tone | factuality | overall |
|---|---:|---:|---:|---:|---:|
| strong | 5 | 0.93 | 5.00 | 4.20 | 4.40 |
| cheap | 5 | 0.87 | 5.00 | 5.00 | 4.60 |
| routed | 5 | 0.93 | 5.00 | 5.00 | 4.60 |

## Where each call site went (calls per model)

**strong**
- email_action_agent: glm-5.2 x14
- email_response_writer: glm-5.2 x17
- filter_emails: glm-5.2 x15

**cheap**
- email_action_agent: gemma4:31b x60
- email_response_writer: gemma4:31b x61
- filter_emails: gemma4:31b x11

**routed**
- email_action_agent: glm-5.2 x6, gemma4:31b x4
- email_response_writer: gemma4:31b x5
- filter_emails: gemma4:31b x11

## Split B: router trained on batches 4-6, evaluated on 1-3

Router: LLMRouter KNNRouter (k=5, Longformer embeddings), trained on 37 requests replayed on gemma4:31b, glm-5.2; held-out oracle accuracy 67% on 9 requests (lambda=0.2).

| mode | runs | expected replies found (tp/fp/fn) | filtering P / R / F1 | drafts done / correct recipient | LLM calls | prompt tok | completion tok | cost USD |
|---|---:|---|---|---|---:|---:|---:|---:|
| strong | 3 | 8/1/0 | 0.89 / 1.00 / 0.94 | 1.00 / 1.00 | 55 | 35,834 | 8,824 | 0.0267 |
| cheap | 3 | 8/1/0 | 0.89 / 1.00 / 0.94 | 1.00 / 1.00 | 224 | 379,088 | 32,843 | 0.0453 |
| routed | 3 | 8/1/0 | 0.89 / 1.00 / 0.94 | 1.00 / 1.00 | 46 | 30,146 | 7,266 | 0.0203 |

LLM judge (glm-5.2 via ollama.com; 1-5 scales, key-point coverage as a fraction):

| mode | drafts judged | key points covered | tone | factuality | overall |
|---|---:|---:|---:|---:|---:|
| strong | 8 | 0.91 | 5.00 | 3.62 | 4.12 |
| cheap | 8 | 0.78 | 4.88 | 3.88 | 3.88 |
| routed | 8 | 0.96 | 5.00 | 3.00 | 3.50 |

## Where each call site went (calls per model)

**strong**
- email_action_agent: glm-5.2 x17
- email_response_writer: glm-5.2 x14
- filter_emails: glm-5.2 x24

**cheap**

**routed**
- email_action_agent: glm-5.2 x15
- email_response_writer: glm-5.2 x14
- filter_emails: gemma4:31b x17
