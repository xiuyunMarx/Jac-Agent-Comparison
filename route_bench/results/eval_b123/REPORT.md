# Learnt model routing on the Email byLLM arm

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
