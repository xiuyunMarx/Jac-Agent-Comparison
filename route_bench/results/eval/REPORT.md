# Learnt model routing on the Email byLLM arm

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
