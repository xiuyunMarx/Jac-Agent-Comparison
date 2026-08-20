# Jac-Rag-GPT on vLLM (Qwen2.5-3B) — prefill share profile

16 turns ok, 0 errored. Server-side times from vLLM /metrics deltas per turn; client e2e includes routing, RAG search, reranking, framework.

| scope | turns | calls/turn | prefill s | decode s | client e2e s | prefill/LLM | prefill/e2e | LLM/e2e | KV computed/prompt tok | prefix-cache hit |
|---|---|---|---|---|---|---|---|---|---|---|
| **overall** | 16 | 2.4 | 2.84 | 165.59 | 201.43 | 1.7% | 1.4% | 83.6% | 50.7% | 49.3% |
| coding | 3 | 2.0 | 0.20 | 15.26 | 32.48 | 1.3% | 0.6% | 47.6% | 40.9% | 59.1% |
| debugging | 2 | 2.0 | 0.18 | 8.04 | 8.28 | 2.2% | 2.2% | 99.4% | 58.6% | 41.4% |
| multi_turn | 4 | 2.2 | 0.94 | 13.59 | 20.31 | 6.5% | 4.6% | 71.5% | 68.7% | 31.3% |
| off_topic | 2 | 2.5 | 0.26 | 3.76 | 8.59 | 6.6% | 3.1% | 46.9% | 53.3% | 46.7% |
| rag_qa | 3 | 3.0 | 0.51 | 4.91 | 7.25 | 9.5% | 7.1% | 74.8% | 41.6% | 58.4% |
| small_talk | 2 | 3.0 | 0.74 | 120.03 | 124.54 | 0.6% | 0.6% | 97.0% | 42.6% | 57.4% |

- `prefill/LLM`: prefill time over prefill+decode (pure LLM view).
- `prefill/e2e`: prefill time over client wall time — the share proactive prefill can attack end-to-end.
- `KV computed/prompt tok`: <100% means vLLM's reactive prefix cache already skipped some prefill; proactive prefill targets the rest.
