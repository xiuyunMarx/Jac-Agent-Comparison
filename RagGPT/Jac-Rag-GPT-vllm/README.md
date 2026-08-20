# Jac-Rag-GPT-vllm

`Jac-Rag-GPT` with a fully local backend: the LLM is Qwen2.5-3B-Instruct served
by vLLM on `:8001` instead of gpt-4.1-mini. Embeddings (bge-small) and the
CrossEncoder reranker were already local, so this version runs with no cloud
dependency. Prompts, sems, and per-agent params are byte-identical to
`Jac-Rag-GPT`; `docs/` and `faiss_index/` are symlinks into it.

Purpose: latency profiling for the proactive-prefill research — vLLM's
`/metrics` endpoint exposes per-request prefill/decode/queue time that no
cloud API reveals. See `eval/harness/profile_vllm.py`.

## Differences vs Jac-Rag-GPT (all in `main.jac` / config)

1. `config/faiss_reranking.json` → `model_name: hosted_vllm/Qwen/Qwen2.5-3B-Instruct`.
2. `main.jac` sets `HOSTED_VLLM_API_BASE` / `HOSTED_VLLM_API_KEY` env defaults.
   The env var is the only channel that works: byllm's litellm path pops
   `api_base` from the call params without forwarding it
   (`jaclang/byllm/llm.impl/model.impl.jac`, `litellm.completion` call), so
   `Model(config={"base_url": ...})` and jac.toml `base_url` are both ignored.
   jac.toml `[plugins.byllm.model]` is additionally not loaded at all by the
   current dev build (defaults come back — see `config_loader`).
3. `Model(ctx_window=16384)` — litellm has no metadata for `hosted_vllm/*`
   models; without it byllm auto-compaction misreads the context window and
   fails with "Compaction produced no reduction" on small histories. Keep in
   sync with `serve_vllm.sh --max-model-len`.

## Run

```bash
./serve_vllm.sh          # vLLM on :8001 (conda env spork; checks for leaked VRAM)
jac run main.jac         # terminal chat
jac start main.jac -p 8010   # REST, same walkers as Jac-Rag-GPT
```

Profiling (from `eval/harness`):

```bash
/home/xiaoyu/miniconda3/envs/jaseci/bin/python profile_vllm.py --limit 30 --fresh
# -> eval/results/vllm_profile/{turns.jsonl,report.md}
```
