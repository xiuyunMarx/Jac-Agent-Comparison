# cache_bench -- how much does byLLM's invariant prompt hoisting help prefix caching?

Measures, for every byLLM agent in this repo, the provider prefix-cache reuse
with `[byllm.optimizations] invariant_prompt_hoisting` off versus on
(jaseci-labs/jac PR #8922). No caching provider is needed: the agents run on
whatever model drives them (GLM on ollama.com by default), the runtime records
every request exactly as sent, and `score.py` replays the requests through a
simulator of the provider cache rules.

## Pieces

| File | Role |
|------|------|
| `score.py` | Offline scorer. Reads `BYLLM_PROMPT_LOG` files and reports cache reads / writes / cost per run under two accountings (below). |
| `run_all.sh` | Drives each arm twice (baseline, hoisted), N sessions each, collecting the logs under `results/<run>/`. |
| `results/` | Logs and the summary table (untracked). |

The request log comes from a small addition to the runtime on the local branch
`pr-8922-bench` of `~/jac-upstream`: when `BYLLM_PROMPT_LOG=<file>` is set,
`log_completion` appends the request (`messages`, `tools`, `response_format`,
with the `cache_control` markers byLLM placed) next to the usage record that
`BYLLM_USAGE_LOG` already writes. Nothing else in the compiler changed.

## Accounting

`score.py` models an SGLang-style radix cache: the server keeps every prompt it
served as a token trie; a new request reuses the longest token prefix it shares
with any earlier prompt and prefills only the tail. No markers, no minimum
size. The prompt is rendered in chat-template order (tools, system messages,
conversation). Reported per run:

* **hit rate** -- cached tokens / prompt tokens (SGLang's `cached_tokens`).
* **prefill** -- tokens that still had to be computed.
* **invariant** -- the PR's own metric: hoisted-prefix tokens / prompt tokens.

Each log file is one session with a cold cache; `--shared` pools a run's
sessions into one cache in timestamp order (one server serving every worker);
`--ttl` evicts idle entries. Token ids come from litellm's tokenizer for the
logged model (chars/4 pseudo-tokens if litellm is missing); the provider's own
count is printed alongside so the error is visible (about +14% on GLM).

## Caveats

* Trajectories differ between the two modes because the model sees different
  prompts, so compare pooled sessions, not single runs.
* Parallel sessions of the same agent share one provider cache in reality; the
  scorer scores each log file (one session) independently, which is the
  conservative reading.
