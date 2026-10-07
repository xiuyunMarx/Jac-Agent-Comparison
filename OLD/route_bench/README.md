# route_bench -- learnt model routing inside the byLLM runtime

Shows that a Jac agent gets per-call model routing with **no change to agent
code**: the byLLM runtime asks a learnt router (LLMRouter's KNN router) which
candidate model should serve each `by llm` call, and the type contract of the
call site both enriches the router's query and guards its cheaper choices.
Measured on the Email byLLM arm: task quality (filtering P/R/F1, draft
completion, correct recipient) and cost, strong-only vs cheap-only vs routed.

## Runtime change (jaseci-labs/jac, branch `routing-bench` in ~/jac-upstream)

`[byllm.routing]` in jac.toml (or `BYLLM_ROUTING=1`, `BYLLM_ROUTING_ENDPOINT`,
`BYLLM_ROUTING_CANDIDATES`):

```toml
[byllm.routing]
enabled = true
endpoint = "http://127.0.0.1:8765/route"
candidates = ["openai/gemma4:31b", "openai/glm-5.2"]   # cheapest first
```

* `jaclang/byllm/routing.jac`: builds the router query = call-site header
  (`site`, declared `returns` type, declared `tools`, all from the compiler's
  MTIR) + the request messages; POSTs it to the router; falls back to the
  configured model on any error; caches identical queries.
* `make_model_params` swaps `params["model"]` for the router's choice, so every
  dispatch path (sync, streaming, async, tools) is routed.
* `_typed_retry_reset` sets `_route_escalate`: a typed-output retry always goes
  to the configured (strong) model. The type check is the guard.
* usage/prompt logs carry the served model, `route_reason` and `route_query`,
  so any run's log is router training data.
* `tests/test_routing.jac` (7 tests, MockLLM-level).

## Protocol

Two cross splits so every batch is evaluated by a router that never saw it:

* split A: router trained on batches 1-3 (`collect`), evaluated on 4-6 (`results/eval`, `results/router`)
* split B: router trained on the strong run's logs from 4-6, evaluated on 1-3 (`results/eval_b123`, `results/router_b456`);
  the strong-only arm of split B is the `collect` run itself.

Modes per split: strong-only (glm-5.2), cheap-only (gemma4:31b), routed. The judge step
(`run.sh judge`) scores drafts with glm-5.2 as LLM judge through the arm's own scorer.

## Pipeline

```
set -a; . ./.env; set +a
route_bench/run.sh collect   # glm-5.2 on batches 1-3, logs every request + route_query
route_bench/run.sh train     # replay each request on both candidates, label, train KNN
route_bench/run.sh eval      # strong / cheap / routed on batches 4-6
route_bench/run.sh judge     # LLM-judge draft quality (optional)
route_bench/run.sh report    # <EVAL_DIR>/REPORT.md
# split B: TRAIN_PROMPTS=results/eval/strong/prompts.jsonl ROUTER_DIR=results/router_b456 \
#          EVAL_DIR=results/eval_b123 EVAL_BATCHES="batch_001 batch_002 batch_003" run.sh train / eval cheap routed / judge / report
```

`build_router.py` labels without human ground truth: a candidate's reply is
*valid* if it satisfies the call's declared contract (JSON schema of the return
type, or a well-formed call to a declared tool) and *agrees* if its typed
skeleton (enums, numbers, bools, short strings, tool name) matches the strong
model's reply. KNN label = valid*agree - lambda*cost/max_cost(query), so among
contract-satisfying candidates the cheaper wins. Router = LLMRouter
`KNNRouter` + `KNNRouterTrainer` (Longformer embeddings, cosine, k=5) served by
`router_server.py` from `~/.venvs/llmrouter` (CPU only).

Prices (`pricing.json`, USD per 1M tokens, OpenRouter list 2026-09-08):
glm-5.2 0.42 in / 1.32 out; gemma4:31b 0.09 in / 0.34 out.

## Other byLLM arms (`run_arm.sh`)

`route_bench/run_arm.sh <arm> collect|train|eval|score|all` runs the same protocol on a few cases of
meeting (train meeting_001-005, eval 006-010), factcheck (train claims 1-3, eval 4-6), ytnav (train q1-6,
eval q7-12). Results under `results/<arm>/{collect,router,eval/<mode>}`; `report_arms.py` writes `results/REPORT_arms.md`.
`FactCheck/Jac/fact_check.jac` now reads `FC_MODEL` so the cheap model can be selected.

Label detail: byLLM wraps non-object return types in `{"schema_object_wrapper": ...}` but accepts the bare
value; `build_router.py` unwraps the same way before validating (gemma answers bare lists). Runs trained
before that fix were retrained with `train --skip-replay` (FactCheck `eval/routed_v1` is the pre-fix run).
