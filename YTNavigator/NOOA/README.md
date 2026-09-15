# YT-Navigator chat agent -- NOOA arm (NVIDIA Object-Oriented Agents)

`agent.py` does what `../byLLM/nodes.jac` does, written the way the
[NOOA docs](https://github.com/NVIDIA-NeMo/labs-OO-Agents/tree/main/docs)
say to write an agent: one Python class, `YTNavigator(Agent)`. An async method
ending in `...` is implemented by the LLM, a method with a real body is
ordinary Python, and the walk lives in a hidden deterministic method. There is
no graph, no walker and no node registry. `main.py` is byLLM's `main.jac`,
record for record.

| Jac (`nodes.jac`)                                        | NOOA (`agent.py`)                                                  | model calls |
|----------------------------------------------------------|--------------------------------------------------------------------|-------------|
| `Router` + `visit [-->] by llm(select=1, intent=...)`, candidates described by node `sem`s | `route(channel, conversation, message) -> Route` (Predict); the intent and the three node `sem` strings are the docstring, `Route` is an enum whose values are the benchmark labels `Yes` / `No` / `Not relevant` | 1 |
| `def direct_reply(...) -> AgentAnswer by llm`            | `direct_reply(...) -> AgentAnswer` (Predict)                        | 1 |
| `def tool_reply(...) by llm(tools=[...], max_react_iterations=6)` | `tool_reply(...) -> AgentAnswer` (CodeAct, `max_iterations=6`); the two tools are visible methods the generated code calls | up to 6 |
| `similarity_videos_search`, `execute_query` (`tools.jac`) | regular methods on the agent, each call logged for the record       |   |
| `StaticReply` node                                        | the constant `STATIC_REPLY`, no model call                          | 0 |
| `walker ChatAgent` (route -> reply, fallback events)      | `chat()`, hidden, deterministic; same `router_error_fallback` / `output_parse_fallback` events |   |
| `obj AgentAnswer / AnswerVideo / AnswerTimestamp` + `sem` | Pydantic models with `Field(description=...)`                       |   |
| `retrieval.py`                                            | loaded from `../byLLM/retrieval.py` by path (as `openai_sdk` does); one copy, one set of tables |   |

## What NOOA supplies that the other Python arm spells out

- **Prompts.** System prompt = class docstring + `doc(type(self))`: the visible
  methods with their docstrings and `Args:` sections (from the `Annotated`
  descriptions) and the referenced Pydantic types with field descriptions.
  Task message = method docstring + the arguments rendered as Python
  assignments. Nothing is interpolated by hand.
- **Structured output.** `PredictStrategy` builds a `response_format`
  json_schema from the return annotation, extracts JSON from the reply
  (fences stripped), validates with Pydantic and feeds the error back for one
  more attempt (`PredictConfig(max_retries=2)`). The router's enum return is
  validated the same way; a reply that never validates takes the walker's
  `router_error_fallback` to the direct reply.
- **The tool loop.** `CodeActStrategy` gives the model `execute_python` and
  `return_result`. The model writes Python that calls
  `self.similarity_videos_search(...)` and `self.execute_query(...)` (their
  signatures and docstrings are the tool schema; no JSON specs), sees the
  printed results, and finishes with `return_result(AgentAnswer(...))`, which
  is validated against the return type. `max_iterations=6` is byLLM's
  `max_react_iterations=6`. A run that ends without a valid answer (text-only
  turns, exhausted retries) takes `output_parse_fallback`, as on the Jac side.

## Settings that keep the arm comparable

- `event_query=EventQuery.current_call()`: NOOA keeps an event history per
  object; without this the router's second call would replay the first
  question's events. With it every call is stateless and the flattened
  "last 3 exchanges" conversation argument is the only carry-over, as in the
  other arms.
- `truncation=TruncationConfig(prefill_format=FormatConfig(max_string=None, ...))`:
  NOOA cuts rendered argument strings at 2000 chars by default.
- The client is NOOA's `CompletionClient` (litellm) with `temperature=0`,
  `drop_params=True`, `reasoning_effort=minimal` on gpt-5 / o-series; it is
  subclassed only to record `{"model", "latency_s", "prompt_tokens",
  "completion_tokens"}` per call, the `llm_calls` element of the shared
  record schema, sliced per question by `main.py`.
- Model: `$BENCH_MODEL` (bare id, default `glm-5.2`, prefixed `openai/` for
  litellm) on `$OPENAI_BASE_URL` (default `https://ollama.com/v1`) with
  `$OPENAI_API_KEY` -- the same three knobs as the other arms.

## Running

```bash
# interpreter: the nooa conda env (nooa 0.0.9, Python 3.12) with psycopg2-binary and
# sentence-transformers (CPU torch) installed
export OPENAI_API_KEY=... OPENAI_BASE_URL=https://ollama.com/v1 BENCH_MODEL=glm-5.2
export POSTGRES_HOST=localhost POSTGRES_PORT=5544 POSTGRES_USER=ytnav \
       POSTGRES_PASSWORD=ytnav_bench POSTGRES_DB=ytnav_bench      # the eval's docker DB
YTNAV_QUESTIONS=../datasets/questions.jsonl YTNAV_OUTPUT=results_nooa.jsonl \
    ~/miniconda3/envs/nooa/bin/python main.py

# or through the harness (DB, dataset and scoring handled for you)
python ../eval/e2e.py --impl nooa          # from eval/, with the ytnav env's Python
python ../eval/run.py --impl nooa          # DB already up; NOOA_PYTHON overrides the interpreter
```

Output: one record per question in the shared schema (`results_nooa.jsonl`,
`framework: "nooa"`), scored by `eval/score.py` unchanged.
