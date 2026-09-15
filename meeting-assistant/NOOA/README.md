

## byLLM -> NOOA

| byLLM (Jac)                                            | NOOA                                                                        |
|--------------------------------------------------------|-----------------------------------------------------------------------------|
| `obj MeetingTask` + `sem MeetingTask.*`                | Pydantic `MeetingTask` with `Field(description=...)` carrying the sems verbatim |
| `node GeneratingTasks`                                 | `class GeneratingTasks(Agent, llm=llm, ...)`                                |
| `def analyse_meeting_transcript(transcript) -> list[MeetingTask] by llm(max_output_retries=3)` | `async def analyse_meeting_transcript(transcript) -> list[MeetingTask]: ...` with `@strategy(PredictStrategy(config=PredictConfig(max_retries=3)))` |
| `sem GeneratingTasks.analyse_meeting_transcript`       | the method docstring, verbatim                                              |
| `sem ...analyse_meeting_transcript.transcript`         | `transcript: Annotated[str, "..."]`, rendered into the `Args:` section      |
| `def repair_meeting_tasks(...) by llm(max_output_retries=3)` + sems | `async def repair_meeting_tasks(...) -> list[MeetingTask]: ...`, same treatment |
| walker `generate_task`: extract, `_tasks_are_valid`, repair once, raise | `@hidden async def extract()` on the agent: the deterministic evidence gate |
| `node AddTask` / `Save2CSV` / `SendNotification`       | one-method plain classes (no model call, so no `Agent`)                     |
| `walker MeetingAssistant`, `visit [-->]`               | `class MeetingAssistant`, `run()`: `extract`, then the three nodes in insertion order |
| `Model(model_name="openai/<BENCH_MODEL>", ...)`        | `CompletionClient(model="openai/<BENCH_MODEL>", api_base=$OPENAI_BASE_URL, ...)` |
| litellm `success_callback` + `settle()`                | `TracedClient.acall` feeds `token_usage` from `LLMResponse.usage` at the call site |

One model call per run, two if the first list has an empty field (the
repair call), plus NOOA's own validation retries inside a call when a reply
does not validate against `list[MeetingTask]` (at most three attempts per
call, byLLM's `max_output_retries=3`).

## What NOOA supplies

- **The prompt.** System prompt = the class docstring (one sentence; the Jac
  node has no `sem` of its own, and byLLM supplies its own generic framing
  there) + `doc(type(self))`: the class rendered with its visible methods,
  their docstrings and `Args:` sections, and the referenced `MeetingTask`
  type with its field descriptions. Task message = the method docstring +
  `transcript = '...'` rendered as a Python assignment. Nothing is written
  by hand.
- **The output contract.** `PredictStrategy` sends a `response_format`
  json_schema built from `list[MeetingTask]` (wrapped in a one-key object,
  as byLLM and the openai_sdk arm also do), extracts JSON from the reply
  (fences stripped), validates with Pydantic and feeds the error back for
  another attempt. ollama.com ignores `response_format` for GLM, so on the
  wire this is the same shape-in-prompt + parse + retry protocol as the
  Python arms.
- **The gate.** `extract()` is ordinary Python: validity is checked by the
  program, the repair prompt is issued by the program, and the failure is a
  raised `ValueError`, exactly as on the Jac side.

## Settings that keep the arm comparable

- `PredictStrategy` on both `...` methods. NOOA's default is CodeAct (the
  model writes Python in a REPL); the Jac side hands the model no tools, so
  Predict is the matching call shape.
- `event_query=EventQuery.current_call()`: NOOA keeps an event history per
  object and would otherwise replay the first extraction into the repair
  call's prompt. With this, each call sees only its own events, as byLLM's
  stateless `by llm` calls do.
- `truncation=TruncationConfig(prefill_format=FormatConfig(max_string=None, ...))`:
  NOOA cuts rendered argument strings at 2000 chars by default; transcripts
  are longer.
- `temperature=0.7`, no `max_tokens` (byLLM's defaults, what the other arms
  send); `reasoning_effort="minimal"` and no temperature on gpt-5 / o-series;
  `drop_params=True` so litellm drops what a model rejects.

## Running

```bash
set -a; source ../../glm.env; set +a      # BENCH_MODEL, OPENAI_BASE_URL, OPENAI_API_KEY
cp ../datasets/meeting_004.txt meeting_notes.txt
~/miniconda3/envs/nooa/bin/python main.py
```

Through the harness, with the other arms:

```bash
cd ../eval
python run.py --impl NOOA --cases meeting_004
python score.py runs/ --judge
```

`run.py` runs `NOOA/main.py` with `$NOOA_PYTHON`, else the conda env
`nooa` (`~/miniconda3/envs/nooa/bin/python`, nooa 0.0.9 on Python 3.12),
else its own interpreter, and exits early if `nooa` is not importable there.

## Layout

| file       | what it holds                                                                   |
|------------|---------------------------------------------------------------------------------|
| `main.py`  | the entry point: read, run, print, dump; byLLM's `main.jac` line for line       |
| `nodes.py` | `MeetingTask`, the client, `GeneratingTasks(Agent)`, the fan-out nodes, `MeetingAssistant` |
| `tools.py` | mock Trello / Slack / CSV collectors, `TokenUsage`, `dump_outputs`               |
