# Email auto-response agent -- NOOA arm (NVIDIA Object-Oriented Agents)

`main.py` does what [`../byLLM/nodes.jac`](../byLLM/nodes.jac) does, written
the way the [NOOA docs](https://github.com/NVIDIA-NeMo/labs-OO-Agents/tree/main/docs)
say to write an agent: one Python class whose `...` methods are implemented
by the LLM, whose regular methods are its tools and mechanics, and whose
hidden `run()` is the workflow. No graph, no walker.

| byLLM (Jac)                                                    | NOOA                                                                 | model calls |
|----------------------------------------------------------------|----------------------------------------------------------------------|-------------|
| `node check_new_emails.fetch_mail_abstracts`                   | `EmailAgent.fetch_mail_abstracts()`, hidden, deterministic            | 0           |
| `def filter_emails(abstract) -> Classification by llm()`       | `filter_emails(abstract) -> Classification: ...` with `PredictStrategy` | 1 per thread |
| `def email_action_agent(thread) by llm(tools=[web_search], max_react_iterations=10)` | `email_action_agent(thread) -> ThreadAnalysis: ...` with `CodeActStrategy(max_iterations=10)` | 1 + tool turns |
| `def email_response_writer(analysis, thread) by llm(tools=[web_search], ...)` | `email_response_writer(analysis, thread) -> DraftReply: ...`, same strategy | 1 + tool turns |
| `def web_search(query) -> str` + `sem web_search`              | `EmailAgent.web_search()`, a public method: NOOA's tool layer         |             |
| `tools.jac` `MailBox.get_thread` -> `Thread`                   | `EmailAgent.get_thread()`, hidden; `Thread`/`Email` Pydantic models   |             |
| `obj ThreadAnalysis` / `obj DraftReply` + `sem` fields         | Pydantic models with `Field(description=...)`                        |             |
| `enum Classification`                                          | `class Classification(str, Enum)`                                    |             |
| `walker EmailAgent` abilities `check` / `draft`                | `EmailAgent.run()`, hidden: the same skip / record / one-retry loop   |             |
| `with entry` block                                             | `main()`: same CLI, same prints, same `save_results` file             |             |

## How NOOA runs the tool stages

byllm's `by llm(tools=[web_search])` is a finish-tool ReAct loop. NOOA's
equivalent is `CodeActStrategy`: the model gets two tools, `execute_python`
and `return_result`; the thread arrives as a live Python variable, the agent's
public API (here only `web_search`) is callable as `self.web_search(...)`
from a code cell, and the stage ends when `return_result(...)` validates
against `ThreadAnalysis` / `DraftReply`. `CodeActConfig(max_iterations=10)`
is byLLM's `max_react_iterations=10`. The three stage methods are `@hidden`
so `doc(self)` lists only `web_search` -- the LLM-facing action space stays
byLLM's -- while the stage's own docstring still carries its prompt.

`filter_emails` has no tools, so it is `PredictStrategy`: one structured call,
validated against the enum, with `PredictConfig(max_retries=4)` matching
byllm's four attempts (`max_output_retries=3`).

## Settings that keep the arm comparable

- `event_query=EventQuery.current_call()`: NOOA keeps an event history per
  object and would otherwise replay every earlier stage into later prompts.
  With this each stage sees only its own call, as byllm's stateless calls do.
- `TruncationConfig(prefill_format=FormatConfig(max_string=None, ...))`:
  arguments (threads, analyses) are rendered whole.
- Temperature 0.7, no `max_tokens`: byllm's defaults, matched as in the
  openai_sdk arm. `drop_params=True` lets litellm drop the temperature that
  gpt-5 / o-series reject; `reasoning_effort=minimal` is set for those.
- Model/endpoint/key: `$BENCH_MODEL` (bare id, default `glm-5.2`, sent as
  litellm `openai/<id>`), `$OPENAI_BASE_URL` (default `https://ollama.com/v1`),
  `$OPENAI_API_KEY` -- what nodes.jac reads.
- Token accounting is the harness's: `MockMailbox` patches the openai SDK,
  NOOA calls litellm, litellm calls that SDK, so `usage` in
  `results_<case_id>.json` is collected identically to the other arms and no
  agent code counts tokens. `BENCH_TRACE=<file.jsonl>` optionally adds one
  row per model call (stage, messages, reply, tool-call names, usage).
- Failure handling is the walker's: an analyzer that exhausts NOOA's retries
  raises, is recorded with `record_draft_error` and the email skipped; the
  writer gets one retry first; a crash still writes the results file, then
  exits non-zero.

## Running

```bash
cd Email-Auto-response
set -a; source ../glm.env; set +a; export OPENAI_API_KEY="$OLLAMA_API_KEY"   # see eval notes
~/miniconda3/envs/nooa/bin/python NOOA/main.py                      # batch_001
~/miniconda3/envs/nooa/bin/python NOOA/main.py mock_mailbox/datasets/batch_003.json
python eval/run.py --impl NOOA --batches batch_001                  # via the harness
```

Results: `NOOA/mock_output/results_<case_id>.json` (`$EMAIL_OUTPUT_DIR`
overrides the directory), scored by `eval/score.py` like every other arm.
Interpreter: `~/miniconda3/envs/nooa/bin/python` (nooa 0.0.9, Python 3.12);
the harness picks it up as the `NOOA` implementation (`$NOOA_PYTHON` overrides).
