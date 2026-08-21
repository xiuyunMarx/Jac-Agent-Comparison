# Meeting assistant — no framework, just the OpenAI SDK

The third implementation of one agent. The others are [`../byLLM`](../byLLM)
(Jac + byLLM) and [`../CrewAI`](../CrewAI) (Python + CrewAI Flows). All three
are scored by the shared harness in [`../eval`](../eval) against the labeled
cases in [`../datasets`](../datasets), so they must present an **identical
action space**: the same three mock tools with the same string contracts
(`create_trello_card`, `send_message_to_channel`, `save_tasks_to_csv`), the
same input file (`meeting_notes.txt` in the working directory), and the same
single output artifact (`tool_outputs.json`, with `trello` / `slack` / `csv`
/ `token_usage`). The tool module is therefore a near-literal port, and the
whole of the interesting difference is in how the one LLM call and the
fan-out are expressed.

This side is the prompt-engineering baseline the other two are measured
against: what a framework is actually buying over one hand-written prompt and
one `client.chat.completions.create`.

**Fidelity target is `byLLM`, not `CrewAI`.** Where the two existing sides
have drifted apart, this one follows the Jac original — see
[Where CrewAI and byLLM disagree](#where-crewai-and-byllm-disagree).

## Running it

```bash
pip install openai            # the only dependency
export OPENAI_API_KEY=...     # read natively by the openai SDK

cp ../datasets/meeting_004.txt meeting_notes.txt
python main.py
```

The run leaves behind `tool_outputs.json` (what the eval scores), plus
`new_tasks.csv` (the CSV fan-out writes it on every side). Through the
harness, with the other two:

```bash
cd ../eval
python run.py --impl openai_sdk --cases meeting_004
python score.py runs/ --judge
```

## The pipeline

One LLM call, then a deterministic fan-out — the same four nodes byLLM walks
and CrewAI's Flow listens on:

```
  meeting_notes.txt
        │
        ▼
  ┌─────────────────┐   the one gpt-4o call: transcript in,
  │ GeneratingTasks │   strict-schema list[MeetingTask] out
  └───────┬─────────┘
          ├──────────────┬───────────────────┐
          ▼              ▼                   ▼
     ┌─────────┐    ┌──────────┐    ┌──────────────────┐
     │ AddTask │    │ Save2CSV │    │ SendNotification │
     └─────────┘    └──────────┘    └──────────────────┘
      mock Trello    new_tasks.csv    one mock Slack line
      (well-formed   (every task,     ("N New tasks have
      tasks only)    malformed too)   been added to Trello!")
```

The fan-out order is byLLM's insertion order (AddTask, Save2CSV,
SendNotification), and the asymmetry is deliberate on all three sides: Trello
skips a task missing a name or description, the CSV records every extracted
task, and the Slack count is the full task count — which is exactly the
disagreement `pipeline_consistent` in the eval exists to catch.

## Layout

| file | what it holds | lines |
| --- | --- | ---: |
| `main.py` | the entry point: read, run, print, dump — byLLM's `main.jac` line for line | 27 |
| `nodes.py` | the pipeline: `MeetingTask`, the prompts, the schema, the one call, the fan-out nodes | 227 |
| `tools.py` | mock Trello / Slack / CSV collectors, `TokenUsage`, `dump_outputs` | 108 |

The basenames are byLLM's (`main.jac` / `nodes.jac` / `tools.jac`) so the
trees diff cleanly; CrewAI spreads the same content over `main.py`,
`crews/…/meeting_assistant_crew.py` + two YAML files, and `utils/`.

## byLLM → CrewAI → no framework

| byLLM (Jac) | CrewAI | this |
| --- | --- | --- |
| `obj MeetingTask` + `sem` field strings | pydantic `MeetingTask` in `types.py` | `@dataclass MeetingTask`, sems moved into the prompt and schema |
| `def analyse_meeting_transcript(...) -> list[MeetingTask] by llm()` | `Agent` + `Task` + `output_pydantic=MeetingTaskList` | `GeneratingTasks.analyse_meeting_transcript`, one `chat.completions.create` |
| `sem GeneratingTasks.analyse_meeting_transcript` | `agents.yaml` role/goal/backstory + `tasks.yaml` description/expected_output | `SYSTEM_PROMPT` / `USER_PROMPT`, hand-written |
| schema derived by `type_to_schema` from the return type | instructor-driven converter call | `RESPONSE_FORMAT`, an explicit strict `json_schema` |
| walker `MeetingAssistant`, `visit [-->]` | `Flow` + `@start` / `@listen` | `MeetingAssistant.run()`, a `for` loop |
| litellm `success_callback` + `settle()` | class-level patch of `SyncAPIClient.post` / `AsyncAPIClient.post` | `response.usage`, read at the call site |
| `Model(model_name="openai/gpt-4o")` | `LLM(model="gpt-4o")` | `MODEL = "gpt-4o"` |

### What the framework was doing, and what replaced it

**The structured extraction.** byLLM gets the prompt, the JSON schema, the
request and the parse from one `by llm()` clause; CrewAI gets them from an
agent/task pair plus a pydantic converter. Here they are four named
constants and two small functions: `SYSTEM_PROMPT` and `USER_PROMPT` carry
the words the frameworks assemble from sem strings and YAML, `RESPONSE_FORMAT`
is the strict schema byLLM's `type_to_schema` would derive from
`-> list[MeetingTask]`, and `_parse_tasks` is the decode. A reply that fails
to decode gets one corrective retry naming the error — the same recovery
class byLLM's runtime performs on a schema miss.

**Nothing at all, for the token accounting.** byLLM hooks
`litellm.success_callback` and then polls in `TokenUsage.settle()` until the
counters stop moving, because litellm runs callbacks on a background pool and
the miss is silent. CrewAI has to patch the openai SDK's shared transport,
because its native provider and its instructor converter construct clients
independently and no single event bus sees both. Here the call site is eleven
lines away from the counter: `_complete` reads `response.usage` before it
returns, so `register_token_tracking()` survives only as a documented no-op
that keeps `main.py` diffing cleanly against `main.jac`.

**Nothing at all, for the fan-out.** Both frameworks dress the three
deterministic steps as pipeline nodes (graph nodes to walk, Flow methods to
listen). They contain no model calls and no branching, so here they are three
one-method classes called in a `for` loop — kept as classes only so the file
mirrors `nodes.jac`.

## The prompts

This is the prompt-engineering side, so the prompts are the implementation.
Every sentence is lifted from what the other two sides put in front of the
same model, and nothing was added beyond it — extra instructions (say, about
deduplication or embedded-instruction traps) would make this side smarter
than the frameworks it is benchmarked against:

| prompt text | source |
| --- | --- |
| "You are a meeting transcript analysis agent." | CrewAI `agents.yaml` role |
| "You are an expert in analyzing meeting transcripts and summarizing the discussions into actionable tasks. Your ability to identify important issues helps ensure teams can follow up and address key points effectively." | CrewAI `agents.yaml` backstory, verbatim |
| "Analyze the meeting transcript and break the discussion down into a list of important, well-structured, actionable tasks that a team can follow up on. Document each task thoroughly." | byLLM `sem GeneratingTasks.analyse_meeting_transcript`, verbatim |
| "name: Short, actionable title for the task." | byLLM `sem MeetingTask.name`, verbatim |
| "description: Detailed description of the task: clear instructions, steps to reproduce, and acceptance criteria where applicable." | byLLM `sem MeetingTask.description`, verbatim (CrewAI's `tasks.yaml` expected_output says the same in one sentence) |
| "Here is the meeting transcript for your reference:" | CrewAI `agents.yaml` goal / `tasks.yaml` description, verbatim |
| the closing JSON-shape sentence | the schema restated in prose, as byLLM's schema hint injection does for its response_format |

The two field sems also appear as `description` entries in `RESPONSE_FORMAT`,
which is where byLLM carries them.

## Where CrewAI and byLLM disagree

Four places, and this side follows byLLM at each:

1. **LLM calls per run.** byLLM extracts the task list in one strict-schema
   call. CrewAI spends at least two — the agent's completion, then the
   converter's instructor call to coerce the text into `MeetingTaskList` —
   which its own `llm_calls` metric records. This side makes one call (two
   only if the first reply fails to decode).
2. **Temperature.** byLLM's runtime sends `temperature=0.7` when jac.toml
   sets no `[byllm.call_params]`, which this project's does not.
   `LLM(model="gpt-4o")` on the CrewAI side sends no temperature at all
   (provider default 1.0). `TEMPERATURE = 0.7` follows byLLM. No `max_tokens`
   on any side.
3. **How often the transcript is sent.** byLLM serializes the `transcript`
   argument into the request once. CrewAI interpolates `{transcript}` into
   both the agent's goal and the task description, so every run pays for the
   transcript twice in one prompt. Once, here.
4. **The prompt words themselves.** The sem strings and the YAML overlap but
   are not identical (only CrewAI names Trello and "steps to reproduce" in
   the instruction body; only byLLM's sems define the two fields). The
   prompts above take byLLM's wording wherever both sides cover the same
   ground, and CrewAI's role/backstory framing where byLLM has no equivalent
   — byLLM's runtime supplies its own generic framing that the sems slot
   into, and a hand-written system prompt needs *some* opening sentence.

Documented divergences from byLLM itself, both invisible to the eval: the
strict-schema wrapper key is `"tasks"` (byLLM's internal spelling is
`"schema_object_wrapper"`; the model sees the key, the scoring never does),
and the schema's name is `MeetingTaskList` rather than a derived type name.

## The eval

Wired into the shared harness as `--impl openai_sdk`:

```bash
cd ../eval
python run.py --impl openai_sdk --repeat 3
python score.py runs/ --judge
```

`run.py` copies the case transcript into an isolated workdir as
`meeting_notes.txt`, runs `main.py` there with the interpreter running the
harness (the only requirement is that it can `import openai`), and captures
`tool_outputs.json`, wall time and token usage into
`runs/results_openai_sdk_<case>_r<k>.json` — the identical shape the other
two produce, so `score.py` needs no changes at all.
