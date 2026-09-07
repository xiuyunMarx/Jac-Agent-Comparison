# Coding agent — LangGraph

A five-phase coding agent using LangGraph for the workflow and LangChain's
`create_agent` for each phase's tool loop.

The Jac, OpenAI SDK, and LangGraph implementations share the repository tools,
task instructions, phase transitions, evidence guards, and budget policy.
This implementation uses standard LangChain schemas, messages, and structured
output; it does not reproduce byLLM's internal prompts or wire protocol.

## Run

```bash
pip install -e .
export OLLAMA_API_KEY=...
export CODEAGENT_MODEL=openai/glm-5.2

python main.py "issue text" /path/to/repo
python tests/tool_checks.py
python -m unittest discover -s tests -p 'test_*.py' -v
python tests/smoke.py  # Calls the configured model.
```

`OLLAMA_API_BASE`, `OPENAI_API_BASE`, or `OPENAI_BASE_URL` selects the compatible
endpoint; the default is `https://ollama.com/v1`. `OPENAI_API_KEY` can supply
the key. `CODEAGENT_TEMPERATURE` defaults to 0.7.

## Files

- `nodes.py`: repository tools with typed signatures and tool docstrings,
  model configuration, shared history/tool log, phase goals, and transitions.
- `main.py`: the outer `StateGraph`, phase calls through `create_agent`,
  budget middleware, evidence guards, and structured routing.
- `orchestrator.py`: the existing SWE-bench result interface and callback
  telemetry. Set `CODEAGENT_TRACE` to save model messages, tool schemas,
  replies, and usage when running through this adapter.

## Workflow and budgets

```text
Plan -> Explore -> Edit -> Verify -> Finish
                    ^        |
                    +-- repair
           ^                 |
           +------ relocate -+
```

The phase model-round limits are 6 / 20 / 20 / 10. The shared budget is
60 repository tool calls. As in Jac, the budget is checked between model
rounds, after the first round of a phase; it is not a hard per-tool cutoff,
so a tool batch may exceed 60. On exhaustion, the model gets one additional
call with no repository tools to summarize. Tool execution uses
`max_concurrency=1`. The summary request is not added to shared history.

Verify first checks the repair and tool budgets, then increments the repair
counter and requires a successful write plus a command/snippet after the
last edit or revert. Only then does the model choose an outgoing transition.
The repair limit is 3. Relocation is offered once, after the first repair
opportunity, and consumes that opportunity even if the model chooses another
edge. An invalid or failed routing decision falls back to Edit.

The tests use scripted HTTP responses with the real LangChain/LangGraph
runtime. They cover tool schemas, shared history, budgets, routing guards,
and an end-to-end edit/reproduce/verify workflow, including adapter telemetry.
They require no model credentials.

## Comparison boundary

Tool descriptions and parameter descriptions now live in method docstrings;
`StructuredTool.from_function(..., parse_docstring=True)` infers schemas and
preserves Python defaults. Routing uses a Pydantic result with
`with_structured_output`. Native message formatting, final-answer handling,
and parser retries can produce different requests and model behavior from
Jac. Score equivalence and byte-identical traces are not claimed.
