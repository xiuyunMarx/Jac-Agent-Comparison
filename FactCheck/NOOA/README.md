# FactCheck -- NOOA arm (NVIDIA Object-Oriented Agents)

`fact_check.py` does what `../Jac/fact_check.jac` does, written the way the
[NOOA docs](https://github.com/NVIDIA-NeMo/labs-OO-Agents/tree/main/docs)
say to write an agent. There is no graph and no walker: an agent is a Python
object, an async method ending in `...` is implemented by the LLM, a method
with a real body is ordinary Python, and the workflow lives in a hidden
deterministic method. The program is two classes.

| Jac                                              | NOOA                                                                 | model calls    |
|--------------------------------------------------|----------------------------------------------------------------------|----------------|
| `node ClaimAnalyzer.decompose_claim by llm`      | `FactChecker.decompose_claim(...) -> list[str]: ...`                  | 1 per round    |
| `node EvidenceAnalyzer.verify_claim by llm`      | `FactChecker.verify_claim(...) -> Verdict: ...`                       | 1 per round    |
| `node EvidenceScout.websearch` (a `def`)         | `EvidenceScout.websearch()`, a regular method with a real body        |                |
| `node EvidenceScout.assess_evidence by llm`      | `EvidenceScout.assess_evidence(...) -> Finding: ...`                  | 1 per question |
| `can investigate with FactCheck entry`           | `EvidenceScout.investigate()`, hidden, deterministic: search then assess |             |
| `walker FactCheck` + `build_graph()` + retry loop| `FactChecker.run()`, hidden, deterministic: the loop, with the walker's fields as locals |  |
| `visit scouts` fan-out, fan-in barrier           | one `EvidenceScout` instance per question, `asyncio.gather`           |                |
| `obj` / `enum` + `sem` strings                   | Pydantic models with `Field(description=...)`; `Annotated[T, "..."]` on parameters |    |

With the default knobs (`FC_SCOUTS=2`, `FC_ROUNDS=6`, `FC_MIN_ROUNDS=6`) a claim
takes 6 rounds: 6 decompose + 12 assess + 6 verify = 24 calls, plus one
correction turn for any reply that did not validate.

## What NOOA supplies that the prompt-engineered arms spell out

- **The prompt.** System prompt = class docstring + `doc(type(self))`, NOOA's
  rendering of the class: visible methods with docstrings and `Args:` sections
  (from the `Annotated` descriptions), and the referenced Pydantic types with
  their field descriptions. Task message = method docstring + the arguments
  rendered as Python assignments. Nothing is interpolated by hand.
- **The output contract.** `PredictStrategy` sends a `response_format`
  json_schema built from the return annotation, extracts JSON from the reply
  (fences stripped), validates with Pydantic and feeds the error back for one
  more attempt (`PredictConfig(max_retries=2)`, the other arms' single
  correction turn). ollama.com ignores `response_format` for GLM, so on the
  wire this is the same shape-in-prompt + parse + retry protocol as the
  Python arms.
- **The workflow.** `run()` is plain Python: it enforces the question cap,
  fans out, waits for every scout, decides whether to loop. The model is
  asked three questions and never asked to remember the procedure.

## Settings that keep the arm comparable

- `@strategy(PredictStrategy(...))` on every `...` method. NOOA's default is
  CodeAct, where the model writes Python in a REPL and would call
  `self.websearch()` itself; that is a different call profile from `by llm`.
- `event_query=EventQuery.current_call()` on `FactChecker`: NOOA keeps an
  event history per object and would otherwise replay earlier rounds' task
  and output events into later prompts. With this, every call is stateless
  and the evidence list argument is the only carry-over, as in the other arms.
  Scouts are fresh objects per question, so they have no history to leak.
- `truncation=TruncationConfig(prefill_format=FormatConfig(max_string=None, ...))`:
  NOOA cuts rendered argument strings at 2000 chars by default; the passages
  are 4000 and the raw search results longer.
- The client is NOOA's `CompletionClient` (litellm) with `temperature=0`,
  `max_tokens=4096`, `drop_params=True`; it is subclassed only to write the
  `FC_TRACE` rows.

## Running

```sh
# from FactCheck/ so ./wiki_cache is the shared search cache
set -a; . ../.env; set +a          # OLLAMA_API_KEY
~/miniconda3/envs/nooa/bin/python NOOA/fact_check.py "The claim to check"
FC_CLAIM="..." FC_TRACE=trace.jsonl ~/miniconda3/envs/nooa/bin/python NOOA/fact_check.py
```

Knobs (identical to the Jac arm): `FC_SCOUTS`, `FC_ROUNDS`, `FC_MIN_ROUNDS`,
`FC_PASSAGE_CHARS`, `FC_CACHE_DIR` (default `./wiki_cache`), `FC_TOOL_DELAY_S`,
`FC_CLAIM`. Extra: `FC_TRACE` (one JSON line per model call: stage, round,
usage, full messages and reply), `FC_MODEL` / `FC_MODEL_BASE` (default
`openai/glm-5.2` at `https://ollama.com/v1`; NOOA is litellm, so the id is
used as-is). Summarise a trace with `python ../trace_summary.py trace.jsonl`.

Interpreter: Python >= 3.12 with `nooa` (0.0.9) and `requests`:
`~/miniconda3/envs/nooa/bin/python`. The eval harness runs it as the `nooa`
arm (`FC_PY_NOOA` overrides the interpreter).
