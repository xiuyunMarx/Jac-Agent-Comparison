# FactCheck -- LangGraph arm

`fact_check.py` is `../Jac/fact_check.jac`'s node graph as a compiled
`StateGraph`, node for node:

| Jac                              | LangGraph                                                     |
|----------------------------------|---------------------------------------------------------------|
| `node ClaimAnalyzer`             | `"analyzer"` node: `decompose_claim`, one model call           |
| `visit scouts` (fan-out)         | `add_conditional_edges("analyzer", fan_out, ["scout"])` returning one `Send` per question |
| `node EvidenceScout`             | `"scout"` node: cached Wikipedia search + `assess_evidence`     |
| fan-in barrier (completed == expected) | LangGraph superstep semantics: `"verifier"` runs once all Sends finish |
| `node EvidenceAnalyzer`          | `"verifier"` node: `verify_claim`, one model call              |
| `verifier ++> analyzer`          | `add_conditional_edges("verifier", after_verdict, {"analyzer", END})` |
| `walker FactCheck` fields        | `State` TypedDict; `evidences` has an `operator.add` reducer   |

The parallel scouts append to `evidences` through the reducer; each entry is
tagged `(round, slot)` and rendered in that order, so every prompt sees the same
deterministic record the other arms build. Each scout sees only the evidence of
earlier rounds (the `sem` contract). The `(round, slot)` tag is bookkeeping and
is stripped from what the model sees.

With the default knobs (`FC_SCOUTS=2`, `FC_ROUNDS=6`, `FC_MIN_ROUNDS=6`) a
claim takes 6 rounds: 6 decompose + 12 assess + 6 verify = 24 calls, plus one
correction turn for any reply that did not parse.

## Model seam

`ChatOpenAI` from `langchain_openai` pointed at ollama.com, `temperature=0`,
`max_tokens=4096`. `with_structured_output` is deliberately not used: GLM 5.2
on ollama.com does not enforce `response_format`, so the JSON shape is stated in
the prompt, the reply is parsed leniently (fences and prose stripped) and one
correction turn is allowed -- the same protocol as the OpenAI SDK arm, so the
two Python arms differ only in orchestration.

## Running

```sh
# from FactCheck/ so ./wiki_cache is the shared search cache
export OLLAMA_API_KEY=...            # ollama.com key
python LangGraph/fact_check.py "The claim to check"
FC_CLAIM="..." FC_TRACE=trace.jsonl python LangGraph/fact_check.py
```

Knobs (identical to the Jac arm): `FC_SCOUTS`, `FC_ROUNDS`, `FC_MIN_ROUNDS`,
`FC_PASSAGE_CHARS`, `FC_CACHE_DIR` (default `./wiki_cache`), `FC_TOOL_DELAY_S`,
`FC_CLAIM`. Extra: `FC_TRACE` (one JSON line per model call), `FC_MODEL` /
`FC_MODEL_BASE` (default `openai/glm-5.2` at `https://ollama.com/v1`).
Summarise a trace with `python ../trace_summary.py trace.jsonl`.

Interpreter: Python >= 3.10 with `langgraph`, `langchain-openai`, `requests`
(`~/miniconda3/envs/jaseci/bin/python`: langgraph 1.6.0, openai 2.6.1).
