# FactCheck -- OpenAI SDK arm (pure prompt engineering)

`fact_check.py` is `../Jac/fact_check.jac` written against the raw `openai`
client. There is no framework between the program and the wire: every
`by llm(...)` clause on the Jac side is one `chat.completions.create` call
here, and everything byLLM derives from the `obj` / `sem` declarations (the
output schema, the parse, the correction retry) is spelled out as prompt text
and `json.loads`.

## Structure (same as the Jac graph)

| Jac                                  | here                                   | model calls          |
|--------------------------------------|----------------------------------------|----------------------|
| `ClaimAnalyzer.decompose_claim`      | `decompose_claim()`                    | 1 per round          |
| `EvidenceScout.websearch` + `assess_evidence` | `investigate()`, threads in parallel | 1 per question   |
| `EvidenceAnalyzer.verify_claim`      | `verify_claim()`                       | 1 per round          |
| `verifier ++> analyzer` retry loop   | the `while True` in `fact_check()`     |                      |

With the default knobs (`FC_SCOUTS=2`, `FC_ROUNDS=6`, `FC_MIN_ROUNDS=6`) a
claim always takes 6 rounds: 6 decompose + 12 assess + 6 verify = 24 calls,
plus one correction turn for any reply that did not parse. The evidence list
(findings plus the retrieved passage, cut at `FC_PASSAGE_CHARS`) is included in
every prompt and grows by two entries per round, exactly as on the Jac side.

Two places where this arm follows the `sem` contract rather than the Jac
execution order: scouts of one round run concurrently and each sees only the
evidence of *earlier* rounds (Jac visits scouts sequentially, so its second
scout also sees the first scout's fresh finding); results are appended in
question order, so the record is deterministic either way.

## Structured output without `response_format`

GLM 5.2 on ollama.com does not enforce `response_format`, so the shape is
carried by the prompt: the system prompt states the exact JSON keys and enum
values, the user message repeats the one-line shape at the end, code fences
and surrounding prose are stripped before `json.loads`, and a reply that still
fails validation gets exactly one correction turn (the error text is what the
model is told to fix). This is the same two-call worst case as byLLM's schema
hint + retry.

## Running

```sh
# from FactCheck/ so ./wiki_cache is the shared search cache
export OLLAMA_API_KEY=...            # ollama.com key
python OpenaiSDK/fact_check.py "The claim to check"
FC_CLAIM="..." FC_TRACE=trace.jsonl python OpenaiSDK/fact_check.py
```

Knobs (identical to the Jac arm): `FC_SCOUTS`, `FC_ROUNDS`, `FC_MIN_ROUNDS`,
`FC_PASSAGE_CHARS`, `FC_CACHE_DIR` (default `./wiki_cache`), `FC_TOOL_DELAY_S`,
`FC_CLAIM`. Extra here: `FC_TRACE` (one JSON line per model call: stage,
round, usage, full messages and reply), `FC_MODEL` / `FC_MODEL_BASE` (default
`openai/glm-5.2` at `https://ollama.com/v1`; the litellm provider prefix is
stripped on the wire). Summarise a trace with `python ../trace_summary.py trace.jsonl`.

Interpreter: any Python >= 3.10 with `openai` and `requests`
(`~/miniconda3/envs/jaseci/bin/python` has both).
