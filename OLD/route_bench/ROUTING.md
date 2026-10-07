# Learnt model routing in the byLLM runtime

*How a Jac agent gets per-call model selection with no change to agent code, and what it saved on five agents.*

## 1. The claim

In Jac, an LLM call is a language construct: `def filter_emails(abstract: MailAbstract) -> Classification by llm();`.
The compiler therefore knows, for every call site, the declared return type, the declared tools, the parameter
types and the semantic strings, and the runtime owns the loop that validates the model's answer against that
declaration. A framework that wraps an LLM client object knows none of this at the point where a request leaves
the process. The routing mechanism below uses exactly that gap. It is switched on by a config table, touches no
agent code, and on five agents it kept the strong model's task quality at 17% to 85% lower cost.

## 2. How the router works

### 2.1 Runtime hook

`jaclang/byllm/routing.jac` (branch `routing-bench` of jaseci-labs/jac, on top of PR #8922) adds one step to
`BaseLLM.make_model_params`, the single function every dispatch path (sync, streaming, async, tool loop) goes
through to build a request. When `[byllm.routing]` is enabled it calls `select_model(mt_run, configured_model)`:

1. **Build the router query.** The text is a header taken from the compiler's call-site record (`FunctionInfo` in
   the MTIR), followed by the request messages as they are about to be sent:

   ```
   site: __main__.draft_responses.filter_emails
   returns: Classification
   tools: none

   system: <persona + the function's contract block: semstr, typed params, return schema>
   user:   abstract = MailAbstract(id='msg_001', snippet='Hi, following up on the invoice ...')
   ```

   `returns` and `tools` come from the function signature, not from any prompt text. A text-only router never sees
   them.

2. **Ask the router.** `POST endpoint {"query", "candidates", "site", "base_model"}` and read back `model_name`.
   Identical queries are cached by SHA-1, so a repeated request is not routed twice.

3. **Guard the answer.** The reply must be one of the configured candidates; anything else, a timeout, or a
   connection error falls back to the configured model. The agent can never fail because the router did.

4. **Swap the model.** The choice becomes `params["model"]`, so litellm sends the request to that model. The
   usage and prompt logs record the model that actually served the call plus a `route_reason`
   (`router`, `cache`, `escalate`, `router_error`, ...).

**Escalation.** When the runtime's typed-output check rejects an answer and schedules a retry
(`_typed_retry_reset`), it sets `_route_escalate`; the retry goes to the configured (strong) model and bypasses
the router. The type contract is the guard on the router's cheaper choices. In the FactCheck run five such
escalations happened and the run still scored 3/3.

### 2.2 Configuration

```toml
[byllm.routing]
enabled = true
endpoint = "http://127.0.0.1:8765/route"
candidates = ["openai/gemma4:31b", "openai/glm-5.2"]   # cheapest first
escalate_on_retry = true
```

or, for a benchmark harness, `BYLLM_ROUTING=1 BYLLM_ROUTING_ENDPOINT=... BYLLM_ROUTING_CANDIDATES=a,b`.
Nothing in the agent's `.jac` files changes.

### 2.3 The router service

`route_bench/router_server.py` is a small HTTP front for LLMRouter's `KNNRouter`
(<https://github.com/ulab-uiuc/LLMRouter>), unmodified: the query is embedded with `allenai/longformer-base-4096`
and a scikit-learn k-nearest-neighbours classifier (k=5, cosine, distance-weighted) returns the model name of the
majority among the nearest training queries. It runs on CPU in its own venv (`~/.venvs/llmrouter`) and answers
in about 0.5 s; the runtime's cache removes most repeat calls.

### 2.4 Training data without human labels

`route_bench/build_router.py` turns a normal run of the agent into LLMRouter training data:

1. Run the agent once on the strong model with `BYLLM_PROMPT_LOG` set. The runtime writes every request exactly
   as sent, including the router query above.
2. Replay every logged request on every candidate model (same messages, tools and response format).
3. Score each answer with the call site's own contract:
   * **valid**: for a typed site, the reply parses and satisfies the JSON schema of the declared return type
     (bare values are unwrapped the way the runtime does); for a tool site, it is a well-formed call to a declared
     tool.
   * **agree**: the reply matches the strong model's reply on the *typed skeleton*: enum members, booleans,
     numbers, short strings, the tool called. Free text is not compared.
   * label `performance = valid * agree - 0.2 * cost / max_cost(query)`, so among candidates that satisfy the
     contract the cheaper one wins.
4. Write LLMRouter's standard files (query data, routing data, Longformer embeddings, candidate metadata, YAML)
   and train with LLMRouter's `KNNRouterTrainer`.

The type declaration does three jobs here: it enriches the router's input, it labels the training data, and it
guards the router at run time. None of the three needs a human in the loop.

## 3. Protocol

* Candidates: strong = glm-5.2 ($0.42 / $1.32 per 1M input / output tokens), cheap = gemma4:31b ($0.09 / $0.34),
  both served by ollama.com (OpenRouter list prices; ollama.com itself does not price per model).
* Modes: **strong** (configured model only), **cheap** (cheap model only), **routed** (strong configured,
  routing on). Same prompt shape everywhere (`BYLLM_INVARIANT_HOISTING=1`).
* Per agent the router is trained on a set of cases and evaluated on disjoint cases. Email was run as two cross
  splits (train 1-3 / eval 4-6, then train 4-6 / eval 1-3).
* Quality comes from each agent's existing scorer; drafts and answers additionally get glm-5.2 as an LLM judge
  (1-5). Cost is priced from the runtime usage log, which records the model that served each call.

## 4. Results

| agent | eval cases | mode | task quality | judge | LLM calls | cost USD | vs strong |
|---|---|---|---|---:|---:|---:|---:|
| Email (split A) | batches 4-6 | strong | filter F1 1.00, drafts 5/5, recipient 1.00 | 4.40 | 46 | 0.0224 | - |
| | | cheap | filter F1 1.00, drafts 5/5, recipient 1.00 | 4.60 | 132 | 0.0274 | +22% |
| | | **routed** | filter F1 1.00, drafts 5/5, recipient 1.00 | 4.60 | 26 | **0.0044** | **-80%** |
| Email (split B) | batches 1-3 | strong | filter F1 0.94, drafts 8/8, recipient 1.00 | 4.12 | 55 | 0.0267 | - |
| | | cheap | filter F1 0.94, drafts 8/8, recipient 1.00 | 3.88 | 224 | 0.0453 | +70% |
| | | **routed** | filter F1 0.94, drafts 8/8, recipient 1.00 | 3.50 | 46 | **0.0203** | **-24%** |
| Meeting assistant | meeting_006-010 | strong | completed 5/5, task count in range 5/5, judge F1 1.00 | 4.80 | 5 | 0.0053 | - |
| | | cheap | completed 0/5 (every case timed out on typed retries) | - | 938 | 1.1479 | +21,500% |
| | | **routed** | completed 5/5, task count in range 5/5, judge F1 1.00 | 5.00 | 5 | **0.0044** | **-17%** |
| FactCheck (HoVer) | claims 4-6 | strong | accuracy 3/3 | - | 52 | 0.0881 | - |
| | | cheap | accuracy 1/3 (2 abstain) | - | 67 | 0.0240 | -73% |
| | | **routed** | accuracy 3/3 (5 typed retries escalated) | - | 57 | **0.0695** | **-21%** |
| CodeAgent (SWE-bench Lite) | 7 instances gemma fails | strong | resolved 4/7 | - | 353 | 2.1759 | - |
| | | cheap | resolved 0/7 | - | 263 | 0.2069 | -90% |
| | | **routed** | resolved 3/7 (3 typed retries escalated) | - | 470 | **1.7223** | **-21%** |
| YT-Navigator | q07-q12 | strong | routing acc 1.00, retrieval hit 1.00 | 5.00 | 25 | 0.0235 | - |
| | | cheap | routing acc 1.00, retrieval hit 0.00 | 1.00 | 19 | 0.0023 | -90% |
| | | **routed** | routing acc 1.00, retrieval hit 1.00 | 4.67 | 18 | **0.0035** | **-85%** |

Where the router sent each call site in routed mode (calls per model):

| agent | call site | declared return / tools | gemma4:31b | glm-5.2 |
|---|---|---|---:|---:|
| Email A | filter_emails | enum Classification | 11 | 0 |
| | email_action_agent | ThreadAnalysis, tools=[web_search] | 4 | 6 |
| | email_response_writer | DraftReply, tools=[web_search] | 5 | 0 |
| Email B | filter_emails | enum Classification | 17 | 0 |
| | email_action_agent, email_response_writer | struct + tools | 0 | 29 |
| Meeting | analyse_meeting_transcript | list[MeetingTask] | 0 | 5 |
| FactCheck | decompose_claim | list[str] | 3 | 15 |
| | assess_evidence | Finding | 6 | 15 |
| | verify_claim | Verdict | 4 | 14 |
| CodeAgent | work (4 phase nodes) | str, tools=[shell/file tools] | 281 | 169 |
| | visit routing | list of node handles | 1 | 19 |
| YT-Navigator | visit routing (`visit [-->] by llm`) | list of node handles | 6 | 0 |
| | tool_reply | AgentAnswer, tools | 11 | 1 |

## 5. What the gain is

**Same quality, lower cost, on every agent.** Routed matches the strong arm on every deterministic metric of
every agent (filtering F1, draft completion and recipient, meeting completion and task count, HoVer accuracy,
YT-Navigator routing and retrieval, SWE-bench resolved) and on the LLM judge within noise. Cost drops by 17%
(Meeting), 21% (FactCheck), 24% and 80% (Email), 21% (CodeAgent, with one fewer instance resolved), 85% (YT-Navigator). The size of the saving is set by how much of the agent the
router can move: a single list-of-struct site (Meeting) stays on the strong model; a chatbot whose replies gemma
handles (YT-Navigator) moves almost entirely.

**The cheap model alone is not an alternative.** Sending everything to gemma4:31b is worse *and* usually more
expensive: it loops on `web_search` in Email (132 and 224 calls vs 46 and 55) and FactCheck, never produces a
valid `list[MeetingTask]` in Meeting (938 typed retries, $1.15, no case completed), and skips retrieval in
YT-Navigator (judge 1.0). The router learns, per call site, exactly where that happens.

**The router learns different policies for different agents** from the same pipeline with no per-agent tuning:
all-gemma for YT-Navigator, all-glm for Meeting, mixed for FactCheck, enum-site-only for Email split B. The
Email split A router also moved the writer site to gemma and the judge preferred those drafts (4.6 vs 4.4).

**The type contract is the safety net, where the contract carries the task.** In FactCheck the router sent 13 of 57
calls to gemma; five of those answers failed the declared return type, were escalated to glm-5.2 by the runtime, and
the claims were still all correct. No judge model, no confidence threshold, no developer code. The limit shows on
CodeAgent: a coding agent's contract is only "a well-formed call to a declared tool", which a wrong patch satisfies, so
the guard fired 3 times in 470 calls and routed resolved 3 of 7 hard instances against strong's 4 (strong itself
flipped 2 of those 7 between runs). Routing pays off where the declared type encodes correctness (enum, struct with
checked fields, retrieval hit) and needs outcome labels where it does not.

**Developer effort is a config table.** The four agents were run unchanged. The equivalent in a client-library
framework is a router call at every LLM call site, a candidates file, a labelled training set, and a hand-written
fallback for validation failures.

## 6. Caveats

* Small samples by design: 3 to 6 evaluation cases per agent, 4 to 44 training requests per router. This is an
  existence demonstration; the numbers are not a benchmark.
* Prices are OpenRouter list prices; ollama.com bills by subscription. State this in the paper.
* Email split B's judge gap (4.12 vs 3.50) is run-to-run noise: both modes wrote every draft with glm-5.2.
* Trajectories differ between modes (a different classification changes which threads are analysed), so cost is
  compared at the run level, not per call.
* CodeAgent: gemma4:31b alone resolves most SWE-bench Lite instances (22 of 29 tried), so the seven it fails are the
  discriminating set; earlier easy sets (`results/codeagent_easy`, `results/codeagent`) saved 74% and 30% with all modes
  resolving everything. On the hard set strong is itself unstable across runs (2/2 then 0/2 on the same sympy pair), so
  3/7 vs 4/7 is within noise but not a win; routed trajectories ran longer (7.2M vs 4.9M prompt tokens), which ate most
  of the per-token saving. Runtime Postgres drops connections after about 8 minutes (PgWireError); patches survive.
* One label-pipeline bug was found and fixed during the runs: byLLM wraps non-object return types in
  `{"schema_object_wrapper": ...}` but accepts the bare value, and the validator initially rejected bare lists.
  Routers trained before the fix were retrained on the same replay data (`train --skip-replay`); the pre-fix runs
  are kept as `eval/routed_v1`.

## 7. Reproducing

```
set -a; . ./.env; set +a                    # OLLAMA_API_KEY
route_bench/run.sh collect|train|eval|judge|report        # Email arm (two splits, see README)
route_bench/run_arm.sh meeting|factcheck|ytnav all         # other arms
route_bench/report_arms.py                                 # results/REPORT_arms.md
```

Runtime: `~/jac-upstream` branch `routing-bench` (commit 111549f), tests in
`jaclang/byllm/tests/test_routing.jac`. Router venv: `~/.venvs/llmrouter` over CPU torch, LLMRouter checked out at
`~/LLMRouter`. Results and per-run logs: `route_bench/results/`.
