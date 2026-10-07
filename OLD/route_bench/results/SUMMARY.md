# Learnt model routing in byLLM -- four agents, 2026-09-08

Candidates: strong = glm-5.2 ($0.42 / $1.32 per 1M tokens), cheap = gemma4:31b ($0.09 / $0.34), both on ollama.com.
Router: LLMRouter KNNRouter, trained per agent on requests the runtime logged from the strong model, replayed on both
candidates and labelled by the call site's own type contract. Routed = agent unchanged, `[byllm.routing]` on.
Judge = glm-5.2 (1-5). Every routed number is on cases the router never saw in training.

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
| | | **routed** | accuracy 3/3 (5 typed retries escalated to glm) | - | 57 | **0.0695** | **-21%** |
| YT-Navigator | q07-q12 | strong | routing acc 1.00, retrieval hit 1.00 | 5.00 | 25 | 0.0235 | - |
| | | cheap | routing acc 1.00, retrieval hit 0.00 | 1.00 | 19 | 0.0023 | -90% |
| | | **routed** | routing acc 1.00, retrieval hit 1.00 | 4.67 | 18 | **0.0035** | **-85%** |

Where the router sent each call site (routed mode, calls per model):

| agent | call site | declared return / tools | gemma4:31b | glm-5.2 |
|---|---|---|---:|---:|
| Email A | filter_emails | enum Classification | 11 | 0 |
| | email_action_agent | ThreadAnalysis, tools=[web_search] | 4 | 6 |
| | email_response_writer | DraftReply, tools=[web_search] | 5 | 0 |
| Email B | filter_emails | enum Classification | 17 | 0 |
| | email_action_agent / email_response_writer | struct + tools | 0 | 29 |
| Meeting | analyse_meeting_transcript | list[MeetingTask] | 0 | 5 |
| FactCheck | decompose_claim | list[str] | 3 | 15 |
| | assess_evidence | Finding | 6 | 15 |
| | verify_claim | Verdict | 4 | 14 |
| YT-Navigator | visit routing (`visit [-->] by llm`) | list of node handles | 6 | 0 |
| | tool_reply | AgentAnswer, tools | 11 | 1 |

Notes
- Same task quality as strong on all four agents; cost -17% to -85% depending on how much of the agent the router can move.
- The cheap model alone is not a baseline anyone would ship: it loops on tools (Email, FactCheck), never satisfies a
  list-of-struct contract (Meeting: 938 retries, $1.15), or skips retrieval (YT-Navigator: judge 1.0).
- Email split B judge gap (4.12 vs 3.50) is noise: both modes wrote drafts with glm-5.2.
- Sample sizes are small by design (3-6 cases per agent); this is an existence demonstration, not a benchmark.
- CodeAgent (SWE-bench Lite, docker), harder set: router trained on django-11099 + django-16255 (58 requests; the sympy-13480
  collection failed on a root-owned workspace left by a killed container), evaluated on sympy-13471, scikit-learn-13439,
  seaborn-3010 (instances small local models had failed on). All three modes resolve 3/3; routed sends 67 of 99 `work` calls to
  gemma, 4 typed retries escalate, cost $0.2179 vs $0.3133 strong (-30%). Earlier easy set (pytest-5227, astropy-14995,
  `results/codeagent_easy`): all 2/2, routed $0.0854 vs $0.3262 (-74%).
  Three jac processes were SIGKILLed mid-run by something outside the agent (35 GB free at the time); the two graded ones had already written their patch.

## Compact view (one row per agent)

| agent | Strong (accuracy + cost) | Cheap | Routed | USD saved |
|---|---|---|---|---|
| YTNavigator (6 questions) | retrieval hit 1.00, answer 5.0/5 · $0.0235 | retrieval hit 0.00, answer 1.0/5 · $0.0023 | retrieval hit 1.00, answer 4.7/5 · $0.0035 | $0.0200 (-85%) |
| Email Auto-response (6 batches, two splits pooled) | filter F1 0.96, drafts 13/13 · $0.0491 | filter F1 0.96, drafts 13/13 · $0.0727 | filter F1 0.96, drafts 13/13 · $0.0247 | $0.0244 (-50%) |
| HoVer Fact Check (3 claims) | accuracy 3/3 · $0.0881 | accuracy 1/3 · $0.0240 | accuracy 3/3 · $0.0695 | $0.0186 (-21%) |
| Meeting Assistant (5 meetings) | completed 5/5, judge F1 1.00 · $0.0053 | completed 0/5 · $1.1479 | completed 5/5, judge F1 1.00 · $0.0044 | $0.0009 (-17%) |
| CodeAgent (SWE-bench Lite, 7 instances the cheap model fails) | resolved 4/7 · $2.1759 | resolved 0/7 · $0.2069 | resolved 3/7 · $1.7223 | $0.4536 (-21%) |

Email pools split A (eval 4-6: -80%) and split B (eval 1-3: -24%). Cheap is cheaper only where it fails the task.

## CodeAgent difficulty ladder (2026-09-09)

gemma4:31b alone resolves most SWE-bench Lite instances: 2/2 (easy pair), 3/3 (harder triple), 10/12 and 7/12 in two
screening rounds (`results/codeagent_screen`). The seven it failed cleanly form the hard set (`results/codeagent_hard2`,
router retrained on 301 requests from nine other instances):

| set | strong | cheap | routed |
|---|---|---|---|
| easy pair (pytest-5227, astropy-14995) | 2/2 · $0.326 | 2/2 · $0.032 | 2/2 · $0.085 (-74%) |
| harder triple (sympy-13471, sklearn-13439, seaborn-3010) | 3/3 · $0.313 | 3/3 · $0.043 | 3/3 · $0.218 (-30%) |
| first hard pair (sympy-17139, sympy-18621), router v1 | 2/2 · $0.318 | 0/2 · $0.036 | 1/2 · $0.136 (-57%) |
| hard set of 7, router v2 | 4/7 · $2.176 | 0/7 · $0.207 | 3/7 · $1.722 (-21%) |

Strong itself is not stable on the hard set: it resolved sympy-17139 and sympy-18621 in the pair run and neither in the
seven-instance run. Routed's 3/7 vs 4/7 is within that run-to-run spread (routed solved sphinx-10325, which strong missed).
The escalation guard fired only 3 times in 470 calls: a well-formed tool call passes the contract even when the patch is
wrong, so on agentic coding the type contract cannot substitute for outcome labels. Routed trajectories were also longer
(7.2M vs 4.9M prompt tokens), which ate most of the per-token saving.
