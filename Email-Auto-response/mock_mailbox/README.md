# Mock Mailbox

A framework-agnostic, in-memory Gmail stand-in for evaluating the email
auto-response agents in this repo (CrewAI-LangGraph and byLLM) against
identical, reproducible inputs — no Gmail credentials, no real drafts, no
Tavily key.

## What it provides

`MockMailbox(dataset_path)` loads one batch of emails from a dataset JSON and
exposes the operations the agents use:

| Method | Replaces | Behavior |
|---|---|---|
| `search()` | `GmailSearch` | Returns `{id, threadId, snippet, sender}` per email |
| `get_thread(thread_id)` | `GmailGetThread` | Returns the full thread as text |
| `web_search(query)` | `TavilySearchResults` | Deterministic canned results from the dataset |
| `create_draft(to, subject, message)` | `GmailCreateDraft` | Captures the draft in `mailbox.drafts` |

Everything is recorded for evaluation: `drafts`, `draft_errors` (failed
Create-Draft tool calls, e.g. pipe-parsing failures), `thread_requests`,
`web_queries`, and `usage` (LLM tokens and dollars — see below).
`save_results(path, final_state)` writes them all to JSON.

## Token metering

Constructing a `MockMailbox` also installs the shared token meter
(`token_meter.py`), which patches the response path inside the `openai` SDK.
Both implementations reach that SDK through litellm, so one hook counts tokens
identically for either, with no agent-side instrumentation:

```python
mailbox = MockMailbox(dataset)      # meter installed, counters at zero
...run the agent...
mailbox.usage_line()                # "9 LLM calls | 13,500 in + 2,250 out = 15,750 tokens | $0.0563"
mailbox.usage_summary()             # totals, per-model breakdown, per-call detail
```

Counters reset per mailbox, i.e. per run over one batch. Prices live in
`token_meter.PRICING` (USD per 1M tokens) and are overridable via
`pricing.json` here or `$EVAL_PRICING_FILE`; `EVAL_TOKEN_METER=0` turns
metering off. Streamed responses are not metered (none of the agents stream);
any that occur are counted in `usage.streamed_calls` so they cannot silently
vanish from the totals.

## Dataset format

See `datasets/batch_001.json`. One file = one inbox batch (one polling cycle):

```jsonc
{
  "case_id": "batch_001",
  "owner_email": "impact@jaseci.org",   // used as MY_EMAIL for self-exclusion
  "emails": [
    {
      "id": "msg_001",
      "threadId": "thr_001",
      "sender": "Name <addr@example.com>",
      "subject": "...",
      "snippet": "...",                  // what the filter agent sees
      "full_thread": [ {"from", "to", "date", "body"}, ... ],
      "labels": {                        // ground truth for scoring (agents never see this)
        "category": "action_required | newsletter | notification | promotional | self_sent | spam | fyi",
        "should_respond": true,
        "expected_recipient": "addr@example.com",
        "key_points_to_address": ["..."],
        "expected_tone": "professional",
        "edge_case": null
      }
    }
  ],
  "web_search_results": { "keyword": "canned result text" }
}
```

`labels` are never surfaced through the mailbox API — they exist only for a
scoring script to compare captured drafts against ground truth.

## Dataset suite

One file per inbox batch, each targeting a distinct failure mode:

| Batch | Focus | Emails | Expected drafts |
|---|---|---|---|
| `batch_001` | Baseline mixed batch: clear positives/negatives, a `\|`-in-content email (stresses the pipe-separated Create Draft format), and a self-sent email the polling node must exclude | 7 | 3 |
| `batch_002` | Hard filtering: negatives that look personal (calendar invite, security alert, receipt, personalized cold outreach), positives that look like noise (terse internal follow-up, conference invitation), and a **prompt-injection** email that must get no reply | 7 | 2 |
| `batch_003` | Tone & context: casual thread (style mimicry), frustrated third follow-up (apologetic tone), technical question rewarding web research, and a polite closure that explicitly needs no reply | 4 | 3 |
| `batch_004` | All noise — correct outcome is **zero drafts**; includes an actionable-looking CI failure and question-shaped SEO spam | 6 | 0 |
| `batch_005` | Empty inbox — the workflow must take the no-new-mail path without invoking the crew | 0 | 0 |
| `batch_006` | High load: 5/5 action-required with distinct recipients — tests draft completeness under batch pressure | 5 | 5 |

Totals: 29 emails, 13 expected drafts. Per-email design notes (what each
email tests and why) are in [`datasets/README.md`](datasets/README.md);
scoring lives in [`../eval/`](../eval/).

## Running the CrewAI-LangGraph agent on it

The CrewAI-LangGraph version uses this mailbox as its only email backend
(the Gmail/Tavily interfaces have been removed):

```bash
cd ../CrewAI-LangGraph
.venv/bin/python main.py                  # uses datasets/batch_001.json
.venv/bin/python main.py path/to/other.json
```

Needs only `OPENAI_API_KEY` (in `.env` or the environment) — the LLM calls are
real, the mailbox is not. Results land in `CrewAI-LangGraph/mock_output/`.
