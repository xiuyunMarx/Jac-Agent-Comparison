# Meeting Assistant Datasets

Framework-neutral evaluation cases for the two meeting-assistant
implementations (CrewAI and byLLM). Both agents do the same job — read a
meeting transcript, extract actionable tasks (`name` + `description`), then
fan out to Trello / CSV / Slack — so one labeled dataset serves both.

Both implementations are pinned to `gpt-4o` so any score difference is the
framework, not the model.

## Layout

Each case is a pair of files:

| File | Contents |
|---|---|
| `meeting_NNN.txt` | The raw transcript, exactly what the agent reads |
| `meeting_NNN.json` | Ground-truth labels for scoring |

`meeting_001.txt` is the original `meeting_notes.txt` both implementations
ship with, now labeled.

## Label schema

```jsonc
{
  "case_id": "meeting_002",
  "title": "...",
  "description": "what this case tests",
  "transcript_file": "meeting_002.txt",
  "edge_case": null,                    // or one of the categories below
  "expected_task_count_range": [5, 8],  // inclusive; sane granularity bounds
  "expected_tasks": [
    {
      "id": "gt_01",
      "name": "canonical short title",
      "owner": "who, if stated (else null)",
      "due": "deadline, if stated (else null)",
      "key_points": ["facts the task description should cover"]
    }
  ],
  "acceptable_extras": ["borderline topics that count as neither hit nor hallucination"],
  "must_not_extract": [
    {"topic": "...", "reason": "why extracting this is an error"}
  ]
}
```

`meeting_007.json` additionally carries an `injection_checks` block with the
three safety assertions specific to that case.

## Cases

| Case | Edge case | Tasks | Tests |
|---|---|---|---|
| 001 | — | 8 | Baseline: the original Stripe kickoff transcript |
| 002 | — | 7 | Owners, deadlines, acceptance criteria; FYIs (hiring, vacation) must not become tasks |
| 003 | `no_action_items` | 0 | Pure status meeting; correct output is an empty list |
| 004 | `retracted_decision` | 3 | Tasks committed then cancelled mid-meeting; rejected proposals |
| 005 | `noisy_transcript` | 7 | Long postmortem; items buried in crosstalk; joke task trap (coffee machine) |
| 006 | `duplicate_discussion` | 3 | Same topics raised twice and explicitly merged; dedup and task-boundary respect |
| 007 | `prompt_injection` | 4 | Verbatim malicious ticket in the transcript; must not create the injected task nor suppress real ones |
| 008 | `off_domain` | 6 | Conference planning (non-engineering); numbers and deadlines must survive |
| 009 | `conditional_tasks` | 3 | Conditional trigger to preserve; delegate not in the room; explicitly deferred topic |
| 010 | `minimal` | 2 | Two-person sync; must not pad a tiny meeting into extra tasks |

## Scoring semantics (for the eval harness)

- **Recall**: every `expected_tasks` entry should match at least one extracted
  task. Matching is semantic (task granularity varies), so match on topic,
  then check `key_points` coverage within the matched description.
- **Precision / hallucination**: every extracted task must map to an
  `expected_tasks` entry or an `acceptable_extras` topic. Anything matching a
  `must_not_extract` topic is an explicit error (worse than a stray extra —
  these are the traps the case was built around).
- **Count discipline**: total extracted tasks should land inside
  `expected_task_count_range`. Splitting or merging beyond the range signals
  granularity problems even when recall is perfect.
- **Duplicates**: two extracted tasks matching the same ground-truth entry
  count as one hit plus one duplicate.
- **Coverage depth**: share of `key_points` present in the matched task's
  description — this is where "well-documented tasks" is actually measured.
- **Owner/deadline capture**: when `owner`/`due` are non-null, check they
  appear in the matched description (the `MeetingTask` schema has no dedicated
  fields for them).

Agent pipelines are noisy even at temperature 0 — run each case 3–5 times and
compare means, not single runs.

## Running an implementation against a case

Both implementations currently read a hardcoded `meeting_notes.txt` from their
working directory, so:

```bash
# byLLM
cd ../byLLM
cp ../datasets/meeting_004.txt meeting_notes.txt
jac run main.jac

# CrewAI
cd ../CrewAI
cp ../datasets/meeting_004.txt meeting_notes.txt
crewai flow kickoff   # or: uv run kickoff
```

(A future eval harness should capture each run's extracted tasks plus mock
Trello/Slack/CSV output as JSON, keyed by `case_id`, and score both
implementations side by side — same pattern as `Email-Auto-response/eval`.)
