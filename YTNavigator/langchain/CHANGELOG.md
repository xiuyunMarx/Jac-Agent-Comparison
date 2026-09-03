# Changelog

## 0.1.3 - 2026-09-02

### Fixed
- ReAct history: `trim_messages(max_tokens=2000, start_on="human")` returned
  an empty list once a transcript-sized tool result arrived, so the model got
  a system-only request and answered nothing (four of 23 benchmark questions).
  The agent now sends the current turn whole plus the last three exchanges,
  like the other benchmark arms; the router window follows the same rule.
- Tool-path answers are normalized to `AgentOutput` JSON (fenced or
  prose-wrapped objects parsed, failures recorded as `output_parse_fallback`),
  as the non-tool path already did; `benchmark_run` tolerates fences too and
  no longer counts a fallback-wrapped reply as parsed.
- `minimise_chunks` dropped every semantic hit because vector-store metadata
  spells the span `start_time`/`duration`, not `start`/`end`; only the BM25
  chunks survived, so the model re-searched and read transcripts via SQL.
- The search tool returns JSON instead of a pydantic `repr`; the SQL tool
  describes the schema as markdown instead of a Python list of dicts.
- The non-tool reply prompt states the answer schema, as the tool path does.

## 0.1.2 - 2025-03-16

### Added
- Added Langsmith dataset creation and example addition
### Known Issues
- Poor exception handling
- UI responsiveness issues on mobile devices
- Supports only English for now
