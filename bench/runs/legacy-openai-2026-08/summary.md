# Benchmark sweep legacy-openai-2026-08

- model: `gpt-5 (CodeAgent) / gpt-4o (meeting, email) / gpt-4.1-mini (RagGPT) / gpt-4o-mini+gpt-4o (YTNavigator)` at `https://api.openai.com/v1`
- context window: provider default
- mode: full, repeats: 3
- judge: gpt-4o (Email, meeting) / gpt-4.1 (RagGPT)

> Judged by gpt-4o/gpt-4.1 (a separate, stronger model), unlike the local-model runs.

## Quality and cost

| benchmark | arm | implementation | metric | score | judge | tok in | tok out | calls | err | parse-fail | over |
|---|---|---|---|---|---|---|---|---|---|---|---|
| meeting-assistant | jac/byLLM | `byLLM` | task count in expected range | 100.0% | - | 28,842 | 7,553 | 30 | 0.0% | 0.0% | 30 |
| meeting-assistant | framework | `CrewAI` | task count in expected range | 100.0% | - | 57,384 | 10,849 | 30 | 0.0% | 0.0% | 30 |
| Email-Auto-response | jac/byLLM | `byLLM` | filtering F1 | 63.3% | 4.59 | 35,510 | 6,594 | 72 | - | 22.2% | 6 |
| Email-Auto-response | framework | `CrewAI-LangGraph` | filtering F1 | 80.0% | 4.01 | 106,420 | 7,509 | 61 | - | 0.0% | 6 |
| YTNavigator | jac/byLLM | `byllm` | retrieval hit rate | 18.8% | - | 58,863 | 6,424 | 3 | 0.0% | 0.0% | 23 |
| YTNavigator | framework | `langgraph` | retrieval hit rate | 20.0% | - | 89,402 | 3,280 | 3 | 8.7% | 61.9% | 23 |
| RagGPT | framework | `langgraph` | routing accuracy | 99.4% | 4.06 | 4,486 | 410 | 3 | 0.0% | - | per turn |
| RagGPT | jac/byLLM | `jac` | routing accuracy | 98.5% | 4.09 | 6,865 | 394 | 4 | 0.0% | - | per turn |
| RagGPT | jac/byLLM (router variant) | `jac-byllm-router` | routing accuracy | 99.3% | 4.13 | 6,959 | 388 | 4 | 0.0% | - | per turn |

## Three-arm coverage

| benchmark | jac/byLLM | framework | openai_sdk |
|---|---|---|---|
| meeting-assistant | yes | yes | **MISSING** |
| Email-Auto-response | yes | yes | **MISSING** |
| YTNavigator | yes | yes | **MISSING** |
| RagGPT | yes | yes | **MISSING** |
| CodeAgent | **MISSING** | **MISSING** | **MISSING** |

**7 empty cell(s):** meeting-assistant/openai_sdk, Email-Auto-response/openai_sdk, YTNavigator/openai_sdk, RagGPT/openai_sdk, CodeAgent/jac/byLLM, CodeAgent/framework, CodeAgent/openai_sdk

A missing cell is not a result. Check that stage's log before reading anything above as a framework comparison.

## Stages

| stage | result | steps |
|---|---|---|
