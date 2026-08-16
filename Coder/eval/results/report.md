# Jac-GPT three-system eval report

220 items, 3 repeats, systems: langgraph, jac, jac-byllm-router.

## Headline (all turns)

| system           | routing_acc           |   judge_score(1-5) | compile_rate   | appropriate_rate   | error_rate   |   llm_calls/turn |   prompt_tok/turn |   completion_tok/turn |   $/1k turns |   latency_p50_ms |   latency_p95_ms |
|:-----------------|:----------------------|-------------------:|:---------------|:-------------------|:-------------|-----------------:|------------------:|----------------------:|-------------:|-----------------:|-----------------:|
| langgraph        | 99.4% [98.3%, 100.0%] |               4.06 | 47.1%          | 84.7%              | 0.0%         |             2.71 |              4486 |                   410 |         2.45 |             7134 |            14567 |
| jac              | 98.5% [97.4%, 99.8%]  |               4.09 | 49.2%          | 87.0%              | 0.0%         |             3.62 |              6865 |                   394 |         3.38 |             8998 |            18526 |
| jac-byllm-router | 99.3% [98.3%, 100.0%] |               4.13 | 51.7%          | 88.4%              | 0.0%         |             3.59 |              6959 |                   388 |         3.4  |             9007 |            17722 |

## Routing accuracy by category

| category   | langgraph   | jac    | jac-byllm-router   |
|:-----------|:------------|:-------|:-------------------|
| rag_qa     | 99.6%       | 99.2%  | 98.8%              |
| coding     | 100.0%      | 97.5%  | 100.0%             |
| debugging  | 100.0%      | 100.0% | 100.0%             |
| small_talk | 100.0%      | 100.0% | 100.0%             |
| off_topic  | 95.0%       | 100.0% | 100.0%             |
| multi_turn | 100.0%      | 95.0%  | 98.3%              |

## Mean judge score (1-5) by category

| category   |   langgraph |   jac |   jac-byllm-router |
|:-----------|------------:|------:|-------------------:|
| rag_qa     |        4.85 |  4.78 |               4.75 |
| coding     |        3    |  3.7  |               3.64 |
| debugging  |        3.53 |  3.12 |               3.38 |
| multi_turn |        5    |  5    |               5    |

## Total tokens / turn by category

| category   |   langgraph |    jac |   jac-byllm-router |
|:-----------|------------:|-------:|-------------------:|
| rag_qa     |       4,791 |  7,471 |              7,636 |
| coding     |       4,123 |  6,034 |              6,305 |
| debugging  |      10,699 | 14,626 |             14,313 |
| small_talk |         390 |    710 |                714 |
| off_topic  |         474 |    783 |                789 |
| multi_turn |       4,547 |  7,214 |              7,446 |

## LLM calls / turn by category

| category   |   langgraph |   jac |   jac-byllm-router |
|:-----------|------------:|------:|-------------------:|
| rag_qa     |        3    |  3.77 |               3.77 |
| coding     |        2.59 |  3.63 |               3.66 |
| debugging  |        3    |  4.9  |               4.77 |
| small_talk |        2    |  2    |               2    |
| off_topic  |        2    |  2    |               2    |
| multi_turn |        2.66 |  3.62 |               3.58 |

## Multi-turn: turn-2 routing accuracy (context-dependent follow-ups)

| system           | routing_correct   |
|:-----------------|:------------------|
| langgraph        | 100.0%            |
| jac              | 90.0%             |
| jac-byllm-router | 96.7%             |

## Routing confusion (gold rows x routed columns, all repeats)

### langgraph

| gold_agent   |   CodingChat |   DebuggerChat |   OffTopicChat |   QAChat |   RagChat |
|:-------------|-------------:|---------------:|---------------:|---------:|----------:|
| RagChat      |            0 |              0 |              1 |        0 |       278 |
| CodingChat   |          153 |              0 |              0 |        0 |         0 |
| DebuggerChat |            0 |            156 |              0 |        0 |         0 |
| QAChat       |            0 |              0 |              0 |       72 |         0 |
| OffTopicChat |            0 |              3 |             57 |        0 |         0 |

### jac

| gold_agent   |   CodingChat |   DebuggerChat |   OffTopicChat |   QAChat |   RagChat |
|:-------------|-------------:|---------------:|---------------:|---------:|----------:|
| RagChat      |            0 |              2 |              0 |        0 |       277 |
| CodingChat   |          144 |              9 |              0 |        0 |         0 |
| DebuggerChat |            0 |            156 |              0 |        0 |         0 |
| QAChat       |            0 |              0 |              0 |       72 |         0 |
| OffTopicChat |            0 |              0 |             60 |        0 |         0 |

### jac-byllm-router

| gold_agent   |   CodingChat |   DebuggerChat |   OffTopicChat |   QAChat |   RagChat |
|:-------------|-------------:|---------------:|---------------:|---------:|----------:|
| RagChat      |            0 |              0 |              3 |        0 |       276 |
| CodingChat   |          151 |              2 |              0 |        0 |         0 |
| DebuggerChat |            0 |            156 |              0 |        0 |         0 |
| QAChat       |            0 |              0 |              0 |       72 |         0 |
| OffTopicChat |            0 |              0 |             60 |        0 |         0 |

## Pairwise comparisons (Wilcoxon signed-rank over per-item means)

| metric | pair | delta (A-B) | p-value |
|---|---|---|---|
| judge_score | langgraph vs jac | -0.04 | 0.6555 |
| judge_score | langgraph vs jac-byllm-router | -0.07 | 0.3544 |
| judge_score | jac vs jac-byllm-router | -0.04 | 0.1720 |
| total_tokens | langgraph vs jac | -2335.60 | 0.0000 |
| total_tokens | langgraph vs jac-byllm-router | -2409.82 | 0.0000 |
| total_tokens | jac vs jac-byllm-router | -74.22 | 0.0567 |
| latency_ms | langgraph vs jac | -2022.77 | 0.0000 |
| latency_ms | langgraph vs jac-byllm-router | -1870.27 | 0.0000 |
| latency_ms | jac vs jac-byllm-router | +152.50 | 0.9074 |
| routing_correct | langgraph vs jac | +0.01 | 0.3991 |
| routing_correct | langgraph vs jac-byllm-router | +0.00 | 1.0000 |
| routing_correct | jac vs jac-byllm-router | -0.01 | 0.4164 |

## Reading notes

- The three systems share verbatim agent prompts and RAG config; differences trace to the router mechanism and framework overhead (byllm vs LangGraph serialization).
- Known asymmetries by design: Jac-Rag-GPT routes at byllm's default temperature 0.7 (others at 0); the LangGraph router never sees chat history; fallback agents differ (LangGraph -> RagChat, ByllmRouter -> OffTopicChat).
- $/1k turns assumes gpt-4.1-mini at $0.4/M input, $1.6/M output.
- multi_turn turn-2 accuracy isolates history-dependent routing.