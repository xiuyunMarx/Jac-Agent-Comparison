# ReAct loop on BFCL v4 web_search

| arm | runs | passed | accuracy | crashed | step cap | calls | prompt tok (mean / median) | completion tok | tool calls | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| jac | 20 | 17 | 85.0% | 0 | 0 | 5.7 | 41,573.1 / 18,312 | 740.7 | 5.2 | 53.8 |
| openai | 20 | 18 | 90.0% | 0 | 0 | 5.4 | 47,874.2 / 11,854 | 776.6 | 5.5 | 74.4 |
| langgraph | 20 | 19 | 95.0% | 0 | 0 | 4.3 | 17,165.2 / 9,348 | 617.5 | 3.9 | 55.8 |
| nooa | 20 | 18 | 90.0% | 0 | 1 | 7.1 | 36,816.8 / 21,292 | 1,205.9 | 5.6 | 47.6 |

## Per task

| task | jac | openai | langgraph | nooa |
|---|---|---|---|---|
| web_search_0 | FAILED | PASSED | PASSED | PASSED |
| web_search_1 | PASSED | PASSED | PASSED | PASSED |
| web_search_2 | PASSED | FAILED | PASSED | PASSED |
| web_search_3 | PASSED | PASSED | PASSED | PASSED |
| web_search_4 | PASSED | PASSED | PASSED | PASSED |
| web_search_5 | PASSED | PASSED | PASSED | PASSED |
| web_search_6 | PASSED | PASSED | PASSED | PASSED |
| web_search_7 | PASSED | PASSED | PASSED | PASSED |
| web_search_8 | PASSED | PASSED | PASSED | PASSED |
| web_search_9 | PASSED | PASSED | PASSED | PASSED |
| web_search_10 | FAILED | PASSED | PASSED | FAILED |
| web_search_11 | PASSED | PASSED | PASSED | PASSED |
| web_search_12 | PASSED | PASSED | PASSED | PASSED |
| web_search_13 | PASSED | PASSED | PASSED | PASSED |
| web_search_14 | PASSED | PASSED | PASSED | PASSED |
| web_search_15 | PASSED | PASSED | PASSED | PASSED |
| web_search_16 | PASSED | PASSED | PASSED | PASSED |
| web_search_17 | PASSED | PASSED | PASSED | PASSED |
| web_search_18 | FAILED | FAILED | FAILED | FAILED |
| web_search_19 | PASSED | PASSED | PASSED | PASSED |
