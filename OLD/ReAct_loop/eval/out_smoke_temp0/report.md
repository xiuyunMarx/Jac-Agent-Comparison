# ReAct loop on BFCL v4 web_search

| arm | runs | passed | accuracy | crashed | step cap | calls | prompt tok (mean / median) | completion tok | tool calls | wall s |
|---|---|---|---|---|---|---|---|---|---|---|
| jac | 5 | 2 | 40.0% | 0 | 2 | 13.6 | 70,939.2 / 68,574 | 574.4 | 12.8 | 45.7 |
| openai | 5 | 3 | 60.0% | 0 | 2 | 12.0 | 74,811.4 / 78,592 | 504.0 | 11.6 | 21.9 |
| langgraph | 5 | 2 | 40.0% | 0 | 3 | 13.4 | 97,514.2 / 85,982 | 547.6 | 13.6 | 22.0 |
| nooa | 5 | 2 | 40.0% | 0 | 2 | 12.4 | 80,912.2 / 58,128 | 1,112.8 | 9.2 | 35.5 |

## Per task

| task | jac | openai | langgraph | nooa |
|---|---|---|---|---|
| web_search_0 | FAILED | PASSED | FAILED | FAILED |
| web_search_1 | PASSED | PASSED | PASSED | PASSED |
| web_search_2 | FAILED | FAILED | FAILED | FAILED |
| web_search_3 | PASSED | PASSED | PASSED | PASSED |
| web_search_4 | FAILED | FAILED | FAILED | FAILED |
