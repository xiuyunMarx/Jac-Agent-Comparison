Core logic LOC with and without protection code (see CRITERION.md, PROTECTION section; ranges in <app>.guard.json)

| app | arm | logic | protection | logic w/o protection | prot % | vs Jac (w/o) |
|---|---|---|---|---|---|---|
| FactCheck | Jac | 211 | 0 | 211 | 0% | 1.00x |
| FactCheck | LangGraph | 265 | 39 | 226 | 15% | 1.07x |
| FactCheck | OpenaiSDK | 245 | 40 | 205 | 16% | 0.97x |
| FactCheck | NOOA | 208 | 0 | 208 | 0% | 0.99x |
| Email-Auto-response | byLLM | 178 | 39 | 139 | 22% | 1.00x |
| Email-Auto-response | CrewAI-LangGraph | 204 | 4 | 200 | 2% | 1.44x |
| Email-Auto-response | openai_sdk | 404 | 72 | 332 | 18% | 2.39x |
| Email-Auto-response | NOOA | 167 | 23 | 144 | 14% | 1.04x |
| meeting-assistant | byLLM | 106 | 30 | 76 | 28% | 1.00x |
| meeting-assistant | CrewAI | 94 | 3 | 91 | 3% | 1.20x |
| meeting-assistant | openai_sdk | 161 | 22 | 139 | 14% | 1.83x |
| meeting-assistant | NOOA | 87 | 22 | 65 | 25% | 0.86x |
| YTNavigator | byLLM | 201 | 66 | 135 | 33% | 1.00x |
| YTNavigator | langchain | 387 | 66 | 321 | 17% | 2.38x |
| YTNavigator | openai_sdk | 442 | 62 | 380 | 14% | 2.81x |
| YTNavigator | NOOA | 139 | 16 | 123 | 12% | 0.91x |
| CodeAgent | Jac | 362 | 60 | 302 | 17% | 1.00x |
| CodeAgent | langgraph | 386 | 70 | 316 | 18% | 1.05x |
| CodeAgent | openai_sdk | 654 | 103 | 551 | 16% | 1.82x |
| CodeAgent | NOOA | 1208 | 571 | 637 | 47% | 2.11x |
