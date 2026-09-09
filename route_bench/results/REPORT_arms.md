## codeagent
Router: KNN trained on 58 requests (held-out oracle acc 36% on 14); offline test picks {'openai/gemma4:31b': 11, 'openai/glm-5.2': 3}

| mode | LLM calls | prompt tok | completion tok | cost USD | escalated retries | quality |
|---|---:|---:|---:|---:|---:|---|
| strong | 118 | 686,114 | 19,066 | 0.3133 | 0 | resolved=3/3; mwaskom__seaborn-3010=steps=8 err=n resolved=yes; scikit-learn__scikit-learn-13439=steps=17 err=n resolved=yes; sympy__sympy-13471=steps=8 err=n resolved=yes |
| cheap | 74 | 443,402 | 8,644 | 0.0428 | 0 | resolved=3/3; mwaskom__seaborn-3010=steps=8 err=n resolved=yes; scikit-learn__scikit-learn-13439=steps=8 err=n resolved=yes; sympy__sympy-13471=steps=8 err=n resolved=yes |
| routed | 109 | 812,898 | 13,716 | 0.2179 | 4 | resolved=3/3; mwaskom__seaborn-3010=steps=8 err=n resolved=yes; scikit-learn__scikit-learn-13439=steps=8 err=n resolved=yes; sympy__sympy-13471=steps=8 err=n resolved=yes |

Call sites (calls per model):
- **strong**: visit: glm-5.2 x5; work: glm-5.2 x113
- **cheap**: visit: gemma4:31b x11; work: gemma4:31b x63
- **routed**: visit: gemma4:31b x6, glm-5.2 x4; work: gemma4:31b x67, glm-5.2 x32
