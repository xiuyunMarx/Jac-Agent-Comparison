# Equivalence vs jac (20 tasks, paired)

Margins: accuracy ±15% points, token ratio 0.70-1.30. Equivalent = 90% CI inside the margin; different = 95% CI excludes no difference.

| arm | accuracy | mean tokens (prompt + completion) |
|---|---|---|
| jac | 85% | 42,314 |
| openai | 90% | 48,651 |
| langgraph | 95% | 17,783 |
| nooa | 90% | 38,023 |

| arm vs jac | accuracy diff (points) | 90% CI | 95% CI | verdict | token ratio | 90% CI | 95% CI | verdict |
|---|---|---|---|---|---|---|---|---|
| openai | +5.0 | [-10.2, +20.2] | [-13.4, +23.4] | inconclusive | x0.78 | [x0.52, x1.18] | [x0.47, x1.28] | inconclusive |
| langgraph | +10.0 | [-1.9, +21.9] | [-4.4, +24.4] | inconclusive | x0.49 | [x0.34, x0.71] | [x0.32, x0.76] | different |
| nooa | +5.0 | [-3.6, +13.6] | [-5.5, +15.5] | equivalent | x1.17 | [x0.84, x1.62] | [x0.79, x1.73] | inconclusive |
