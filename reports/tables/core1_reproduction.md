| Metric | City (R, published) | Python, published rows | Python, 2019 snapshot join |
|---|---|---|---|
| Test inspections / with a critical violation | 1,637 / 258 | 1,637 / 258 | 1,665 / 263 |
| Training inspections | 17,075 | 17,075 | 17,116 |
| Avg. days earlier | 7.44 | 7.44 | 7.59 |
| SD of days earlier | 25.20 | 25.24 | 25.10 |
| Median / max days earlier | 6 / 58 | 6 / 58 | 6 / 58 |
| Violations found later (share) | 99 (38%) | 99 (38%) | 100 (38%) |
| First-half share, actual -> model | 54.7% -> 68.2% | 54.7% -> 68.2% | 54.4% -> 68.4% |
| Test AUC (normalized Gini) | not published | 0.672 (0.344) | 0.675 (0.350) |
| Brier score | not published | 0.1266 | 0.1265 |
| City XGBoost (saved scores): days earlier / AUC | 7.79 / 0.679 | 7.79 / 0.679 |  |
