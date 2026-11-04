| Model | Inspector | Validation days earlier | Test days earlier [95% CI] | Minus city [95% CI] | SD | Found later | First-half share | AUC |
|---|---|---|---|---|---|---|---|---|
| City GLM (Python port, published) | yes | 10.41 | 7.44 [4.38, 9.94] | - | 25.2 | 99 | 68.2% | 0.672 |
| City XGBoost (saved 2017 scores) | yes |  | 7.79 [4.91, 10.33] | +0.38 [-0.31, +1.09] | 25.0 | 104 | 68.2% | 0.679 |
| LightGBM, city features | yes | 10.33 | 7.74 [4.74, 10.25] | +0.32 [-0.02, +0.72] | 24.8 | 99 | 67.8% | 0.678 |
| Ridge logistic, extended features | yes | 10.85 | 8.23 [5.24, 10.84] | +0.89 [+0.11, +1.67] | 25.4 | 96 | 70.9% | 0.689 |
| LightGBM, extended features | yes | 10.81 | 8.99 [6.18, 11.55] | +1.59 [+0.68, +2.44] | 24.5 | 94 | 70.5% | 0.702 |
| Blend (ridge + LightGBM), extended features **(selected)** | yes | 10.94 | 8.70 [5.77, 11.30] | +1.32 [+0.53, +2.12] | 25.0 | 95 | 70.2% | 0.697 |
| City GLM (Python port) | no | 3.15 | 2.20 [-0.66, 4.66] | -5.26 [-7.07, -3.42] | 25.4 | 124 | 56.6% | 0.570 |
| LightGBM, city features | no | 2.58 | 2.74 [0.15, 5.10] | -4.68 [-6.69, -2.72] | 24.2 | 120 | 57.0% | 0.580 |
| Ridge logistic, extended features | no | 6.18 | 5.51 [2.73, 8.15] | -1.85 [-3.51, -0.12] | 25.8 | 104 | 64.7% | 0.636 |
| LightGBM, extended features | no | 6.20 | 6.42 [3.58, 9.06] | -0.97 [-2.80, +0.83] | 25.5 | 93 | 69.0% | 0.653 |
| Blend (ridge + LightGBM), extended features | no | 6.39 | 6.09 [3.28, 8.78] | -1.27 [-3.03, +0.39] | 25.6 | 98 | 66.3% | 0.647 |
