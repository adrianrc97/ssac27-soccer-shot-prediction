# Results Summary

## Sample

- 1,566 open-play shots from 64 matches
- 134 goals (8.56%)
- Training set: 1,220 shots from 51 matches
- Test set: 346 shots from 13 matches, including 40 goals
- Grouped 80/20 train/test split by match, random seed 42

## Model performance

| Model | Brier score | Log loss | ROC-AUC |
|---|---:|---:|---:|
| A: distance | 0.0960 | 0.3375 | 0.7077 |
| B: distance + angle | 0.0913 | 0.3242 | 0.7087 |
| C: distance + angle + header | 0.0877 | 0.3115 | 0.7376 |

Constant-rate baseline: Brier score 0.1037; log loss 0.3672.

## Pairwise comparisons

Match-cluster bootstrap with 1,000 resamples and 95% percentile confidence intervals:

- Brier reduction, Model A minus Model B: **0.0047** (95% CI: 0.0022 to 0.0081)
- Brier reduction, Model B minus Model C: **0.0036** (95% CI: 0.0014 to 0.0057)
- ROC-AUC change, Model B minus Model A: **0.0010** (95% CI: -0.0066 to 0.0100)
- ROC-AUC change, Model C minus Model B: **0.0289** (95% CI: 0.0017 to 0.0554)

Angle improved held-out probability accuracy, while its ROC-AUC change was not clearly different from zero. Adding header status produced an additional improvement in probability accuracy and a positive ROC-AUC change in the held-out sample.
