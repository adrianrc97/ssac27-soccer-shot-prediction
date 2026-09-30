# Is Shot Distance Enough?

This project looks at a simple question in soccer analytics: how much do shot angle and header information improve goal prediction compared with using shot distance alone?

The analysis uses open-play shots from the 2023 FIFA Women's World Cup and compares three logistic regression models.

## Research question

Can shot angle and header status add useful predictive information beyond shot distance when estimating the probability of a goal?

## Data

The project uses StatsBomb Open Data for the 2023 FIFA Women's World Cup.

- 64 matches
- 1,566 open-play shots
- 134 goals
- Competition ID: 72
- Season ID: 107

The train/test split was done by match so that shots from the same match were not divided between training and testing.

- Training: 51 matches, 1,220 shots
- Testing: 13 matches, 346 shots
- Goals in test set: 40

## Models

Three logistic regression models were compared:

- **Model A:** shot distance
- **Model B:** shot distance + shot angle
- **Model C:** shot distance + shot angle + header status

The predictors were standardized using the training data only.

## Evaluation

The models were evaluated using:

- Brier score
- Log loss
- ROC-AUC

I also used 1,000 match-level bootstrap resamples to estimate uncertainty for the differences between models.

## Results

| Model | Brier Score | Log Loss | ROC-AUC |
|---|---:|---:|---:|
| A: Distance | 0.0960 | 0.3375 | 0.7077 |
| B: Distance + Angle | 0.0913 | 0.3242 | 0.7087 |
| C: Distance + Angle + Header | 0.0877 | 0.3115 | 0.7376 |

Adding shot angle reduced the Brier score by 0.0047, with a 95% bootstrap confidence interval from 0.0022 to 0.0081. The ROC-AUC change was 0.0010, with a 95% interval from -0.0066 to 0.0100, so there was no clear improvement in discrimination from angle alone.

Adding header status reduced the Brier score by another 0.0036, with a 95% interval from 0.0014 to 0.0057. ROC-AUC increased by 0.0289, with a 95% interval from 0.0017 to 0.0554.

Overall, the results show that adding simple shot information can improve probability estimates while keeping the model easy to interpret.

## How to run

Install the required Python packages:

```bash
pip install -r requirements.txt
```

Then run:

```bash
python analysis.py
```

The script creates the model results and output files in the `results` folder.

## Limitations

This analysis is based on one tournament and one held-out match split, so the results may not generalize to other leagues, tournaments, or seasons.

The models also use only a small number of pre-shot variables. More detailed contextual information could improve prediction, but the purpose of this project is to measure how much value can be added with a simple and interpretable model.

## Data source

Data comes from **StatsBomb Open Data**.

https://github.com/statsbomb/open-data
