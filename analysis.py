"""Reproduce the 2023 Women's World Cup shot-model comparison.

Data source: StatsBomb Open Data, competition 72, season 107.

Usage:
    python analysis.py
    python analysis.py --from-csv data/shot_level.csv
"""

import argparse
import json
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from sklearn.calibration import calibration_curve
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, roc_auc_score
from sklearn.model_selection import GroupShuffleSplit
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from urllib3.util.retry import Retry

BASE_URL = "https://raw.githubusercontent.com/hudl/open-data/master/data"
COMPETITION_ID = 72
SEASON_ID = 107
RANDOM_SEED = 42

FEATURES = {
    "A: distance": ["distance"],
    "B: distance + angle": ["distance", "angle_deg"],
    "C: distance + angle + header": ["distance", "angle_deg", "is_header"],
}


def build_session():
    session = requests.Session()
    retry = Retry(
        total=5,
        connect=5,
        read=5,
        backoff_factor=1.3,
        status_forcelist=[429, 500, 502, 503, 504],
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers["User-Agent"] = "ssac27-soccer-research/1.0"
    return session


def load_json(session, url, cache_path):
    if cache_path.exists():
        with cache_path.open(encoding="utf-8") as handle:
            return json.load(handle)

    response = session.get(url, timeout=90)
    response.raise_for_status()
    obj = response.json()
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    cache_path.write_text(json.dumps(obj), encoding="utf-8")
    return obj


def shot_angle_degrees(x, y):
    """Return the angle subtended by the goalposts from the shot location."""
    dx = 120.0 - float(x)
    dy_near = 36.0 - float(y)
    dy_far = 44.0 - float(y)
    return math.degrees(math.atan2(abs(8.0 * dx), dx * dx + dy_near * dy_far))


def parse_shot(event, match):
    if event.get("type", {}).get("name") != "Shot":
        return None
    if event.get("period") == 5:
        return None

    shot = event.get("shot") or {}
    if (shot.get("type") or {}).get("name") != "Open Play":
        return None

    location = event.get("location") or []
    if len(location) < 2 or not all(isinstance(v, (int, float)) for v in location[:2]):
        return None

    x, y = map(float, location[:2])
    if not (0 <= x <= 120 and 0 <= y <= 80):
        return None

    body_part = (shot.get("body_part") or {}).get("name")
    outcome = (shot.get("outcome") or {}).get("name")
    if not body_part or not outcome:
        return None

    return {
        "match_id": int(match["match_id"]),
        "match_date": match["match_date"],
        "event_id": event["id"],
        "x": x,
        "y": y,
        "distance": math.hypot(120.0 - x, 40.0 - y),
        "angle_deg": shot_angle_degrees(x, y),
        "body_part": body_part,
        "is_header": int(body_part == "Head"),
        "is_goal": int(outcome == "Goal"),
    }


def download_dataset(root):
    session = build_session()
    cache_dir = root / "source_cache"

    matches_url = f"{BASE_URL}/matches/{COMPETITION_ID}/{SEASON_ID}.json"
    matches = load_json(session, matches_url, cache_dir / "matches.json")

    records = []
    manifest = []

    for match in matches:
        match_id = int(match["match_id"])
        events_url = f"{BASE_URL}/events/{match_id}.json"
        events = load_json(session, events_url, cache_dir / f"events_{match_id}.json")

        for event in events:
            row = parse_shot(event, match)
            if row is not None:
                records.append(row)

        manifest.append(
            {
                "match_id": match_id,
                "date": match["match_date"],
                "url": events_url,
            }
        )

    data = pd.DataFrame(records)
    if data.empty:
        raise ValueError("No eligible open-play shots were found in the source data.")

    data_dir = root / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    data.to_csv(data_dir / "shot_level.csv", index=False)
    pd.DataFrame(manifest).to_csv(data_dir / "source_match_manifest.csv", index=False)
    return data


def calculate_metrics(y_true, probabilities):
    return {
        "brier": float(brier_score_loss(y_true, probabilities)),
        "log_loss": float(log_loss(y_true, probabilities, labels=[0, 1])),
        "roc_auc": float(roc_auc_score(y_true, probabilities)),
    }


def bootstrap_difference(test, score_a, score_b, metric="brier", n_resamples=1000):
    rng = np.random.default_rng(RANDOM_SEED)
    match_ids = np.unique(test["match_id"].to_numpy())
    row_index = {
        match_id: np.flatnonzero(test["match_id"].to_numpy() == match_id)
        for match_id in match_ids
    }
    y_true = test["is_goal"].to_numpy()
    differences = []

    for _ in range(n_resamples):
        sampled_matches = rng.choice(match_ids, size=len(match_ids), replace=True)
        rows = np.concatenate([row_index[m] for m in sampled_matches])

        if metric == "brier":
            value = (
                brier_score_loss(y_true[rows], score_a[rows])
                - brier_score_loss(y_true[rows], score_b[rows])
            )
        else:
            if len(np.unique(y_true[rows])) < 2:
                continue
            value = (
                roc_auc_score(y_true[rows], score_b[rows])
                - roc_auc_score(y_true[rows], score_a[rows])
            )
        differences.append(value)

    return tuple(map(float, np.quantile(differences, [0.025, 0.975])))


def run_analysis(data, root):
    required = ["match_id", "distance", "angle_deg", "is_header", "is_goal"]
    data = data.dropna(subset=required).copy()

    for column in ["distance", "angle_deg", "is_header", "is_goal"]:
        data[column] = pd.to_numeric(data[column], errors="raise")

    data["match_id"] = data["match_id"].astype(int)
    data["is_goal"] = data["is_goal"].astype(int)

    splitter = GroupShuffleSplit(
        n_splits=1,
        test_size=0.20,
        random_state=RANDOM_SEED,
    )
    train_idx, test_idx = next(
        splitter.split(data, data["is_goal"], groups=data["match_id"])
    )
    train = data.iloc[train_idx].copy().reset_index(drop=True)
    test = data.iloc[test_idx].copy().reset_index(drop=True)

    if set(train["match_id"]) & set(test["match_id"]):
        raise AssertionError("Training and test matches overlap.")

    output_dir = root / "results"
    output_dir.mkdir(parents=True, exist_ok=True)

    y_test = test["is_goal"].to_numpy()
    predictions = {
        "match_id": test["match_id"],
        "is_goal": test["is_goal"],
    }
    probabilities_by_model = {}
    metric_rows = []

    for model_name, feature_names in FEATURES.items():
        model = make_pipeline(
            StandardScaler(),
            LogisticRegression(max_iter=2000, random_state=RANDOM_SEED),
        )
        model.fit(train[feature_names], train["is_goal"])
        probabilities = model.predict_proba(test[feature_names])[:, 1]

        probabilities_by_model[model_name] = probabilities
        predictions[model_name] = probabilities
        metric_rows.append(
            {
                "model": model_name,
                "train_shots": len(train),
                "test_shots": len(test),
                "train_matches": train["match_id"].nunique(),
                "test_matches": test["match_id"].nunique(),
                **calculate_metrics(y_test, probabilities),
            }
        )

    metrics_df = pd.DataFrame(metric_rows)
    metrics_df.to_csv(output_dir / "model_metrics.csv", index=False)
    pd.DataFrame(predictions).to_csv(output_dir / "test_predictions.csv", index=False)

    baseline_probability = train["is_goal"].mean()
    baseline = calculate_metrics(y_test, np.repeat(baseline_probability, len(test)))

    model_a = probabilities_by_model["A: distance"]
    model_b = probabilities_by_model["B: distance + angle"]
    model_c = probabilities_by_model["C: distance + angle + header"]

    ci_ab_brier = bootstrap_difference(test, model_a, model_b, metric="brier")
    ci_bc_brier = bootstrap_difference(test, model_b, model_c, metric="brier")
    ci_ab_auc = bootstrap_difference(test, model_a, model_b, metric="auc")
    ci_bc_auc = bootstrap_difference(test, model_b, model_c, metric="auc")

    summary = f"""# Results Summary

## Sample

- {len(data):,} open-play shots from {data['match_id'].nunique()} matches
- {int(data['is_goal'].sum())} goals ({100 * data['is_goal'].mean():.2f}%)
- Training set: {len(train):,} shots from {train['match_id'].nunique()} matches
- Test set: {len(test):,} shots from {test['match_id'].nunique()} matches, including {int(test['is_goal'].sum())} goals

## Baseline

- Brier score: {baseline['brier']:.4f}
- Log loss: {baseline['log_loss']:.4f}

## Pairwise comparisons

- Brier reduction, Model A minus Model B: {metrics_df.iloc[0].brier - metrics_df.iloc[1].brier:.4f} (95% CI: {ci_ab_brier[0]:.4f} to {ci_ab_brier[1]:.4f})
- Brier reduction, Model B minus Model C: {metrics_df.iloc[1].brier - metrics_df.iloc[2].brier:.4f} (95% CI: {ci_bc_brier[0]:.4f} to {ci_bc_brier[1]:.4f})
- ROC-AUC change, Model B minus Model A: {metrics_df.iloc[1].roc_auc - metrics_df.iloc[0].roc_auc:.4f} (95% CI: {ci_ab_auc[0]:.4f} to {ci_ab_auc[1]:.4f})
- ROC-AUC change, Model C minus Model B: {metrics_df.iloc[2].roc_auc - metrics_df.iloc[1].roc_auc:.4f} (95% CI: {ci_bc_auc[0]:.4f} to {ci_bc_auc[1]:.4f})
"""
    (output_dir / "results_summary.md").write_text(summary, encoding="utf-8")

    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    for model_name, probabilities in probabilities_by_model.items():
        observed, predicted = calibration_curve(
            y_test,
            probabilities,
            n_bins=6,
            strategy="quantile",
        )
        ax.plot(predicted, observed, marker="o", label=model_name)

    ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
    ax.set(
        xlim=(0, 0.6),
        ylim=(0, 0.6),
        xlabel="Mean predicted goal probability",
        ylabel="Observed goal fraction",
        title="2023 Women's World Cup: test-set calibration",
    )
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(output_dir / "calibration.png", dpi=170)
    plt.close(fig)

    return metrics_df


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--from-csv",
        type=Path,
        help="Use a previously generated shot-level CSV instead of downloading source data.",
    )
    args = parser.parse_args()

    root = Path(__file__).resolve().parent
    data = pd.read_csv(args.from_csv) if args.from_csv else download_dataset(root)
    metrics_df = run_analysis(data, root)
    print(metrics_df.to_string(index=False))


if __name__ == "__main__":
    main()
