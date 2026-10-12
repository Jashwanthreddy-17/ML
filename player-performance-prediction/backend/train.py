"""Train Ridge and Lasso models for cricket batting-innings run prediction.

Mirrors the project abstract ("Player Performance Prediction Using
Ridge and Lasso Regression"):

- Observation unit: one batter in one match innings.
- Target: innings_runs = sum(runs_off_bat). Extras are NOT credited to
  the batter. Valid zero scores are retained.
- Features use ONLY the player's earlier history (strictly before the
  current match date) plus match context known before the innings
  (innings number, venue, teams) -> no target leakage.
- Chronological split: train on matches before 2022, test on 2022-2023.
- Ridge (L2 shrinkage) and Lasso (L1 shrinkage + feature selection)
  are compared against a mean baseline using MAE, MSE/RMSE and R^2.

Usage:
    python train.py            # trains and writes backend/artifacts/

Outputs (backend/artifacts/):
    models.joblib   fitted Ridge / Lasso / baseline pipelines + schema
    metrics.json    test-set MAE, MSE, RMSE, R^2 for all three models
    players.json    latest pre-innings stats per batter (for /api/predict)
    meta.json       teams, venues, feature names
"""

import json
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.dummy import DummyRegressor
from sklearn.linear_model import LassoCV, RidgeCV
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data" / "match_data.csv"
OUT = HERE / "artifacts"

SPLIT_DATE = "2022-01-01"      # chronological split: train < 2022, test >= 2022
MIN_PRIOR_INNINGS = 3          # minimum history needed for reliable features

NUMERIC_FEATURES = [
    "prior_innings",
    "career_avg", "career_sr", "career_innings_runs", "career_balls_faced",
    "career_fours", "career_sixes", "career_dismissal_rate", "career_best",
    "last5_avg_runs", "last5_avg_balls", "last5_sr", "last5_std_runs",
    "last3_avg_runs", "last3_avg_balls", "last3_sr",
    "days_since_last",
]
CATEGORICAL_FEATURES = ["innings", "venue", "batting_team", "bowling_team"]
TARGET = "innings_runs"


def load_and_aggregate(path: Path) -> pd.DataFrame:
    """Aggregate ball-by-ball deliveries into player-innings rows."""
    cols = ["match_id", "season", "start_date", "venue", "innings",
            "batting_team", "bowling_team", "striker",
            "runs_off_bat", "wides", "player_dismissed"]
    df = pd.read_csv(path, usecols=cols, dtype={"season": "string"}, low_memory=False)
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df = df.dropna(subset=["start_date", "striker"]).reset_index(drop=True)

    # Balls faced: deliveries the striker faced (wides don't count as faced).
    df["is_ball_faced"] = df["wides"].isna().astype(int)
    df["is_four"] = (df["runs_off_bat"] == 4).astype(int)
    df["is_six"] = (df["runs_off_bat"] == 6).astype(int)
    df["is_dismissed"] = (df["player_dismissed"] == df["striker"]).astype(int)

    keys = ["match_id", "innings", "batting_team", "bowling_team",
            "venue", "start_date", "striker"]
    pi = (df.groupby(keys, observed=True)
            .agg(innings_runs=("runs_off_bat", "sum"),
                 balls_faced=("is_ball_faced", "sum"),
                 fours=("is_four", "sum"),
                 sixes=("is_six", "sum"),
                 dismissed=("is_dismissed", "max"))
            .reset_index()
            .sort_values(["striker", "start_date", "match_id"])
            .reset_index(drop=True))
    return pi


def add_history_features(pi: pd.DataFrame) -> pd.DataFrame:
    """Add leakage-free features: only innings strictly before the current one."""
    g = pi.groupby("striker", sort=False)

    # Career totals from prior innings only.
    base = ["innings_runs", "balls_faced", "fours", "sixes", "dismissed"]
    cum = g[base].transform(lambda s: s.cumsum().shift(1))
    cum.columns = [f"career_{c}" for c in base]
    pi = pd.concat([pi, cum], axis=1)
    pi["prior_innings"] = g.cumcount()

    # Recent-form windows (last 5 / last 3 innings), prior only.
    for window, tag in [(5, "last5"), (3, "last3")]:
        pi[f"{tag}_avg_runs"] = g["innings_runs"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        pi[f"{tag}_avg_balls"] = g["balls_faced"].transform(
            lambda s, w=window: s.shift(1).rolling(w, min_periods=1).mean())
        balls = pi[f"{tag}_avg_balls"]
        pi[f"{tag}_sr"] = np.where(balls > 0,
                                   pi[f"{tag}_avg_runs"] / balls * 100.0, 0.0)

    pi["last5_std_runs"] = g["innings_runs"].transform(
        lambda s: s.shift(1).rolling(5, min_periods=1).std().fillna(0.0))
    pi["career_best"] = g["innings_runs"].transform(
        lambda s: s.shift(1).cummax())

    pi["career_avg"] = pi["career_innings_runs"] / pi["prior_innings"]
    pi["career_sr"] = np.where(
        pi["career_balls_faced"] > 0,
        pi["career_innings_runs"] / pi["career_balls_faced"] * 100.0, 0.0)
    pi["career_dismissal_rate"] = pi["career_dismissed"] / pi["prior_innings"]

    prev_date = g["start_date"].transform(lambda s: s.shift(1))
    pi["days_since_last"] = (pi["start_date"] - prev_date).dt.days.fillna(0).astype(int)

    # Keep only rows with enough history for stable features.
    pi = pi[pi["prior_innings"] >= MIN_PRIOR_INNINGS].reset_index(drop=True)
    return pi


def build_preprocessor() -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])


def evaluate(name: str, y_true, y_pred) -> dict:
    return {
        "MAE": round(float(mean_absolute_error(y_true, y_pred)), 3),
        "MSE": round(float(mean_squared_error(y_true, y_pred)), 3),
        "RMSE": round(float(np.sqrt(mean_squared_error(y_true, y_pred))), 3),
        "R2": round(float(r2_score(y_true, y_pred)), 4),
    }


def main() -> None:
    t0 = time.time()
    print("Loading and aggregating deliveries ...")
    pi = load_and_aggregate(DATA)
    print(f"  player-innings rows: {len(pi):,}")

    print("Engineering leakage-free history features ...")
    pi = add_history_features(pi)
    print(f"  rows with >= {MIN_PRIOR_INNINGS} prior innings: {len(pi):,}")

    train = pi[pi["start_date"] < SPLIT_DATE]
    test = pi[pi["start_date"] >= SPLIT_DATE]
    print(f"  train: {len(train):,} rows (before {SPLIT_DATE})")
    print(f"  test:  {len(test):,} rows ({SPLIT_DATE} onwards)")

    X_train, y_train = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES], train[TARGET]
    X_test, y_test = test[NUMERIC_FEATURES + CATEGORICAL_FEATURES], test[TARGET]

    models = {
        "ridge": Pipeline([("pre", build_preprocessor()),
                           ("model", RidgeCV(alphas=np.logspace(-2, 4, 25)))]),
        "lasso": Pipeline([("pre", build_preprocessor()),
                           ("model", LassoCV(alphas=np.logspace(-2, 4, 25),
                                             max_iter=10000))]),
        "baseline": Pipeline([("pre", build_preprocessor()),
                              ("model", DummyRegressor(strategy="mean"))]),
    }

    metrics = {}
    for name, pipe in models.items():
        print(f"Training {name} ...")
        pipe.fit(X_train, y_train)
        metrics[name] = evaluate(name, y_test, pipe.predict(X_test))
        if name == "ridge":
            print(f"  best alpha: {pipe.named_steps['model'].alpha_:.4f}")
        if name == "lasso":
            nz = int((pipe.named_steps["model"].coef_ != 0).sum())
            metrics[name]["nonzero_features"] = nz
            print(f"  best alpha: {pipe.named_steps['model'].alpha_:.4f}, "
                  f"non-zero coefs: {nz}")

    print("\nTest metrics:")
    for name, m in metrics.items():
        print(f"  {name:8s}  MAE={m['MAE']:<7} RMSE={m['RMSE']:<7} R2={m['R2']}")

    OUT.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models,
                 "numeric": NUMERIC_FEATURES,
                 "categorical": CATEGORICAL_FEATURES,
                 "target": TARGET}, OUT / "models.joblib")
    joblib.dump({"X_test": X_test, "y_test": y_test}, OUT / "test_data.joblib")
    (OUT / "metrics.json").write_text(json.dumps(metrics, indent=2))

    # Latest pre-innings stats per batter -> used by /api/predict.
    latest = pi.sort_values("start_date").groupby("striker").tail(1)
    players = {}
    for _, r in latest.iterrows():
        players[r["striker"]] = {
            "features": {f: float(r[f]) for f in NUMERIC_FEATURES},
            "career_avg": round(float(r["career_avg"]), 2),
            "career_sr": round(float(r["career_sr"]), 2),
            "last5_avg": round(float(r["last5_avg_runs"]), 2),
            "career_best": int(r["career_best"]),
            "innings_played": int(r["prior_innings"]),
            "team": r["batting_team"],
        }
    (OUT / "players.json").write_text(json.dumps(players))
    (OUT / "meta.json").write_text(json.dumps({
        "teams": sorted(pi["batting_team"].unique().tolist()),
        "venues": sorted(pi["venue"].unique().tolist()),
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "n_train": len(train), "n_test": len(test),
        "split_date": SPLIT_DATE,
    }, indent=2))
    print(f"\nArtifacts written to {OUT} in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
