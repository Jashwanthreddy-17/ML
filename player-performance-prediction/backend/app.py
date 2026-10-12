"""Flask backend for the Player Performance Prediction project.

Serves the HTML frontend and exposes a JSON API:

    GET  /api/meta      -> teams, venues, model metrics, feature lists
    GET  /api/players   -> batters with career stats (for the dropdown)
    POST /api/predict   -> predicted runs from Ridge and Lasso
                           body: {player, opponent, venue, innings}

Run:
    python train.py        # once, to build backend/artifacts/
    python app.py          # serves http://localhost:5000
"""

import json
from pathlib import Path

import joblib
import pandas as pd
from flask import Flask, jsonify, request, send_from_directory

HERE = Path(__file__).resolve().parent
ART = HERE / "artifacts"
FRONT = HERE.parent / "frontend"

bundle = joblib.load(ART / "models.joblib")
MODELS = bundle["models"]
NUMERIC = bundle["numeric"]
CATEGORICAL = bundle["categorical"]

PLAYERS = json.loads((ART / "players.json").read_text())
METRICS = json.loads((ART / "metrics.json").read_text())
META = json.loads((ART / "meta.json").read_text())

app = Flask(__name__)


@app.get("/")
def index():
    return send_from_directory(FRONT, "index.html")


@app.get("/<path:filename>")
def static_files(filename):
    return send_from_directory(FRONT, filename)


@app.get("/api/meta")
def meta():
    return jsonify({
        "teams": META["teams"],
        "venues": META["venues"],
        "metrics": METRICS,
        "n_train": META["n_train"],
        "n_test": META["n_test"],
        "split_date": META["split_date"],
        "numeric_features": NUMERIC,
    })


@app.get("/api/players")
def players():
    out = [{"name": name, **info} for name, info in PLAYERS.items()]
    out.sort(key=lambda p: p["career_avg"], reverse=True)
    return jsonify(out)


@app.post("/api/predict")
def predict():
    data = request.get_json(force=True)
    player = data.get("player")
    if player not in PLAYERS:
        return jsonify({"error": f"Unknown player: {player}"}), 400

    features = dict(PLAYERS[player]["features"])
    row = {
        **features,
        "innings": int(data.get("innings", 1)),
        "venue": data.get("venue", ""),
        "batting_team": PLAYERS[player]["team"],
        "bowling_team": data.get("opponent", ""),
    }
    X = pd.DataFrame([{c: row.get(c) for c in NUMERIC + CATEGORICAL}])

    preds = {}
    for name in ("ridge", "lasso"):
        pred = float(MODELS[name].predict(X)[0])
        preds[name] = round(max(pred, 0.0), 1)  # runs can't be negative

    info = PLAYERS[player]
    return jsonify({
        "player": player,
        "predictions": preds,
        "player_stats": {
            "career_avg": info["career_avg"],
            "career_sr": info["career_sr"],
            "last5_avg": info["last5_avg"],
            "career_best": info["career_best"],
            "innings_played": info["innings_played"],
            "team": info["team"],
        },
        "context": {
            "opponent": row["bowling_team"],
            "venue": row["venue"],
            "innings": row["innings"],
        },
    })


if __name__ == "__main__":
    app.run(debug=True)
