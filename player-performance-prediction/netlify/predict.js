// In-browser prediction engine.
//
// Replicates the trained sklearn pipeline (StandardScaler on the numeric
// features + OneHotEncoder(handle_unknown="ignore") on the categoricals,
// followed by the Ridge / Lasso linear model) using the parameters exported
// to data/model.json. No backend needed.

let MODEL = null;

async function loadModel() {
  if (!MODEL) MODEL = await (await fetch("data/model.json")).json();
  return MODEL;
}

// Build the transformed feature vector in exactly the ColumnTransformer order:
// [scaled numerics (17)] + [one-hot(innings), one-hot(venue),
// one-hot(batting_team), one-hot(bowling_team)].
function buildFeatureVector(model, playerFeatures, innings, venue, battingTeam, bowlingTeam) {
  const num = model.numeric;
  const x = new Array(model.feature_order.length).fill(0);

  for (let i = 0; i < num.names.length; i++) {
    const v = playerFeatures[num.names[i]];
    x[i] = (v - num.mean[i]) / num.scale[i];
  }

  const ctx = {
    innings: String(innings),
    venue: String(venue),
    batting_team: String(battingTeam),
    bowling_team: String(bowlingTeam),
  };
  let j = num.names.length;
  for (const feat of ["innings", "venue", "batting_team", "bowling_team"]) {
    for (const cat of model.categorical[feat]) {
      x[j++] = String(cat) === ctx[feat] ? 1 : 0; // unknown -> all zeros
    }
  }
  return x;
}

function linearPredict(model, modelName, x) {
  const m = model.models[modelName];
  let s = m.intercept;
  const coef = m.coef;
  for (let i = 0; i < x.length; i++) s += coef[i] * x[i];
  return Math.max(0, Math.round(s * 10) / 10); // runs can't be negative
}

// Full prediction for one (player, opponent, venue, innings) combo.
// playerEntry: the player's record from data/players.json.
async function predictRuns(playerEntry, opponent, venue, innings) {
  const model = await loadModel();
  const x = buildFeatureVector(
    model,
    playerEntry.features,
    innings,
    venue,
    playerEntry.team,
    opponent
  );
  return {
    ridge: linearPredict(model, "ridge", x),
    lasso: linearPredict(model, "lasso", x),
  };
}
