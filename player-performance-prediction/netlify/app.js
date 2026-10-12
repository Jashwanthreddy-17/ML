// Frontend logic for the Player Performance Predictor (static build —
// predictions run in the browser via predict.js, no backend needed).

const $ = (id) => document.getElementById(id);
let ALL_PLAYERS = [];
let PLAYERS_BY_NAME = {};
let activeIdx = -1;

// ---- tabs ----
document.querySelectorAll(".tab").forEach((btn) => {
  btn.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((b) => b.classList.remove("active"));
    document.querySelectorAll(".panel").forEach((p) => p.classList.remove("active"));
    btn.classList.add("active");
    $("tab-" + btn.dataset.tab).classList.add("active");
  });
});

function option(value, text) {
  const o = document.createElement("option");
  o.value = value;
  o.textContent = text || value;
  return o;
}

async function init() {
  try {
    const [meta, metrics, ...shards] = await Promise.all([
      (await fetch("data/meta.json")).json(),
      (await fetch("data/metrics.json")).json(),
      (await fetch("data/players-0.json")).json(),
      (await fetch("data/players-1.json")).json(),
      (await fetch("data/players-2.json")).json(),
      (await fetch("data/players-3.json")).json(),
    ]);
    await loadModel(); // warm up predict.js

    const playersRaw = Object.assign({}, ...shards);

    PLAYERS_BY_NAME = playersRaw;
    ALL_PLAYERS = Object.entries(playersRaw)
      .map(([name, info]) => ({ name, ...info }))
      .sort((a, b) => b.career_avg - a.career_avg);

    $("player-count").textContent = `${ALL_PLAYERS.length} batters in the database — averages shown as you type`;
    wireAutocomplete();

    // Opponent + venue selects
    meta.teams.forEach((t) => $("opponent-select").appendChild(option(t)));
    meta.venues.forEach((v) => $("venue-select").appendChild(option(v)));

    renderMetrics(metrics, meta.split_date);
    renderTakeaways(metrics);
    $("about-train").textContent =
      meta.n_train.toLocaleString("en-IN") + " train / " +
      meta.n_test.toLocaleString("en-IN") + " test";
  } catch (e) {
    console.error("Failed to load site data:", e);
  }
}

// ---- player autocomplete with career averages ----
function wireAutocomplete() {
  const input = $("player-input");
  const box = $("suggestions");

  function close() {
    box.classList.add("hidden");
    activeIdx = -1;
  }

  function matches(q) {
    q = q.trim().toLowerCase();
    if (q.length < 1) return [];
    const starts = [], contains = [];
    for (const p of ALL_PLAYERS) {
      const n = p.name.toLowerCase();
      if (n.startsWith(q)) starts.push(p);
      else if (n.includes(q)) contains.push(p);
    }
    const byAvg = (a, b) => b.career_avg - a.career_avg;
    return [...starts.sort(byAvg), ...contains.sort(byAvg)].slice(0, 8);
  }

  function render(list) {
    if (!list.length) { close(); return; }
    box.innerHTML = list.map((p, i) => `
      <div class="suggestion${i === activeIdx ? " active" : ""}" data-i="${i}">
        <span>
          <span class="s-name">${p.name}</span><br>
          <span class="s-team">${p.team} · last-5 avg ${p.last5_avg}</span>
        </span>
        <span class="s-avg">avg ${p.career_avg}</span>
      </div>`).join("");
    box.classList.remove("hidden");
    box.querySelectorAll(".suggestion").forEach((el) => {
      el.addEventListener("mousedown", (e) => {
        e.preventDefault();
        pick(list[+el.dataset.i]);
      });
    });
  }

  function pick(p) {
    input.value = p.name;
    close();
  }

  let current = [];
  input.addEventListener("input", () => {
    activeIdx = -1;
    current = matches(input.value);
    render(current);
  });
  input.addEventListener("focus", () => {
    if (input.value.trim()) {
      current = matches(input.value);
      render(current);
    }
  });
  input.addEventListener("keydown", (e) => {
    if (box.classList.contains("hidden")) return;
    if (e.key === "ArrowDown" || e.key === "ArrowUp") {
      e.preventDefault();
      activeIdx = e.key === "ArrowDown"
        ? Math.min(activeIdx + 1, current.length - 1)
        : Math.max(activeIdx - 1, 0);
      render(current);
    } else if (e.key === "Enter" && activeIdx >= 0) {
      e.preventDefault();
      pick(current[activeIdx]);
    } else if (e.key === "Escape") {
      close();
    }
  });
  input.addEventListener("blur", () => setTimeout(close, 120));
}

// ---- prediction (in-browser, via predict.js) ----
$("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errEl = $("predict-error");
  errEl.classList.add("hidden");

  const name = $("player-input").value.trim();
  const entry = PLAYERS_BY_NAME[name];
  if (!entry) {
    errEl.textContent = "Please pick a player from the suggestions list.";
    errEl.classList.remove("hidden");
    return;
  }

  const btn = $("predict-btn");
  btn.disabled = true;
  btn.innerHTML = '<span class="spinner"></span> Predicting…';

  try {
    const opponent = $("opponent-select").value;
    const venue = $("venue-select").value;
    const innings = parseInt(document.querySelector('input[name="innings"]:checked').value, 10);

    const preds = await predictRuns(entry, opponent, venue, innings);
    const s = {
      career_avg: entry.career_avg,
      career_sr: entry.career_sr,
      last5_avg: entry.last5_avg,
      career_best: entry.career_best,
      innings_played: entry.innings_played,
      team: entry.team,
    };

    $("ridge-val").textContent = preds.ridge;
    $("lasso-val").textContent = preds.lasso;
    $("player-name").textContent = name;
    $("player-stats").innerHTML = [
      ["Career avg", s.career_avg],
      ["Career SR", s.career_sr],
      ["Last-5 avg", s.last5_avg],
      ["Career best", s.career_best],
      ["Innings played", s.innings_played],
      ["Team", s.team],
    ].map(([k, v]) => `<span class="stat">${k}: <b>${v}</b></span>`).join("");
    const dR = (preds.ridge - s.career_avg).toFixed(1);
    const dL = (preds.lasso - s.career_avg).toFixed(1);
    const fmt = (d) => `${d >= 0 ? "+" : ""}${d}`;
    $("delta-line").textContent =
      `vs career average of ${s.career_avg}: Ridge ${fmt(dR)}, Lasso ${fmt(dL)}`;
    $("context-line").textContent =
      `vs ${opponent} · ${venue} · ${innings === 1 ? "1st" : "2nd"} innings`;
    $("result").classList.remove("hidden");
    $("result").scrollIntoView({ behavior: "smooth", block: "nearest" });
  } catch (err) {
    errEl.textContent = err.message;
    errEl.classList.remove("hidden");
  } finally {
    btn.disabled = false;
    btn.textContent = "Predict runs";
  }
});

// ---- comparison ----
function renderMetrics(metrics, splitDate) {
  $("split-badge").textContent = `test ≥ ${splitDate}`;
  const rows = ["ridge", "lasso", "baseline"];
  const labels = { ridge: "Ridge (L2)", lasso: "Lasso (L1)", baseline: "Baseline (mean)" };
  const bestMae = Math.min(...rows.map((r) => metrics[r].MAE));
  $("metrics-body").innerHTML = rows.map((r) => {
    const m = metrics[r];
    const feat = r === "lasso" && m.nonzero_features != null
      ? `${m.nonzero_features} kept`
      : "all";
    return `<tr class="${m.MAE === bestMae ? "winner" : ""}">
      <td>${labels[r]}</td><td>${m.MAE}</td><td>${m.MSE}</td>
      <td>${m.RMSE}</td><td>${m.R2}</td><td>${feat}</td></tr>`;
  }).join("");

  if (typeof Chart !== "undefined") {
    new Chart($("error-chart"), {
      type: "bar",
      data: {
        labels: ["Ridge (L2)", "Lasso (L1)", "Baseline (mean)"],
        datasets: [
          { label: "MAE", data: rows.map((r) => metrics[r].MAE),
            backgroundColor: ["#1d7a44", "#d4a017", "#9aa5a0"] },
          { label: "RMSE", data: rows.map((r) => metrics[r].RMSE),
            backgroundColor: ["#14532d", "#7a5c00", "#6b7470"] },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: { title: { display: true, text: "Test error: lower is better" } },
        scales: { y: { beginAtZero: true, title: { display: true, text: "runs" } } },
      },
    });
  }
}

function renderTakeaways(metrics) {
  const items = [];
  const order = ["ridge", "lasso", "baseline"].sort((a, b) => metrics[a].MAE - metrics[b].MAE);
  const names = { ridge: "Ridge", lasso: "Lasso", baseline: "the mean baseline" };
  items.push(`<strong>${names[order[0]]}</strong> had the lowest test MAE (${metrics[order[0]].MAE} runs).`);
  if (metrics.lasso.nonzero_features != null) {
    items.push(`Lasso kept only <strong>${metrics.lasso.nonzero_features}</strong> features with non-zero coefficients — automatic feature selection among correlated predictors.`);
  }
  items.push(`R² of ${metrics[order[0]].R2} means the model explains about ${Math.round(metrics[order[0]].R2 * 100)}% of the variance in innings scores on unseen matches.`);
  items.push("All models were validated chronologically — trained on matches before 2022 and tested on 2022–2023, so no future information leaked into training.");
  $("takeaways").innerHTML = items.map((i) => `<li>${i}</li>`).join("");
}

init();
