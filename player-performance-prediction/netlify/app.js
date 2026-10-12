// Frontend logic for the Player Performance Predictor.

const $ = (id) => document.getElementById(id);
const esc = (s) => String(s).replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
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

// ---- chart lightbox ----
document.querySelectorAll(".fig-btn").forEach((b) => {
  b.addEventListener("click", () => {
    $("lb-img").src = b.dataset.src;
    $("lb-img").alt = b.dataset.alt;
    $("lightbox").showModal();
  });
});
$("lb-close").addEventListener("click", () => $("lightbox").close());
$("lightbox").addEventListener("click", (e) => { if (e.target === $("lightbox")) $("lightbox").close(); });

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

    $("player-count").textContent = `${ALL_PLAYERS.length} batters available. Career averages show as you type.`;
    wireAutocomplete();

    meta.teams.forEach((t) => $("opponent-select").appendChild(option(t)));
    meta.venues.forEach((v) => $("venue-select").appendChild(option(v)));

    const fmtN = (n) => n.toLocaleString("en-IN");
    $("data-chips").innerHTML = [
      `${fmtN(meta.n_train + meta.n_test)} player-innings`,
      `${ALL_PLAYERS.length} batters`,
      `${meta.numeric_features.length} numeric features`,
    ].map((t) => `<span class="chip">${t}</span>`).join("");

    renderMetrics(metrics, meta.split_date);
    renderKpis(metrics);
    renderTakeaways(metrics);
    renderFeatureChips(meta.numeric_features);
    $("about-train").textContent = `${fmtN(meta.n_train)} train / ${fmtN(meta.n_test)} test`;
  } catch (e) {
    console.error("Failed to load site data:", e);
    const err = $("predict-error");
    err.textContent = "Couldn't load the site data. Check your connection and reload.";
    err.classList.remove("hidden");
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
          <span class="s-name">${esc(p.name)}</span><br>
          <span class="s-team">${esc(p.team)} · last-5 avg ${p.last5_avg}</span>
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

// ---- prediction ----
const signed = (d) => `${d >= 0 ? "+" : "−"}${Math.abs(d).toFixed(1)}`;

function renderRuler(preds, s) {
  const marks = [
    { key: "career", label: "Career avg", value: s.career_avg },
    { key: "last5", label: "Last-5 avg", value: s.last5_avg },
    { key: "ridge", label: "Ridge", value: preds.ridge },
    { key: "lasso", label: "Lasso", value: preds.lasso },
  ];
  const top = Math.max(...marks.map((m) => m.value));
  const max = Math.max(40, Math.ceil((top * 1.25) / 10) * 10);
  const pct = (v) => Math.min(100, Math.max(0, (v / max) * 100));

  const step = max <= 60 ? 10 : 20;
  let ticks = "";
  for (let v = 0; v <= max; v += step) {
    ticks += `<span style="left:${pct(v)}%">${v}</span>`;
  }
  $("ruler-ticks").innerHTML = ticks;
  $("ruler-markers").innerHTML = marks.map((m) =>
    `<span class="marker ${m.key}" style="left:${pct(m.value)}%" title="${m.label}: ${m.value}"></span>`
  ).join("");
  $("ruler-legend").innerHTML = marks.map((m) =>
    `<li><span class="swatch ${m.key}"></span>${m.label} <b>${m.value}</b></li>`
  ).join("");
}

$("predict-form").addEventListener("submit", async (e) => {
  e.preventDefault();
  const errEl = $("predict-error");
  errEl.classList.add("hidden");

  const name = $("player-input").value.trim();
  const entry = PLAYERS_BY_NAME[name];
  if (!entry) {
    errEl.textContent = "Pick a batter from the suggestions list.";
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

    const p = await predictRuns(entry, opponent, venue, innings);
    const s = {
      career_avg: entry.career_avg,
      career_sr: entry.career_sr,
      last5_avg: entry.last5_avg,
      career_best: entry.career_best,
      innings_played: entry.innings_played,
      team: entry.team,
    };
    $("ridge-val").textContent = p.ridge;
    $("lasso-val").textContent = p.lasso;
    $("ridge-delta").textContent = `${signed(p.ridge - s.career_avg)} vs career avg`;
    $("lasso-delta").textContent = `${signed(p.lasso - s.career_avg)} vs career avg`;
    $("player-name").textContent = name;
    $("player-stats").innerHTML = [
      ["Career avg", s.career_avg],
      ["Career SR", s.career_sr],
      ["Last-5 avg", s.last5_avg],
      ["Career best", s.career_best],
      ["Innings played", s.innings_played],
      ["Team", s.team],
    ].map(([k, v]) => `<span class="stat">${k}: <b>${esc(v)}</b></span>`).join("");
    $("delta-line").textContent =
      `Against a career average of ${s.career_avg}, Ridge forecasts ${signed(p.ridge - s.career_avg)} and Lasso ${signed(p.lasso - s.career_avg)} runs.`;
    $("context-line").textContent =
      `vs ${opponent} · ${venue} · ${innings === 1 ? "1st" : "2nd"} innings`;
    renderRuler(p, s);

    $("empty-state").classList.add("hidden");
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
function renderKpis(metrics) {
  const best = ["ridge", "lasso"].sort((a, b) => metrics[a].MAE - metrics[b].MAE)[0];
  const bestName = best === "ridge" ? "Ridge" : "Lasso";
  const gain = (1 - metrics[best].MAE / metrics.baseline.MAE) * 100;
  const cards = [
    ["Best test MAE", `${metrics[best].MAE}`, `${bestName}, in runs`],
    ["Error cut vs mean baseline", `${gain.toFixed(1)}%`, `MAE ${metrics.baseline.MAE} → ${metrics[best].MAE}`],
    ["R² on unseen matches", `${metrics[best].R2}`, "variance explained"],
    ["Features kept by Lasso", metrics.lasso.nonzero_features != null ? `${metrics.lasso.nonzero_features}` : "–", "non-zero coefficients"],
  ];
  $("kpis").innerHTML = cards.map(([k, v, sub]) =>
    `<div class="kpi"><div class="kpi-label">${k}</div><div class="kpi-value">${v}</div><div class="kpi-sub">${sub}</div></div>`
  ).join("");
}

function renderMetrics(metrics, splitDate) {
  $("split-badge").textContent = `test from ${splitDate}`;
  const rows = ["ridge", "lasso", "baseline"];
  const labels = { ridge: "Ridge (L2)", lasso: "Lasso (L1)", baseline: "Baseline (mean)" };
  const bestMae = Math.min(...rows.map((r) => metrics[r].MAE));
  $("metrics-body").innerHTML = rows.map((r) => {
    const m = metrics[r];
    const feat = r === "lasso" && m.nonzero_features != null ? `${m.nonzero_features} kept` : "all";
    return `<tr class="${m.MAE === bestMae ? "winner" : ""}">
      <td>${labels[r]}</td><td>${m.MAE}</td><td>${m.MSE}</td>
      <td>${m.RMSE}</td><td>${m.R2}</td><td>${feat}</td></tr>`;
  }).join("");

  if (typeof Chart !== "undefined") {
    Chart.defaults.color = "#aab8b2";
    Chart.defaults.borderColor = "#26323a";
    Chart.defaults.font.family = '"Inter", "Segoe UI", system-ui, sans-serif';
    new Chart($("error-chart"), {
      type: "bar",
      data: {
        labels: ["Ridge (L2)", "Lasso (L1)", "Baseline (mean)"],
        datasets: [
          { label: "MAE", data: rows.map((r) => metrics[r].MAE),
            backgroundColor: ["#4cc38a", "#e3a73b", "#6b7a82"], borderRadius: 6 },
          { label: "RMSE", data: rows.map((r) => metrics[r].RMSE),
            backgroundColor: ["#1f6b4a", "#7a5a16", "#3d484e"], borderRadius: 6 },
        ],
      },
      options: {
        responsive: true, maintainAspectRatio: false,
        plugins: {
          title: { display: true, text: "Test error: lower is better", color: "#e6ece9" },
          legend: { labels: { color: "#aab8b2" } },
        },
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
    items.push(`Lasso kept only <strong>${metrics.lasso.nonzero_features}</strong> features with non-zero coefficients, which selects features automatically among correlated predictors.`);
  }
  items.push(`R² of ${metrics[order[0]].R2} means the model explains about ${Math.round(metrics[order[0]].R2 * 100)}% of the variance in innings scores on unseen matches. Single innings are noisy, so the gain over the baseline is modest.`);
  items.push("All models were validated chronologically: trained on matches before 2022 and tested on 2022–2023, so no future information leaked into training.");
  $("takeaways").innerHTML = items.map((i) => `<li>${i}</li>`).join("");
}

function renderFeatureChips(features) {
  $("feature-chips").innerHTML = features.map((f) => `<code>${esc(f)}</code>`).join("");
}

init();
