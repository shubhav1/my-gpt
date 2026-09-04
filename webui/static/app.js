// ---- helpers ----

function fmt(n, digits = 4) {
  if (n === null || n === undefined) return "—";
  return Number(n).toFixed(digits);
}

function fmtInt(n) {
  if (n === null || n === undefined) return "—";
  return Number(n).toLocaleString();
}

function fmtDate(ts) {
  if (!ts) return "";
  return new Date(ts * 1000).toLocaleString();
}

// Mirrors checkpoints._safe_name() on the backend, so a client-side name-collision check agrees
// with the server's actual enforcement instead of guessing at a different rule.
function sanitizeRunName(name) {
  const cleaned = (name || "").replace(/[^a-zA-Z0-9_-]/g, "_");
  return cleaned || "run";
}

function pointsFromHistory(history) {
  const stepPoints = [];
  const evalPoints = [];
  (history || []).forEach((e) => {
    if (e.type === "step") stepPoints.push({ iter: e.iter, loss: e.loss });
    else if (e.type === "eval") evalPoints.push(e);
  });
  return { stepPoints, evalPoints };
}

function findEvent(history, type) {
  return (history || []).find((e) => e.type === type) || null;
}

// Mirrors the CSS custom properties in styles.css -- canvas drawing can't read CSS variables
// directly, so these are kept in one place and reused by every chart (single-model and compare).
const CHART_COLORS = {
  grid: "#e2e4e9",
  axisText: "#6b7280",
  accent: "#2563eb",
  accent2: "#d97706",
};

function drawLossChart(canvas, stepPoints, evalPoints, emptyMessage) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 640;
  const cssHeight = canvas.clientHeight || 260;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const padding = { top: 10, right: 10, bottom: 10, left: 42 };
  const w = cssWidth - padding.left - padding.right;
  const h = cssHeight - padding.top - padding.bottom;

  const allLosses = stepPoints.map((p) => p.loss).concat(evalPoints.flatMap((p) => [p.train_loss, p.val_loss]));
  if (allLosses.length === 0) {
    ctx.fillStyle = CHART_COLORS.axisText;
    ctx.font = "13px sans-serif";
    ctx.fillText(emptyMessage, padding.left, cssHeight / 2);
    return;
  }

  const maxIter = Math.max(
    1,
    stepPoints.length ? stepPoints[stepPoints.length - 1].iter : 0,
    evalPoints.length ? evalPoints[evalPoints.length - 1].iter : 0
  );
  const minLoss = Math.min(...allLosses);
  const maxLoss = Math.max(...allLosses);
  const lossRange = maxLoss - minLoss || 1;

  const xFor = (iter) => padding.left + (iter / maxIter) * w;
  const yFor = (loss) => padding.top + h - ((loss - minLoss) / lossRange) * h;

  ctx.strokeStyle = CHART_COLORS.grid;
  ctx.fillStyle = CHART_COLORS.axisText;
  ctx.font = "11px sans-serif";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + w, y);
    ctx.stroke();
    const val = maxLoss - (lossRange / 4) * i;
    ctx.fillText(val.toFixed(2), 2, y + 3);
  }

  if (stepPoints.length) {
    ctx.strokeStyle = CHART_COLORS.accent;
    ctx.globalAlpha = 0.85;
    ctx.lineWidth = 1.25;
    ctx.beginPath();
    stepPoints.forEach((p, i) => {
      const x = xFor(p.iter);
      const y = yFor(p.loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.globalAlpha = 1;
  }

  if (evalPoints.length) {
    ctx.strokeStyle = CHART_COLORS.accent2;
    ctx.lineWidth = 2;
    ctx.beginPath();
    evalPoints.forEach((p, i) => {
      const x = xFor(p.iter);
      const y = yFor(p.val_loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = CHART_COLORS.accent2;
    evalPoints.forEach((p) => {
      const x = xFor(p.iter);
      const y = yFor(p.val_loss);
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  }
}

// ---- app state ----

let currentView = "train"; // "train" | "model"
let openRunName = null; // run name of the model currently open, if any
let openGroup = null; // that model's full group entry from modelsCache
let openCheckpoint = null; // { filename, location } of the currently selected iteration

// ---- sidebar (grouped by model, not by individual checkpoint file) ----

const newModelBtn = document.getElementById("new-model-btn");
const modelListEl = document.getElementById("model-list");

let modelsCache = [];

async function loadModels() {
  const res = await fetch("/api/models");
  modelsCache = await res.json();

  modelListEl.innerHTML = "";
  if (!modelsCache.length) {
    modelListEl.innerHTML = '<li class="sidebar-empty muted">No models yet — train one to get started.</li>';
  } else {
    modelsCache.forEach((group) => {
      const li = document.createElement("li");
      const btn = document.createElement("button");
      btn.type = "button";
      btn.className = "sidebar-item";

      const bestLoss =
        group.best_val_loss !== null && group.best_val_loss !== undefined
          ? group.best_val_loss.toFixed(3)
          : "—";
      const countLabel = `${group.total_count} checkpoint${group.total_count === 1 ? "" : "s"}`;
      const badge = group.saved_count === 0 ? '<span class="unsaved-badge">unsaved</span>' : "";

      btn.innerHTML =
        `<div class="sidebar-item-run">${group.run_name}${badge}</div>` +
        `<div class="sidebar-item-meta">${countLabel} · best val loss ${bestLoss}</div>`;

      if (currentView === "model" && openRunName === group.run_name) {
        btn.classList.add("is-active");
      }

      btn.addEventListener("click", () => openModel(group.run_name));
      li.appendChild(btn);
      modelListEl.appendChild(li);
    });
  }

  newModelBtn.classList.toggle("is-active", currentView === "train");
  compareBtn.classList.toggle("is-active", currentView === "compare");

  // Keep the compare dropdowns current if new checkpoints land while that view is open.
  if (currentView === "compare") populateCompareSelects();
}

// ---- view switching ----

const viewTrain = document.getElementById("view-train");
const viewModel = document.getElementById("view-model");
const viewCompare = document.getElementById("view-compare");
const compareBtn = document.getElementById("compare-btn");

function showTrainView() {
  currentView = "train";
  openRunName = null;
  openGroup = null;
  openCheckpoint = null;
  viewTrain.hidden = false;
  viewModel.hidden = true;
  viewCompare.hidden = true;
  // Only keep the chart/stats on screen if a run is actually live right now (trainingES open).
  // Otherwise this is a fresh visit to "Train new model" and last run's leftover curve/stats
  // would incorrectly look like something is already trained.
  if (!trainingES) {
    resetTrainUI();
    statEls.status.textContent = "idle";
    statEls.device.textContent = "—";
    statEls.iter.textContent = "—";
    statEls.trainLoss.textContent = "—";
    statEls.valLoss.textContent = "—";
    statEls.bpb.textContent = "—";
    statEls.ms.textContent = "—";
    statEls.params.textContent = "—";
    statEls.tpp.textContent = "—";
    trainError.textContent = "";
  }
  loadModels();
}

function showModelView() {
  currentView = "model";
  viewTrain.hidden = true;
  viewModel.hidden = false;
  viewCompare.hidden = true;
  loadModels();
}

function showCompareView() {
  currentView = "compare";
  viewTrain.hidden = true;
  viewModel.hidden = true;
  viewCompare.hidden = false;
  loadModels(); // also populates the compare selects, since currentView is now "compare"
}

newModelBtn.addEventListener("click", showTrainView);
compareBtn.addEventListener("click", showCompareView);

// ---- Train view ----

const trainForm = document.getElementById("train-form");
const startBtn = document.getElementById("start-btn");
const stopBtn = document.getElementById("stop-btn");
const trainError = document.getElementById("train-error");
const chart = document.getElementById("loss-chart");
const chartLoading = document.getElementById("chart-loading");

const statEls = {
  status: document.getElementById("stat-status"),
  device: document.getElementById("stat-device"),
  iter: document.getElementById("stat-iter"),
  trainLoss: document.getElementById("stat-train-loss"),
  valLoss: document.getElementById("stat-val-loss"),
  bpb: document.getElementById("stat-bpb"),
  ms: document.getElementById("stat-ms"),
  params: document.getElementById("stat-params"),
  tpp: document.getElementById("stat-tpp"),
};

let stepPoints = [];
let evalPoints = [];
let trainingES = null;

function suggestUniqueRunName(base) {
  const existing = new Set(modelsCache.map((m) => sanitizeRunName(m.run_name)));
  if (!existing.has(sanitizeRunName(base))) return base;
  let n = 2;
  let candidate = `${base}-${n}`;
  while (existing.has(sanitizeRunName(candidate))) {
    n += 1;
    candidate = `${base}-${n}`;
  }
  return candidate;
}

async function loadDefaultConfig() {
  const res = await fetch("/api/config");
  const cfg = await res.json();
  for (const [key, value] of Object.entries(cfg)) {
    const field = trainForm.elements.namedItem(key);
    if (!field) continue;
    if (field.type === "checkbox") field.checked = Boolean(value);
    else field.value = value;
  }
  const runNameField = trainForm.elements.namedItem("run_name");
  if (runNameField) runNameField.value = suggestUniqueRunName(runNameField.value);
}

function readTrainForm() {
  const data = new FormData(trainForm);
  return {
    run_name: data.get("run_name") || "webui-run",
    batch_size: Number(data.get("batch_size")),
    block_size: Number(data.get("block_size")),
    max_iters: Number(data.get("max_iters")),
    eval_interval: Number(data.get("eval_interval")),
    eval_iters: Number(data.get("eval_iters")),
    learning_rate: Number(data.get("learning_rate")),
    n_embd: Number(data.get("n_embd")),
    n_head: Number(data.get("n_head")),
    n_layer: Number(data.get("n_layer")),
    dropout: Number(data.get("dropout")),
    weight_decay: Number(data.get("weight_decay")),
    use_bpe: trainForm.elements.namedItem("use_bpe").checked,
    use_bf16: trainForm.elements.namedItem("use_bf16").checked,
  };
}

function showChartLoading() {
  chartLoading.classList.add("is-active");
}

function hideChartLoading() {
  chartLoading.classList.remove("is-active");
}

function resetTrainUI() {
  stepPoints = [];
  evalPoints = [];
  hideChartLoading();
  drawLossChart(chart, stepPoints, evalPoints, "Loss curve appears once training starts");
}

function setRunning(isRunning) {
  startBtn.disabled = isRunning;
  stopBtn.disabled = !isRunning;
}

function handleTrainEvent(event) {
  switch (event.type) {
    case "start":
      statEls.status.textContent = "running";
      statEls.device.textContent = event.device;
      break;
    case "params":
      statEls.params.textContent = fmtInt(event.params);
      statEls.tpp.textContent = event.tokens_per_param ? fmt(event.tokens_per_param, 2) : "—";
      break;
    case "step":
      hideChartLoading();
      stepPoints.push({ iter: event.iter, loss: event.loss });
      statEls.iter.textContent = fmtInt(event.iter);
      statEls.ms.textContent = fmt(event.ms_per_iter, 1);
      drawLossChart(chart, stepPoints, evalPoints, "Loss curve appears once training starts");
      break;
    case "eval":
      evalPoints.push(event);
      statEls.trainLoss.textContent = fmt(event.train_loss);
      statEls.valLoss.textContent = fmt(event.val_loss);
      statEls.bpb.textContent = event.val_bpb !== null && event.val_bpb !== undefined ? fmt(event.val_bpb) : "—";
      drawLossChart(chart, stepPoints, evalPoints, "Loss curve appears once training starts");
      break;
    case "checkpoint":
      // A new iteration of this model just landed -- reflect it in the sidebar right away,
      // wherever the user currently is.
      loadModels();
      break;
    case "error":
      hideChartLoading();
      trainError.textContent = event.message;
      statEls.status.textContent = "error";
      setRunning(false);
      loadModels();
      break;
    case "done":
    case "stopped":
      hideChartLoading();
      statEls.status.textContent = event.type;
      setRunning(false);
      loadModels();
      break;
  }
}

function connectTrainStream() {
  if (trainingES) trainingES.close();
  trainingES = new EventSource("/api/train/stream");
  trainingES.onmessage = (e) => {
    const event = JSON.parse(e.data);
    handleTrainEvent(event);
    if (["done", "stopped", "error"].includes(event.type)) {
      trainingES.close();
      trainingES = null;
    }
  };
  trainingES.onerror = () => {
    if (trainingES) {
      trainingES.close();
      trainingES = null;
    }
  };
}

trainForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  trainError.textContent = "";

  const values = readTrainForm();
  const collision = modelsCache.some(
    (m) => sanitizeRunName(m.run_name) === sanitizeRunName(values.run_name)
  );
  if (collision) {
    trainError.textContent = `a model named "${values.run_name}" already exists — choose a different name`;
    return;
  }

  resetTrainUI();
  setRunning(true);
  statEls.status.textContent = "starting…";
  showChartLoading();

  try {
    const res = await fetch("/api/train/start", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(values),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "failed to start training");
    connectTrainStream();
  } catch (err) {
    hideChartLoading();
    trainError.textContent = err.message;
    setRunning(false);
  }
});

stopBtn.addEventListener("click", async () => {
  stopBtn.disabled = true;
  await fetch("/api/train/stop", { method: "POST" });
});

window.addEventListener("resize", () => {
  if (currentView === "train") {
    drawLossChart(chart, stepPoints, evalPoints, "Loss curve appears once training starts");
  } else if (currentView === "model" && openCheckpoint) {
    drawLossChart(modelChart, modelStepPoints, modelEvalPoints, "No training history recorded for this checkpoint");
  } else if (currentView === "compare" && lastCompareA && lastCompareB) {
    drawCompareChart(compareChart, lastCompareA, lastCompareB);
  }
});

async function resyncTrain() {
  const res = await fetch("/api/train/status");
  const body = await res.json();
  // The server keeps the *last* run's status/history around indefinitely (useful for
  // /api/train/status as a record), but that's not the same as a run being live right now --
  // "done"/"stopped"/"error" could be from a run that finished long ago, in a totally different
  // session. Only an actually in-progress run is worth reconnecting to; anything else should
  // leave this fresh page load looking idle, since nothing has happened in it yet.
  if (body.status !== "running") return;

  resetTrainUI();
  statEls.status.textContent = "running";
  setRunning(true);
  showChartLoading();
  // The SSE endpoint replays full history before switching to live events, so this alone
  // reconstructs everything -- no need to also walk body.history here.
  connectTrainStream();
}

// ---- Model view ----

const modelTitle = document.getElementById("model-title");
const modelSubtitle = document.getElementById("model-subtitle");
const modelError = document.getElementById("model-error");
const modelArch = document.getElementById("model-arch");
const iterationPicker = document.getElementById("iteration-picker");
const iterationTitle = document.getElementById("iteration-title");
const modelSaveBtn = document.getElementById("model-save-btn");
const modelRenameBtn = document.getElementById("model-rename-btn");
const modelDeleteBtn = document.getElementById("model-delete-btn");
const modelChart = document.getElementById("model-loss-chart");
const chatHeading = document.getElementById("chat-heading");

const modelStatEls = {
  iter: document.getElementById("model-stat-iter"),
  trainLoss: document.getElementById("model-stat-train-loss"),
  valLoss: document.getElementById("model-stat-val-loss"),
  bpb: document.getElementById("model-stat-bpb"),
};

let modelStepPoints = [];
let modelEvalPoints = [];

let lastCompareA = null;
let lastCompareB = null;

function pickDefaultCheckpoint(checkpoints) {
  const withLoss = checkpoints.filter(
    (c) => c.stats && c.stats.val_loss !== undefined && c.stats.val_loss !== null
  );
  if (withLoss.length) {
    return withLoss.reduce((best, c) => (c.stats.val_loss < best.stats.val_loss ? c : best));
  }
  return checkpoints[checkpoints.length - 1]; // checkpoints[] is sorted ascending by step
}

function renderIterationPicker(group, activeFilename, activeLocation) {
  iterationPicker.innerHTML = "";
  group.checkpoints.forEach((c) => {
    const btn = document.createElement("button");
    btn.type = "button";
    btn.className = "iteration-btn";
    if (c.filename === activeFilename && c.location === activeLocation) btn.classList.add("is-active");

    const valLoss = c.stats && c.stats.val_loss !== undefined ? c.stats.val_loss.toFixed(3) : "—";
    const badge = c.location === "temp" ? '<span class="unsaved-badge">unsaved</span>' : "";
    btn.innerHTML =
      `<span class="iteration-btn-step">step ${c.step}</span>` +
      `<span class="iteration-btn-meta">val ${valLoss}${badge}</span>`;

    btn.addEventListener("click", () => selectIteration(c.filename, c.location));
    iterationPicker.appendChild(btn);
  });
}

async function openModel(runName, preferred) {
  modelError.textContent = "";

  const group = modelsCache.find((m) => m.run_name === runName);
  if (!group) {
    modelError.textContent = `no model named "${runName}"`;
    return;
  }

  openRunName = runName;
  openGroup = group;

  const bestLoss =
    group.best_val_loss !== null && group.best_val_loss !== undefined
      ? group.best_val_loss.toFixed(3)
      : "—";
  modelTitle.textContent = runName;
  modelSubtitle.textContent =
    `${group.total_count} checkpoint${group.total_count === 1 ? "" : "s"} of the same model` +
    ` · ${group.saved_count} saved · best val loss ${bestLoss}`;

  const target = preferred
    ? group.checkpoints.find((c) => c.filename === preferred.filename && c.location === preferred.location)
    : pickDefaultCheckpoint(group.checkpoints);

  renderIterationPicker(group, target.filename, target.location);
  await selectIteration(target.filename, target.location);

  showModelView();
}

async function selectIteration(filename, location) {
  modelError.textContent = "";

  let detail;
  try {
    const res = await fetch(
      `/api/checkpoints/detail?filename=${encodeURIComponent(filename)}&location=${encodeURIComponent(location)}`
    );
    detail = await res.json();
    if (!res.ok) throw new Error(detail.error || "failed to load this checkpoint");
  } catch (err) {
    modelError.textContent = err.message;
    return;
  }

  openCheckpoint = { filename: detail.filename, location: detail.location };

  document.querySelectorAll(".iteration-btn").forEach((btn) => btn.classList.remove("is-active"));
  if (openGroup) {
    const idx = openGroup.checkpoints.findIndex((c) => c.filename === filename && c.location === location);
    if (idx >= 0 && iterationPicker.children[idx]) iterationPicker.children[idx].classList.add("is-active");
  }

  iterationTitle.textContent = `Checkpoint — step ${detail.step}`;

  modelSaveBtn.hidden = detail.location !== "temp";
  modelSaveBtn.disabled = false;
  modelSaveBtn.textContent = "Save this checkpoint";

  const stats = detail.stats || {};
  modelStatEls.iter.textContent = fmtInt(stats.iter ?? detail.step);
  modelStatEls.trainLoss.textContent = fmt(stats.train_loss);
  modelStatEls.valLoss.textContent = fmt(stats.val_loss);
  modelStatEls.bpb.textContent =
    stats.val_bpb !== null && stats.val_bpb !== undefined ? fmt(stats.val_bpb) : "—";

  const startEvent = findEvent(detail.history, "start");
  const paramsEvent = findEvent(detail.history, "params");
  const mc = detail.model_config || {};
  modelArch.innerHTML = "";
  [
    ["Tokenizer", mc.use_bpe ? "BPE" : "byte-level"],
    ["Vocab size", fmtInt(mc.vocab_size)],
    ["n_embd", mc.n_embd],
    ["n_head", mc.n_head],
    ["n_layer", mc.n_layer],
    ["Block size", mc.block_size],
    ["Dropout", mc.dropout],
    ["Device", startEvent ? startEvent.device : "—"],
    ["Params", paramsEvent ? fmtInt(paramsEvent.params) : "—"],
  ].forEach(([label, value]) => {
    const dt = document.createElement("dt");
    dt.textContent = label;
    const dd = document.createElement("dd");
    dd.textContent = value === undefined || value === null ? "—" : value;
    modelArch.append(dt, dd);
  });

  const points = pointsFromHistory(detail.history);
  modelStepPoints = points.stepPoints;
  modelEvalPoints = points.evalPoints;
  drawLossChart(modelChart, modelStepPoints, modelEvalPoints, "No training history recorded for this checkpoint");

  chatMessagesEl.innerHTML = '<div class="chat-empty muted">Say something to this model.</div>';
  chatError.textContent = "";
  chatHeading.textContent = `Chat — ${detail.run_name || detail.filename} (step ${detail.step})`;
}

modelSaveBtn.addEventListener("click", async () => {
  if (!openCheckpoint || openCheckpoint.location !== "temp") return;

  modelSaveBtn.disabled = true;
  modelSaveBtn.textContent = "Saving…";

  try {
    const res = await fetch("/api/checkpoints/save", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ filename: openCheckpoint.filename }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "save failed");

    const runName = openRunName;
    await loadModels();
    await openModel(runName, { filename: body.filename, location: body.location });
  } catch (err) {
    modelSaveBtn.disabled = false;
    modelSaveBtn.textContent = "Save this checkpoint";
    modelError.textContent = err.message;
  }
});

modelRenameBtn.addEventListener("click", async () => {
  if (!openRunName) return;

  const newName = prompt("Rename this model to:", openRunName);
  if (newName === null) return;
  const trimmed = newName.trim();
  if (!trimmed || trimmed === openRunName) return;

  modelError.textContent = "";
  try {
    const res = await fetch("/api/models/rename", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_name: openRunName, new_run_name: trimmed }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "rename failed");

    await loadModels();
    await openModel(body.run_name);
  } catch (err) {
    modelError.textContent = err.message;
  }
});

modelDeleteBtn.addEventListener("click", async () => {
  if (!openRunName || !openGroup) return;

  const confirmed = confirm(
    `Delete "${openRunName}" and all ${openGroup.total_count} of its checkpoints? This cannot be undone.`
  );
  if (!confirmed) return;

  modelError.textContent = "";
  try {
    const res = await fetch("/api/models/delete", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ run_name: openRunName }),
    });
    const body = await res.json();
    if (!res.ok) throw new Error(body.error || "delete failed");
    showTrainView();
  } catch (err) {
    modelError.textContent = err.message;
  }
});

// ---- Chat (scoped to whichever iteration is currently selected) ----

const chatMessagesEl = document.getElementById("chat-messages");
const chatForm = document.getElementById("chat-form");
const chatInput = document.getElementById("chat-input");
const chatError = document.getElementById("chat-error");

function addBubble(role, text) {
  const empty = chatMessagesEl.querySelector(".chat-empty");
  if (empty) empty.remove();
  const div = document.createElement("div");
  div.className = `bubble ${role}`;
  div.textContent = text;
  chatMessagesEl.appendChild(div);
  chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
  return div;
}

chatForm.addEventListener("submit", async (e) => {
  e.preventDefault();
  chatError.textContent = "";

  if (!openCheckpoint) {
    chatError.textContent = "open a model first";
    return;
  }

  const prompt = chatInput.value;
  const maxTokens = Number(document.getElementById("chat-max-tokens").value) || 200;

  addBubble("user", prompt || "(empty prompt)");
  chatInput.value = "";
  const modelBubble = addBubble("model", "");

  try {
    const res = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        checkpoint: openCheckpoint.filename,
        location: openCheckpoint.location,
        prompt,
        max_new_tokens: maxTokens,
      }),
    });

    if (!res.ok || !res.body) {
      const body = await res.json().catch(() => ({}));
      throw new Error(body.error || "chat request failed");
    }

    const reader = res.body.getReader();
    const decoder = new TextDecoder();
    let buffer = "";

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;
      buffer += decoder.decode(value, { stream: true });

      const lines = buffer.split("\n\n");
      buffer = lines.pop();

      for (const line of lines) {
        if (!line.startsWith("data: ")) continue;
        const payload = JSON.parse(line.slice(6));
        if (payload.error) throw new Error(payload.error);
        if (payload.text) {
          modelBubble.textContent += payload.text;
          chatMessagesEl.scrollTop = chatMessagesEl.scrollHeight;
        }
      }
    }
  } catch (err) {
    chatError.textContent = err.message;
  }
});

// ---- Compare view ----
//
// Deliberately its own view behind its own sidebar button, reusing the same /api/models and
// /api/checkpoints/detail endpoints the model view already uses -- no new backend surface, and
// nothing about the existing Train/Model views changes to make room for this.

const compareSelectA = document.getElementById("compare-select-a");
const compareSelectB = document.getElementById("compare-select-b");
const compareError = document.getElementById("compare-error");
const compareResults = document.getElementById("compare-results");
const compareChart = document.getElementById("compare-chart");
const compareLegend = document.getElementById("compare-legend");
const compareTable = document.getElementById("compare-table");

function populateCompareSelects() {
  const previousA = compareSelectA.value;
  const previousB = compareSelectB.value;

  const options = modelsCache
    .map((m) => `<option value="${m.run_name}">${m.run_name}</option>`)
    .join("");
  compareSelectA.innerHTML = options;
  compareSelectB.innerHTML = options;

  compareError.textContent = "";

  if (modelsCache.length < 2) {
    compareResults.hidden = true;
    compareError.textContent = "train or save at least 2 models to compare them";
    return;
  }

  const names = modelsCache.map((m) => m.run_name);
  compareSelectA.value = names.includes(previousA) ? previousA : names[0];
  compareSelectB.value =
    names.includes(previousB) && previousB !== compareSelectA.value
      ? previousB
      : names.find((n) => n !== compareSelectA.value) || names[0];

  runComparison();
}

function finalCheckpointFor(runName) {
  const group = modelsCache.find((m) => m.run_name === runName);
  if (!group || !group.checkpoints.length) return null;
  return group.checkpoints[group.checkpoints.length - 1]; // checkpoints[] is sorted ascending by step
}

async function fetchCheckpointDetail(filename, location) {
  const res = await fetch(
    `/api/checkpoints/detail?filename=${encodeURIComponent(filename)}&location=${encodeURIComponent(location)}`
  );
  const body = await res.json();
  if (!res.ok) throw new Error(body.error || "failed to load checkpoint");
  return body;
}

function summarizeForCompare(detail) {
  const stats = detail.stats || {};
  const mc = detail.model_config || {};
  const startEvent = findEvent(detail.history, "start");
  const paramsEvent = findEvent(detail.history, "params");
  const points = pointsFromHistory(detail.history);
  return {
    runName: detail.run_name || detail.filename,
    step: detail.step,
    trainLoss: stats.train_loss,
    valLoss: stats.val_loss,
    valBpb: stats.val_bpb,
    device: startEvent ? startEvent.device : null,
    params: paramsEvent ? paramsEvent.params : null,
    tokensPerParam: paramsEvent ? paramsEvent.tokens_per_param : null,
    tokenizer: mc.use_bpe ? "BPE" : "byte-level",
    vocabSize: mc.vocab_size,
    nEmbd: mc.n_embd,
    nHead: mc.n_head,
    nLayer: mc.n_layer,
    blockSize: mc.block_size,
    dropout: mc.dropout,
    evalPoints: points.evalPoints,
  };
}

function drawCompareChart(canvas, a, b) {
  const ctx = canvas.getContext("2d");
  const dpr = window.devicePixelRatio || 1;
  const cssWidth = canvas.clientWidth || 640;
  const cssHeight = canvas.clientHeight || 280;
  canvas.width = cssWidth * dpr;
  canvas.height = cssHeight * dpr;
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, cssWidth, cssHeight);

  const padding = { top: 10, right: 10, bottom: 10, left: 42 };
  const w = cssWidth - padding.left - padding.right;
  const h = cssHeight - padding.top - padding.bottom;

  const series = [
    { ...a, color: CHART_COLORS.accent },
    { ...b, color: CHART_COLORS.accent2 },
  ];

  const allLosses = series.flatMap((s) => s.evalPoints.map((p) => p.val_loss));
  if (allLosses.length === 0) {
    ctx.fillStyle = CHART_COLORS.axisText;
    ctx.font = "13px sans-serif";
    ctx.fillText("Not enough recorded history to compare these two", padding.left, cssHeight / 2);
    return;
  }

  const maxIter = Math.max(
    1,
    ...series.map((s) => (s.evalPoints.length ? s.evalPoints[s.evalPoints.length - 1].iter : 0))
  );
  const minLoss = Math.min(...allLosses);
  const maxLoss = Math.max(...allLosses);
  const lossRange = maxLoss - minLoss || 1;

  const xFor = (iter) => padding.left + (iter / maxIter) * w;
  const yFor = (loss) => padding.top + h - ((loss - minLoss) / lossRange) * h;

  ctx.strokeStyle = CHART_COLORS.grid;
  ctx.fillStyle = CHART_COLORS.axisText;
  ctx.font = "11px sans-serif";
  ctx.lineWidth = 1;
  for (let i = 0; i <= 4; i++) {
    const y = padding.top + (h / 4) * i;
    ctx.beginPath();
    ctx.moveTo(padding.left, y);
    ctx.lineTo(padding.left + w, y);
    ctx.stroke();
    const val = maxLoss - (lossRange / 4) * i;
    ctx.fillText(val.toFixed(2), 2, y + 3);
  }

  series.forEach((s) => {
    if (!s.evalPoints.length) return;
    ctx.strokeStyle = s.color;
    ctx.lineWidth = 2;
    ctx.beginPath();
    s.evalPoints.forEach((p, i) => {
      const x = xFor(p.iter);
      const y = yFor(p.val_loss);
      if (i === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();

    ctx.fillStyle = s.color;
    s.evalPoints.forEach((p) => {
      const x = xFor(p.iter);
      const y = yFor(p.val_loss);
      ctx.beginPath();
      ctx.arc(x, y, 2.5, 0, Math.PI * 2);
      ctx.fill();
    });
  });
}

function renderCompareLegend(a, b) {
  compareLegend.innerHTML =
    `<span><i class="dot" style="background:${CHART_COLORS.accent}"></i> ${a.runName} — val loss</span>` +
    `<span><i class="dot" style="background:${CHART_COLORS.accent2}"></i> ${b.runName} — val loss</span>`;
}

function renderCompareTable(a, b) {
  const rows = [
    ["Final iteration", fmtInt(a.step), fmtInt(b.step), false],
    ["Train loss", fmt(a.trainLoss), fmt(b.trainLoss), true],
    ["Val loss", fmt(a.valLoss), fmt(b.valLoss), true],
    [
      "Val bpb",
      a.valBpb !== null && a.valBpb !== undefined ? fmt(a.valBpb) : "—",
      b.valBpb !== null && b.valBpb !== undefined ? fmt(b.valBpb) : "—",
      true,
    ],
    ["Params", fmtInt(a.params), fmtInt(b.params), false],
    [
      "Tokens/param",
      a.tokensPerParam ? fmt(a.tokensPerParam, 2) : "—",
      b.tokensPerParam ? fmt(b.tokensPerParam, 2) : "—",
      false,
    ],
    ["Device", a.device || "—", b.device || "—", false],
    ["Tokenizer", a.tokenizer, b.tokenizer, false],
    ["Vocab size", fmtInt(a.vocabSize), fmtInt(b.vocabSize), false],
    ["n_embd", a.nEmbd ?? "—", b.nEmbd ?? "—", false],
    ["n_head", a.nHead ?? "—", b.nHead ?? "—", false],
    ["n_layer", a.nLayer ?? "—", b.nLayer ?? "—", false],
    ["Block size", a.blockSize ?? "—", b.blockSize ?? "—", false],
    ["Dropout", a.dropout ?? "—", b.dropout ?? "—", false],
  ];

  let html = `<thead><tr><th>Metric</th><th>${a.runName}</th><th>${b.runName}</th></tr></thead><tbody>`;
  rows.forEach(([label, valA, valB, lowerIsBetter]) => {
    let classA = "";
    let classB = "";
    if (lowerIsBetter) {
      const numA = Number(valA);
      const numB = Number(valB);
      if (!Number.isNaN(numA) && !Number.isNaN(numB) && numA !== numB) {
        classA = numA < numB ? ' class="is-better"' : "";
        classB = numB < numA ? ' class="is-better"' : "";
      }
    }
    html += `<tr><td>${label}</td><td${classA}>${valA}</td><td${classB}>${valB}</td></tr>`;
  });
  html += "</tbody>";
  compareTable.innerHTML = html;
}

async function runComparison() {
  compareError.textContent = "";
  const runA = compareSelectA.value;
  const runB = compareSelectB.value;
  if (!runA || !runB) return;

  const ckptA = finalCheckpointFor(runA);
  const ckptB = finalCheckpointFor(runB);
  if (!ckptA || !ckptB) {
    compareResults.hidden = true;
    compareError.textContent = "couldn't find a checkpoint for one of these models";
    return;
  }

  try {
    const [detailA, detailB] = await Promise.all([
      fetchCheckpointDetail(ckptA.filename, ckptA.location),
      fetchCheckpointDetail(ckptB.filename, ckptB.location),
    ]);
    const a = summarizeForCompare(detailA);
    const b = summarizeForCompare(detailB);
    lastCompareA = a;
    lastCompareB = b;

    compareResults.hidden = false;
    drawCompareChart(compareChart, a, b);
    renderCompareLegend(a, b);
    renderCompareTable(a, b);
  } catch (err) {
    compareResults.hidden = true;
    compareError.textContent = err.message;
  }
}

compareSelectA.addEventListener("change", runComparison);
compareSelectB.addEventListener("change", runComparison);

// ---- session cleanup ----
//
// Browsers can't reliably distinguish "tab closed" from "page refreshed" -- both fire the same
// pagehide event -- so a refresh also wipes any not-yet-saved checkpoints. Save anything you
// want to keep before reloading.

function cleanupSession() {
  navigator.sendBeacon("/api/session/cleanup");
}

window.addEventListener("pagehide", cleanupSession);

// ---- init ----

async function init() {
  await loadModels();
  await loadDefaultConfig();
  resetTrainUI();
  resyncTrain();
}

init();
