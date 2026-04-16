const statusEl = document.getElementById("status");
const fileInput = document.getElementById("file-input");
const dirInput = document.getElementById("dir-input");
const tokenizerInput = document.getElementById("tokenizer-input");
const overviewEl = document.getElementById("overview");
const detailEl = document.getElementById("detail");
const detailTitleEl = document.getElementById("detail-title");
const rowsEl = document.getElementById("rows");
const fileListEl = document.getElementById("file-list");
const policyRootInput = document.getElementById("policy-root-input");
const policyFileListEl = document.getElementById("policy-file-list");
const policyFileListTitleEl = document.getElementById("policy-file-list-title");
const policyRecordStatusEl = document.getElementById("policy-record-status");
const policyRecordOverviewEl = document.getElementById("policy-record-overview");
const policyRecordRowsEl = document.getElementById("policy-record-rows");
const policyOverviewTitleEl = document.getElementById("policy-overview-title");
const policyRecordDetailTitleEl = document.getElementById("policy-record-detail-title");
const policyAgent0ObservationEl = document.getElementById("policy-agent0-observation");
const policyAgent0ResponseEl = document.getElementById("policy-agent0-response");
const policyAgent0MessagesEl = document.getElementById("policy-agent0-messages");
const policyAgent0MetadataEl = document.getElementById("policy-agent0-metadata");
const policyAgent1ObservationEl = document.getElementById("policy-agent1-observation");
const policyAgent1ResponseEl = document.getElementById("policy-agent1-response");
const policyAgent1MessagesEl = document.getElementById("policy-agent1-messages");
const policyAgent1MetadataEl = document.getElementById("policy-agent1-metadata");
const policyRowIndexInput = document.getElementById("policy-row-index-input");
const policyPrevRowBtn = document.getElementById("policy-prev-row-btn");
const policyNextRowBtn = document.getElementById("policy-next-row-btn");
const policyJumpRowBtn = document.getElementById("policy-jump-row-btn");
const rewardCsvInput = document.getElementById("reward-csv-input");
const trainCsvInput = document.getElementById("train-csv-input");
const rewardChartGridEl = document.getElementById("reward-chart-grid");
const trainChartGridEl = document.getElementById("train-chart-grid");
const evalDirInput = document.getElementById("eval-dir-input");
const evalChartGridEl = document.getElementById("eval-chart-grid");
const rewardTailEl = document.getElementById("reward-tail");
const trainTailEl = document.getElementById("train-tail");
const evalTailEl = document.getElementById("eval-tail");
const rewardMetaEl = document.getElementById("reward-meta");
const trainMetaEl = document.getElementById("train-meta");
const evalMetaEl = document.getElementById("eval-meta");
const monitorStatusEl = document.getElementById("monitor-status");
const toggleMonitorBtn = document.getElementById("toggle-monitor-btn");
const presetButtons = Array.from(document.querySelectorAll(".preset-btn"));
const viewTabs = Array.from(document.querySelectorAll(".view-tab"));
const DEFAULT_POLICY_ROOT = "/home/zhangshuwen/Collab-Overcooked/runs/rl_policy_records";

let currentPath = fileInput.value;
let currentPolicyRecordPath = null;
let currentPolicyRecordRowCount = 0;
let currentPolicyRecordIndex = null;
let activePtRow = null;
let activePolicyRow = null;
let autoRefreshEnabled = true;
let monitorTimer = null;
let lastMonitorStamp = { reward: null, train: null, eval: null };
const CHART_COLORS = ["#a44b20", "#264653", "#2a9d8f", "#c96d3d", "#6d597a", "#d62828"];

function axisLabel(key) {
  if (key === "_round") {
    return "轮数";
  }
  if (key === "update_idx") {
    return "轮数";
  }
  if (key === "step") {
    return "步数";
  }
  if (key === "num_transitions") {
    return "样本数";
  }
  return key;
}

function setStatus(text, isError = false) {
  statusEl.textContent = text;
  statusEl.style.color = isError ? "#a42828" : "";
}

function prettyJson(value) {
  return JSON.stringify(value, null, 2);
}

function escapeHtml(value) {
  return String(value)
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}

function displayValue(value) {
  return value === undefined || value === null ? "" : String(value);
}

function displayPairCell(rowIndex, value, emptyText = "No row") {
  if (rowIndex === undefined || rowIndex === null) {
    return emptyText;
  }
  return displayValue(value);
}

function setMonitorStatus(text, isError = false) {
  monitorStatusEl.textContent = text;
  monitorStatusEl.style.color = isError ? "#a42828" : "";
}

function setPolicyRecordStatus(text, isError = false) {
  policyRecordStatusEl.textContent = text;
  policyRecordStatusEl.style.color = isError ? "#a42828" : "";
}

function ensurePolicyRootValue() {
  const current = (policyRootInput.value || "").trim();
  if (current) {
    return current;
  }
  policyRootInput.value = DEFAULT_POLICY_ROOT;
  return DEFAULT_POLICY_ROOT;
}

async function fetchJson(url) {
  const response = await fetch(url);
  const data = await response.json();
  if (!response.ok) {
    throw new Error(data.detail || "Request failed");
  }
  return data;
}

function renderFiles(files) {
  fileListEl.innerHTML = "";
  if (!files.length) {
    fileListEl.textContent = "目录下没有找到 .pt 文件";
    return;
  }

  files.forEach((file) => {
    const item = document.createElement("button");
    item.className = "file-item";
    item.textContent = `${file.name} (${(file.size_bytes / 1024).toFixed(1)} KB)`;
    item.addEventListener("click", () => {
      fileInput.value = file.path;
      loadFile(file.path);
    });
    fileListEl.appendChild(item);
  });
}

function renderRows(rows) {
  rowsEl.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(displayValue(row.index))}</td>
      <td>${escapeHtml(displayValue(row.agent_index))}</td>
      <td>${escapeHtml(displayValue(row.timestep))}</td>
      <td>${escapeHtml(displayValue(row.reward))}</td>
      <td>${escapeHtml(displayValue(row.value))}</td>
      <td>${escapeHtml(displayValue(row.log_prob))}</td>
      <td>${escapeHtml(displayValue(row.done))}</td>
      <td>${escapeHtml(displayValue(row.prompt_len))}</td>
      <td>${escapeHtml(displayValue(row.response_len))}</td>
      <td>${escapeHtml(displayValue(row.critic_len))}</td>
    `;
    tr.addEventListener("click", () => loadItem(row.index, tr));
    rowsEl.appendChild(tr);
  });
}

function renderPolicyFiles(files) {
  policyFileListEl.innerHTML = "";
  if (!files.length) {
    policyFileListEl.textContent = "目录下没有找到 .jsonl 文件";
    return;
  }

  files.forEach((file) => {
    const item = document.createElement("button");
    item.className = "file-item";
    item.innerHTML = `
      <span class="file-item-path">${escapeHtml(displayValue(file.relative_path || file.name))}</span>
      <span class="file-item-meta">${escapeHtml(displayValue(file.session_name))} · ${(file.size_bytes / 1024).toFixed(1)} KB</span>
    `;
    item.addEventListener("click", () => loadPolicyRecordFile(file.path));
    policyFileListEl.appendChild(item);
  });
}

function renderPolicyRecordRows(rows) {
  policyRecordRowsEl.innerHTML = "";
  rows.forEach((row) => {
    const tr = document.createElement("tr");
    tr.innerHTML = `
      <td>${escapeHtml(displayValue(row.index))}</td>
      <td>${escapeHtml(displayValue(row.timestep))}</td>
      <td>${escapeHtml(displayValue(row.slot_index))}</td>
      <td>${escapeHtml(displayValue(row.event_kind))}</td>
      <td>${escapeHtml(displayPairCell(row.agent0_row_index, row.agent0_row_index))}</td>
      <td>${escapeHtml(displayPairCell(row.agent0_row_index, row.agent0_call_type))}</td>
      <td>${escapeHtml(displayPairCell(row.agent0_row_index, row.agent0_reward))}</td>
      <td>${escapeHtml(displayPairCell(row.agent0_row_index, row.agent0_observation_preview))}</td>
      <td>${escapeHtml(displayPairCell(row.agent1_row_index, row.agent1_row_index))}</td>
      <td>${escapeHtml(displayPairCell(row.agent1_row_index, row.agent1_call_type))}</td>
      <td>${escapeHtml(displayPairCell(row.agent1_row_index, row.agent1_reward))}</td>
      <td>${escapeHtml(displayPairCell(row.agent1_row_index, row.agent1_observation_preview))}</td>
    `;
    tr.addEventListener("click", () => loadPolicyRecordItem(row.index, tr));
    policyRecordRowsEl.appendChild(tr);
  });
}

function renderAgentDetail(prefix, detail) {
  const observationEl =
    prefix === "agent0" ? policyAgent0ObservationEl : policyAgent1ObservationEl;
  const responseEl = prefix === "agent0" ? policyAgent0ResponseEl : policyAgent1ResponseEl;
  const messagesEl = prefix === "agent0" ? policyAgent0MessagesEl : policyAgent1MessagesEl;
  const metadataEl = prefix === "agent0" ? policyAgent0MetadataEl : policyAgent1MetadataEl;
  if (!detail) {
    observationEl.textContent = "";
    responseEl.textContent = "";
    messagesEl.textContent = "No matched row";
    metadataEl.textContent = "";
    return;
  }
  observationEl.textContent = detail.observation || detail.prompt || "";
  responseEl.textContent = detail.response || "";
  messagesEl.textContent = prettyJson(detail.messages || []);
  metadataEl.textContent = prettyJson({
    row_index: detail.index,
    call_type: detail.call_type,
    reward: detail.reward,
    done: detail.done,
    reward_breakdown: detail.reward_breakdown || {},
    metadata: detail.metadata || {},
  });
}

function clearPolicyRecordDetail() {
  currentPolicyRecordIndex = null;
  policyRecordDetailTitleEl.textContent = "未选择";
  policyRowIndexInput.value = "";
  renderAgentDetail("agent0", null);
  renderAgentDetail("agent1", null);
}

function updatePolicyRowControls() {
  const hasSelection = Number.isInteger(currentPolicyRecordIndex);
  const hasRows = currentPolicyRecordRowCount > 0;
  policyRowIndexInput.disabled = !hasRows;
  policyJumpRowBtn.disabled = !hasRows;
  policyPrevRowBtn.disabled = !hasSelection || currentPolicyRecordIndex <= 0;
  policyNextRowBtn.disabled =
    !hasSelection || currentPolicyRecordIndex >= currentPolicyRecordRowCount - 1;
}

function switchView(targetId, scope = document) {
  const tabs = Array.from(scope.querySelectorAll(".view-tab")).filter(
    (tab) => tab.closest(".view-scope") === scope
  );
  const panels = Array.from(scope.querySelectorAll(".view-panel")).filter(
    (panel) => panel.closest(".view-scope") === scope
  );
  tabs.forEach((tab) => {
    tab.classList.toggle("active", tab.dataset.viewTarget === targetId);
  });
  panels.forEach((panel) => {
    panel.classList.toggle("active", panel.id === targetId);
  });
}

function buildPath(points, bounds) {
  return points
    .map((point, index) => {
      const x = scaleX(point.x, bounds);
      const y =
        scaleY(point.y, bounds);
      return `${index === 0 ? "M" : "L"} ${x.toFixed(2)} ${y.toFixed(2)}`;
    })
    .join(" ");
}

function scaleX(x, bounds) {
  return bounds.left + ((x - bounds.xMin) / Math.max(bounds.xMax - bounds.xMin, 1e-9)) * bounds.width;
}

function scaleY(y, bounds) {
  return (
    bounds.top +
    bounds.height -
    ((y - bounds.yMin) / Math.max(bounds.yMax - bounds.yMin, 1e-9)) * bounds.height
  );
}

function integerTicks(min, max) {
  const ticks = [];
  const start = Math.ceil(min);
  const end = Math.floor(max);
  for (let value = start; value <= end; value += 1) {
    ticks.push(value);
  }
  if (!ticks.length) {
    ticks.push(min, max);
  }
  return Array.from(new Set(ticks));
}

function attachTooltip(container) {
  let tooltip = container.querySelector(".chart-tooltip");
  if (!tooltip) {
    tooltip = document.createElement("div");
    tooltip.className = "chart-tooltip";
    tooltip.hidden = true;
    container.appendChild(tooltip);
  }
  return tooltip;
}

function renderChart(container, payload, titleMetaEl) {
  const allPoints = payload.series.flatMap((series) => series.points);
  if (!allPoints.length) {
    container.innerHTML = "<div class='muted'>暂无可绘制数据</div>";
    if (titleMetaEl) {
      titleMetaEl.textContent = "";
    }
    return;
  }

  const xVals = allPoints.map((point) => point.x);
  const yVals = allPoints.map((point) => point.y);
  const bounds = {
    left: 46,
    top: 16,
    width: 700,
    height: 236,
    xMin: Math.min(...xVals),
    xMax: Math.max(...xVals),
    yMin: Math.min(...yVals),
    yMax: Math.max(...yVals),
  };
  if (bounds.yMin === bounds.yMax) {
    bounds.yMin -= 1;
    bounds.yMax += 1;
  }

  const circles = payload.series
    .map((series, idx) => {
      const color = CHART_COLORS[idx % CHART_COLORS.length];
      return series.points
        .map((point) => {
          const x = scaleX(point.x, bounds);
          const y = scaleY(point.y, bounds);
          const label = `${series.name}\n轮数: ${point.x}\n值: ${point.y.toFixed(6)}`;
          return `<circle cx="${x.toFixed(2)}" cy="${y.toFixed(2)}" r="4.5" fill="${color}" data-tip="${label}"></circle>`;
        })
        .join("");
    })
    .join("");

  const lines = payload.series
    .map((series, idx) => {
      const color = CHART_COLORS[idx % CHART_COLORS.length];
      const d = buildPath(series.points, bounds);
      return `<path d="${d}" fill="none" stroke="${color}" stroke-width="2.5" stroke-linecap="round" />`;
    })
    .join("");

  const legend = payload.series
    .map((series, idx) => {
      const color = CHART_COLORS[idx % CHART_COLORS.length];
      return `<span class="legend-item"><span class="legend-swatch" style="background:${color}"></span>${series.name}</span>`;
    })
    .join("");

  const yTicks = [0, 0.25, 0.5, 0.75, 1]
    .map((ratio) => {
      const value = bounds.yMin + (bounds.yMax - bounds.yMin) * (1 - ratio);
      const y = bounds.top + bounds.height * ratio;
      return `
        <line x1="${bounds.left}" y1="${y}" x2="${bounds.left + bounds.width}" y2="${y}" stroke="rgba(0,0,0,0.08)" />
        <text x="${bounds.left - 8}" y="${y + 4}" text-anchor="end" font-size="11" fill="#7b6956">${value.toFixed(3)}</text>
      `;
    })
    .join("");

  const xTicks = integerTicks(bounds.xMin, bounds.xMax)
    .map((value) => {
      const x = scaleX(value, bounds);
      return `
        <line x1="${x}" y1="${bounds.top}" x2="${x}" y2="${bounds.top + bounds.height}" stroke="rgba(0,0,0,0.05)" />
        <text x="${x}" y="${bounds.top + bounds.height + 18}" text-anchor="middle" font-size="11" fill="#7b6956">${value}</text>
      `;
    })
    .join("");

  container.innerHTML = `
    <div class="chart-legend">${legend}</div>
    <svg viewBox="0 0 780 280" preserveAspectRatio="none" aria-label="line chart">
      <rect x="${bounds.left}" y="${bounds.top}" width="${bounds.width}" height="${bounds.height}" fill="rgba(255,255,255,0.35)" rx="12"></rect>
      ${yTicks}
      ${xTicks}
      <line x1="${bounds.left}" y1="${bounds.top + bounds.height}" x2="${bounds.left + bounds.width}" y2="${bounds.top + bounds.height}" stroke="rgba(0,0,0,0.28)" />
      <line x1="${bounds.left}" y1="${bounds.top}" x2="${bounds.left}" y2="${bounds.top + bounds.height}" stroke="rgba(0,0,0,0.28)" />
      ${lines}
      ${circles}
      <text x="${bounds.left + bounds.width / 2}" y="275" text-anchor="middle" font-size="12" fill="#7b6956">${axisLabel(payload.x_key)}</text>
    </svg>
  `;

  const tooltip = attachTooltip(container);
  container.querySelectorAll("circle[data-tip]").forEach((node) => {
    node.addEventListener("mouseenter", (event) => {
      tooltip.hidden = false;
      tooltip.textContent = event.target.dataset.tip;
    });
    node.addEventListener("mousemove", (event) => {
      const rect = container.getBoundingClientRect();
      tooltip.style.left = `${event.clientX - rect.left}px`;
      tooltip.style.top = `${event.clientY - rect.top}px`;
    });
    node.addEventListener("mouseleave", () => {
      tooltip.hidden = true;
    });
  });

  if (titleMetaEl) {
    titleMetaEl.textContent = `${payload.series.length} metrics`;
  }
}

function renderChartGroups(container, groups, titleMetaEl) {
  container.innerHTML = "";
  groups.forEach((group) => {
    const panel = document.createElement("section");
    panel.className = "subchart-panel";
    const title = document.createElement("h3");
    title.className = "subchart-title";
    title.textContent = group.title;
    panel.appendChild(title);
    if (group.charts?.length) {
      const stack = document.createElement("div");
      stack.className = "chart-stack";
      group.charts.forEach((subchart) => {
        const subpanel = document.createElement("div");
        subpanel.className = "chart-subpanel";
        const subtitle = document.createElement("h4");
        subtitle.className = "chart-subtitle";
        subtitle.textContent = subchart.title;
        const chart = document.createElement("div");
        chart.className = "chart";
        subpanel.appendChild(subtitle);
        subpanel.appendChild(chart);
        stack.appendChild(subpanel);
        renderChart(chart, subchart.chart, null);
      });
      panel.appendChild(stack);
    } else {
      const chart = document.createElement("div");
      chart.className = "chart";
      panel.appendChild(chart);
      renderChart(chart, group.chart, null);
    }
    container.appendChild(panel);
  });
  if (titleMetaEl) {
    const chartCount = groups.reduce((count, group) => count + (group.charts?.length || 1), 0);
    titleMetaEl.textContent = `${chartCount} charts`;
  }
}

function mergeEvalGroups(data) {
  const merged = [];
  if (data.performance?.groups?.length) {
    data.performance.groups.forEach((group) => {
      merged.push({ title: `[Performance] ${group.title}`, chart: group.chart, charts: group.charts });
    });
  }
  if (data.episode?.groups?.length) {
    data.episode.groups.forEach((group) => {
      merged.push({ title: `[Episode] ${group.title}`, chart: group.chart, charts: group.charts });
    });
  }
  if (data.reward?.groups?.length) {
    data.reward.groups.forEach((group) => {
      merged.push({ title: `[Reward] ${group.title}`, chart: group.chart, charts: group.charts });
    });
  }
  return merged;
}

function buildEvalTail(data) {
  return prettyJson({
    eval_dir: data.eval_dir,
    performance_tail: data.performance?.rows?.slice(-5) ?? [],
    episode_tail: data.episode?.rows?.slice(-5) ?? [],
    reward_tail: data.reward?.rows?.slice(-5) ?? [],
  });
}

function applyPreset(button) {
  const kind = button.dataset.presetKind;
  if (kind === "train") {
    rewardCsvInput.value = button.dataset.rewardPath || rewardCsvInput.value;
    trainCsvInput.value = button.dataset.trainPath || trainCsvInput.value;
  } else if (kind === "eval") {
    evalDirInput.value = button.dataset.evalDir || evalDirInput.value;
  }
}

function renderPanelError(container, metaEl, tailEl, title, message) {
  container.innerHTML = `<section class="subchart-panel"><h3 class="subchart-title">${title}</h3><div class="muted">${message}</div></section>`;
  if (metaEl) {
    metaEl.textContent = "error";
  }
  if (tailEl) {
    tailEl.textContent = message;
  }
}

function isUnavailableEvalMessage(message) {
  const text = String(message || "");
  return (
    text.includes("No supported eval CSV files found") ||
    text.includes("Path not found")
  );
}

async function refreshMonitor(force = false) {
  try {
    const [monitorResult, evalResult] = await Promise.allSettled([
      fetchJson(
        `/api/monitor?reward_path=${encodeURIComponent(rewardCsvInput.value)}&train_path=${encodeURIComponent(trainCsvInput.value)}`
      ),
      fetchJson(`/api/eval_monitor?eval_dir=${encodeURIComponent(evalDirInput.value)}`),
    ]);

    let hasChange = !!force;
    let latestMtime = 0;
    const statusParts = [];
    const errors = [];

    if (monitorResult.status === "fulfilled") {
      const data = monitorResult.value;
      const monitorChanged =
        force ||
        data.reward.mtime !== lastMonitorStamp.reward ||
        data.train.mtime !== lastMonitorStamp.train;
      if (monitorChanged) {
        renderChartGroups(rewardChartGridEl, data.reward.groups, rewardMetaEl);
        renderChartGroups(trainChartGridEl, data.train.groups, trainMetaEl);
        rewardTailEl.textContent = prettyJson(data.reward.rows.slice(-10));
        trainTailEl.textContent = prettyJson(data.train.rows.slice(-10));
        lastMonitorStamp.reward = data.reward.mtime;
        lastMonitorStamp.train = data.train.mtime;
        latestMtime = Math.max(latestMtime, data.reward.mtime, data.train.mtime);
        hasChange = true;
      }
      statusParts.push(`reward(${data.reward.row_count})`);
      statusParts.push(`train(${data.train.row_count})`);
    } else {
      const message = monitorResult.reason?.message || "monitor request failed";
      renderPanelError(rewardChartGridEl, rewardMetaEl, rewardTailEl, "Reward Monitor", message);
      renderPanelError(trainChartGridEl, trainMetaEl, trainTailEl, "Train Monitor", message);
      errors.push(`monitor: ${message}`);
    }

    if (evalResult.status === "fulfilled") {
      const evalData = evalResult.value;
      const evalChanged = force || evalData.mtime !== lastMonitorStamp.eval;
      if (evalChanged) {
        renderChartGroups(evalChartGridEl, mergeEvalGroups(evalData), evalMetaEl);
        evalTailEl.textContent = buildEvalTail(evalData);
        lastMonitorStamp.eval = evalData.mtime;
        latestMtime = Math.max(latestMtime, evalData.mtime);
        hasChange = true;
      }
      statusParts.push(`eval(${evalData.eval_dir})`);
    } else {
      const message = evalResult.reason?.message || "eval monitor request failed";
      if (isUnavailableEvalMessage(message)) {
        renderPanelError(evalChartGridEl, evalMetaEl, evalTailEl, "Eval Monitor", "当前目录没有评估 CSV，可切换到其它 eval / collect 结果目录。");
        statusParts.push("eval(unavailable)");
      } else {
        renderPanelError(evalChartGridEl, evalMetaEl, evalTailEl, "Eval Monitor", message);
        errors.push(`eval: ${message}`);
      }
    }

    if (!hasChange && !errors.length) {
      setMonitorStatus("监控中，文件未变化");
      return;
    }

    if (errors.length) {
      const suffix =
        latestMtime > 0
          ? `，最近成功更新时间 ${new Date(latestMtime * 1000).toLocaleString()}`
          : "";
      setMonitorStatus(`${statusParts.join(" / ")}${statusParts.length ? "；" : ""}${errors.join(" ; ")}${suffix}`, true);
      return;
    }

    setMonitorStatus(
      `已更新 ${statusParts.join(" / ")}，最近修改 ${new Date(latestMtime * 1000).toLocaleString()}`
    );
  } catch (error) {
    const message = error?.message || String(error);
    setMonitorStatus(`监控启动失败: ${message}`, true);
  }
}

function scheduleMonitor() {
  if (monitorTimer) {
    clearInterval(monitorTimer);
  }
  monitorTimer = setInterval(() => {
    if (autoRefreshEnabled) {
      refreshMonitor(false);
    }
  }, 4000);
}

async function browseFiles() {
  setStatus("正在扫描目录...");
  try {
    const data = await fetchJson(`/api/list?dir_path=${encodeURIComponent(dirInput.value)}`);
    renderFiles(data.files);
    setStatus(`已找到 ${data.files.length} 个 .pt 文件`);
  } catch (error) {
    setStatus(error.message, true);
  }
}

async function browsePolicyFiles(autoSelect = false) {
  const rootPath = ensurePolicyRootValue();
  setPolicyRecordStatus("正在扫描全部 policy record 文件...");
  try {
    const data = await fetchJson(
      `/api/policy_records/all_files?dir_path=${encodeURIComponent(rootPath)}`
    );
    renderPolicyFiles(data.files);
    policyFileListTitleEl.textContent = `${data.files.length} files`;
    setPolicyRecordStatus(`已找到 ${data.files.length} 个 policy record 文件`);
    if (autoSelect && data.files.length) {
      await loadPolicyRecordFile(data.files[0].path);
    }
  } catch (error) {
    const message =
      error.message === "Not Found"
        ? "后端接口 /api/policy_records/all_files 不存在。ptWeb 服务还在旧版本，请重启 ptWeb 后再试。"
        : error.message;
    setPolicyRecordStatus(message, true);
  }
}

async function loadPolicyRecordFile(path) {
  const targetPath = path || currentPolicyRecordPath;
  if (!targetPath) {
    setPolicyRecordStatus("请先选择 policy record 文件", true);
    return;
  }
  currentPolicyRecordPath = targetPath;
  currentPolicyRecordRowCount = 0;
  policyRecordOverviewEl.textContent = "";
  policyRecordRowsEl.innerHTML = "";
  clearPolicyRecordDetail();
  setPolicyRecordStatus("正在读取 policy record 文件...");
  try {
    const data = await fetchJson(`/api/policy_records/paired_file?path=${encodeURIComponent(targetPath)}`);
    policyRecordOverviewEl.textContent = prettyJson(data.overview);
    policyOverviewTitleEl.textContent =
      data.overview.session_dir || data.path;
    renderPolicyRecordRows(data.rows);
    currentPolicyRecordRowCount = data.rows.length;
    updatePolicyRowControls();
    setPolicyRecordStatus(`已加载 ${data.rows.length} 条 policy record`);
    if (data.rows.length > 0) {
      await loadPolicyRecordItem(0);
    }
  } catch (error) {
    policyOverviewTitleEl.textContent = "未选择";
    setPolicyRecordStatus(error.message, true);
  }
}

async function loadPolicyRecordItem(index, rowEl = null) {
  if (index === undefined || index === null || !currentPolicyRecordPath) {
    return;
  }
  try {
    const data = await fetchJson(
      `/api/policy_records/paired_item?path=${encodeURIComponent(currentPolicyRecordPath)}&index=${index}`
    );
    const detail = data.detail;
    currentPolicyRecordIndex = index;
    policyRowIndexInput.value = String(index);
    policyRecordDetailTitleEl.textContent =
      `${displayValue(detail.event_kind)} / event ${index} / timestep ${displayValue(detail.timestep)} / slot ${displayValue(detail.slot_index)}`;
    renderAgentDetail("agent0", detail.agent0 || null);
    renderAgentDetail("agent1", detail.agent1 || null);
    if (activePolicyRow) {
      activePolicyRow.classList.remove("active");
    }
    activePolicyRow = rowEl || policyRecordRowsEl.children[index] || null;
    if (activePolicyRow) {
      activePolicyRow.classList.add("active");
    }
    updatePolicyRowControls();
  } catch (error) {
    setPolicyRecordStatus(error.message, true);
  }
}

async function loadFile(path) {
  const targetPath = path || fileInput.value;
  setStatus("正在读取文件...");
  currentPath = targetPath;
  detailEl.textContent = "";
  detailTitleEl.textContent = "未选择";
  try {
    const data = await fetchJson(`/api/file?path=${encodeURIComponent(targetPath)}`);
    overviewEl.textContent = prettyJson(data.overview);
    renderRows(data.rows);
    setStatus(`已加载 ${data.rows.length} 条样本`);
    if (data.rows.length > 0) {
      loadItem(0);
    }
  } catch (error) {
    overviewEl.textContent = "";
    rowsEl.innerHTML = "";
    setStatus(error.message, true);
  }
}

async function loadItem(index, rowEl = null) {
  if (index === undefined || index === null) {
    return;
  }
  try {
    const data = await fetchJson(
      `/api/item?path=${encodeURIComponent(currentPath)}&index=${index}&tokenizer_path=${encodeURIComponent(tokenizerInput.value)}`
    );
    detailEl.textContent = prettyJson(data.detail);
    detailTitleEl.textContent = `index ${index}`;
    if (activePtRow) {
      activePtRow.classList.remove("active");
    }
    activePtRow = rowEl || rowsEl.children[index] || null;
    if (activePtRow) {
      activePtRow.classList.add("active");
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

presetButtons.forEach((button) => {
  button.addEventListener("click", async () => {
    applyPreset(button);
    lastMonitorStamp = { reward: null, train: null, eval: null };
    await refreshMonitor(true);
  });
});

viewTabs.forEach((button) => {
  button.addEventListener("click", () => {
    const scope = button.closest(".view-scope") || document;
    switchView(button.dataset.viewTarget, scope);
  });
});

window.addEventListener("error", (event) => {
  setMonitorStatus(`前端错误: ${event.message}`, true);
});

document.getElementById("browse-btn").addEventListener("click", browseFiles);
document.getElementById("load-btn").addEventListener("click", () => loadFile());
document
  .getElementById("browse-policy-files-btn")
  .addEventListener("click", () => browsePolicyFiles(false));
policyPrevRowBtn.addEventListener("click", () => {
  if (Number.isInteger(currentPolicyRecordIndex) && currentPolicyRecordIndex > 0) {
    loadPolicyRecordItem(currentPolicyRecordIndex - 1);
  }
});
policyNextRowBtn.addEventListener("click", () => {
  if (
    Number.isInteger(currentPolicyRecordIndex) &&
    currentPolicyRecordIndex < currentPolicyRecordRowCount - 1
  ) {
    loadPolicyRecordItem(currentPolicyRecordIndex + 1);
  }
});
policyJumpRowBtn.addEventListener("click", () => {
  const value = Number.parseInt(policyRowIndexInput.value, 10);
  if (!Number.isInteger(value)) {
    setPolicyRecordStatus("请输入有效的 index", true);
    return;
  }
  if (value < 0 || value >= currentPolicyRecordRowCount) {
    setPolicyRecordStatus(`index 超出范围，当前文件共有 ${currentPolicyRecordRowCount} 条`, true);
    return;
  }
  loadPolicyRecordItem(value);
});
policyRowIndexInput.addEventListener("keydown", (event) => {
  if (event.key === "Enter") {
    event.preventDefault();
    policyJumpRowBtn.click();
  }
});
document.getElementById("refresh-monitor-btn").addEventListener("click", () => refreshMonitor(true));
toggleMonitorBtn.addEventListener("click", () => {
  autoRefreshEnabled = !autoRefreshEnabled;
  toggleMonitorBtn.textContent = autoRefreshEnabled ? "暂停自动更新" : "恢复自动更新";
  setMonitorStatus(autoRefreshEnabled ? "自动更新已开启" : "自动更新已暂停");
});

browseFiles().then(() => loadFile(currentPath));
ensurePolicyRootValue();
updatePolicyRowControls();
browsePolicyFiles(true);
setMonitorStatus("正在启动监控...");
refreshMonitor(true);
scheduleMonitor();
