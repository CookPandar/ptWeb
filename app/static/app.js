const statusEl = document.getElementById("status");
const fileInput = document.getElementById("file-input");
const dirInput = document.getElementById("dir-input");
const tokenizerInput = document.getElementById("tokenizer-input");
const overviewEl = document.getElementById("overview");
const detailEl = document.getElementById("detail");
const detailTitleEl = document.getElementById("detail-title");
const rowsEl = document.getElementById("rows");
const fileListEl = document.getElementById("file-list");
const rewardCsvInput = document.getElementById("reward-csv-input");
const trainCsvInput = document.getElementById("train-csv-input");
const rewardChartGridEl = document.getElementById("reward-chart-grid");
const trainChartGridEl = document.getElementById("train-chart-grid");
const rewardTailEl = document.getElementById("reward-tail");
const trainTailEl = document.getElementById("train-tail");
const rewardMetaEl = document.getElementById("reward-meta");
const trainMetaEl = document.getElementById("train-meta");
const monitorStatusEl = document.getElementById("monitor-status");
const toggleMonitorBtn = document.getElementById("toggle-monitor-btn");

let currentPath = fileInput.value;
let activeRow = null;
let autoRefreshEnabled = true;
let monitorTimer = null;
let lastMonitorStamp = { reward: null, train: null };
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

function setMonitorStatus(text, isError = false) {
  monitorStatusEl.textContent = text;
  monitorStatusEl.style.color = isError ? "#a42828" : "";
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
      <td>${row.index ?? ""}</td>
      <td>${row.agent_index ?? ""}</td>
      <td>${row.timestep ?? ""}</td>
      <td>${row.reward ?? ""}</td>
      <td>${row.value ?? ""}</td>
      <td>${row.log_prob ?? ""}</td>
      <td>${row.done ?? ""}</td>
      <td>${row.prompt_len ?? ""}</td>
      <td>${row.response_len ?? ""}</td>
      <td>${row.critic_len ?? ""}</td>
    `;
    tr.addEventListener("click", () => loadItem(row.index, tr));
    rowsEl.appendChild(tr);
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
    const chart = document.createElement("div");
    chart.className = "chart";
    panel.appendChild(title);
    panel.appendChild(chart);
    container.appendChild(panel);
    renderChart(chart, group.chart, null);
  });
  if (titleMetaEl) {
    titleMetaEl.textContent = `${groups.length} charts`;
  }
}

async function refreshMonitor(force = false) {
  try {
    const data = await fetchJson(
      `/api/monitor?reward_path=${encodeURIComponent(rewardCsvInput.value)}&train_path=${encodeURIComponent(trainCsvInput.value)}`
    );
    const changed =
      force ||
      data.reward.mtime !== lastMonitorStamp.reward ||
      data.train.mtime !== lastMonitorStamp.train;

    if (!changed) {
      setMonitorStatus("监控中，文件未变化");
      return;
    }

    lastMonitorStamp = { reward: data.reward.mtime, train: data.train.mtime };
    renderChartGroups(rewardChartGridEl, data.reward.groups, rewardMetaEl);
    renderChartGroups(trainChartGridEl, data.train.groups, trainMetaEl);
    rewardTailEl.textContent = prettyJson(data.reward.rows.slice(-10));
    trainTailEl.textContent = prettyJson(data.train.rows.slice(-10));
    setMonitorStatus(
      `已更新 reward(${data.reward.row_count}) / train(${data.train.row_count})，最近修改 ${new Date(
        Math.max(data.reward.mtime, data.train.mtime) * 1000
      ).toLocaleString()}`
    );
  } catch (error) {
    setMonitorStatus(error.message, true);
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
    if (activeRow) {
      activeRow.classList.remove("active");
    }
    activeRow = rowEl || rowsEl.children[index] || null;
    if (activeRow) {
      activeRow.classList.add("active");
    }
  } catch (error) {
    setStatus(error.message, true);
  }
}

document.getElementById("browse-btn").addEventListener("click", browseFiles);
document.getElementById("load-btn").addEventListener("click", () => loadFile());
document.getElementById("refresh-monitor-btn").addEventListener("click", () => refreshMonitor(true));
toggleMonitorBtn.addEventListener("click", () => {
  autoRefreshEnabled = !autoRefreshEnabled;
  toggleMonitorBtn.textContent = autoRefreshEnabled ? "暂停自动更新" : "恢复自动更新";
  setMonitorStatus(autoRefreshEnabled ? "自动更新已开启" : "自动更新已暂停");
});

browseFiles().then(() => loadFile(currentPath));
refreshMonitor(true);
scheduleMonitor();
