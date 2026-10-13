"use strict";

// ---------------------------------------------------------------------------
// Small helpers

const $ = (selector, root = document) => root.querySelector(selector);
const $$ = (selector, root = document) => [...root.querySelectorAll(selector)];

function el(tag, attributes = {}, ...children) {
  const node = document.createElement(tag);
  for (const [key, value] of Object.entries(attributes)) {
    if (value === undefined || value === null || value === false) continue;
    if (key.startsWith("on")) node.addEventListener(key.slice(2), value);
    else if (key === "class") node.className = value;
    else node.setAttribute(key, value === true ? "" : value);
  }
  for (const child of children.flat()) {
    if (child === null || child === undefined || child === false) continue;
    node.append(child instanceof Node ? child : document.createTextNode(String(child)));
  }
  return node;
}

async function api(path, options = {}) {
  const response = await fetch(path, {
    method: options.body ? "POST" : "GET",
    headers: options.body ? { "Content-Type": "application/json" } : {},
    body: options.body ? JSON.stringify(options.body) : undefined,
  });
  if (options.binary && response.ok) return response.blob();
  const data = await response.json();
  if (!response.ok) throw new Error(data.error || response.statusText);
  return data;
}

function toast(message, error = false) {
  const box = $("#toast");
  box.textContent = message;
  box.className = "show" + (error ? " error" : "");
  clearTimeout(toast.timer);
  toast.timer = setTimeout(() => (box.className = ""), error ? 6000 : 3000);
}

const basename = (path) => path.split(/[\\/]/).pop();
const dirname = (path) => path.replace(/[\\/][^\\/]*$/, "") || "/";
const fileUrl = (path) => "/api/file?path=" + encodeURIComponent(path);

function splitArguments(text) {
  // Shell-like split: spaces separate, quotes group.
  const out = [];
  const pattern = /"([^"]*)"|'([^']*)'|(\S+)/g;
  let match;
  while ((match = pattern.exec(text))) out.push(match[1] ?? match[2] ?? match[3]);
  return out;
}

function parseTime(text) {
  text = String(text).trim();
  if (!/^(\d+(\.\d+)?|(\d+:){1,2}\d+(\.\d+)?)$/.test(text)) return null;
  return text.split(":").reduce((total, part) => total * 60 + Number(part), 0);
}

function formatClock(seconds) {
  const s = Math.max(0, seconds);
  const h = Math.floor(s / 3600), m = Math.floor((s % 3600) / 60), r = (s % 60).toFixed(1);
  return (h ? `${h}:${String(m).padStart(2, "0")}` : `${m}`) + `:${r.padStart(4, "0")}`;
}

// ---------------------------------------------------------------------------
// Saved form state (this browser only)

const STORE_KEY = "mic-array-synchronizer.form";
let saved = {};
try { saved = JSON.parse(localStorage.getItem(STORE_KEY) || "{}"); } catch { saved = {}; }
function persist() {
  try { localStorage.setItem(STORE_KEY, JSON.stringify(saved)); } catch { /* private mode */ }
}

function bindField(id) {
  const field = document.getElementById(id);
  const key = "field:" + id;
  if (key in saved) {
    if (field.type === "checkbox") field.checked = saved[key];
    else field.value = saved[key];
  }
  field.addEventListener("input", () => {
    saved[key] = field.type === "checkbox" ? field.checked : field.value;
    persist();
  });
  field.addEventListener("change", () => field.dispatchEvent(new Event("input")));
}

function bindRadio(name, onChange) {
  const key = "radio:" + name;
  const inputs = $$(`input[name="${name}"]`);
  if (saved[key]) inputs.forEach((input) => (input.checked = input.value === saved[key]));
  inputs.forEach((input) => input.addEventListener("change", () => {
    saved[key] = input.value;
    persist();
    onChange?.();
  }));
}
const radioValue = (name) => $(`input[name="${name}"]:checked`).value;

// ---------------------------------------------------------------------------
// Ordered file lists.  Every row is a list of files: one file, or several
// for a split recording ("group" lists only).

const lists = {};

class FileList {
  constructor(id, { groups = false, onChange = null } = {}) {
    this.id = id;
    this.root = document.getElementById(id);
    this.groups = groups;
    this.onChange = onChange;
    this.rows = saved["list:" + id] || [];
    this.dragFrom = null;
    lists[id] = this;
    this.render();
  }

  set(rows) {
    this.rows = rows.map((row) => (Array.isArray(row) ? row : [row]));
    this.changed();
  }

  add(paths) {
    for (const path of paths) this.rows.push([path]);
    this.changed();
  }

  addToRow(index, paths) {
    this.rows[index].push(...paths);
    this.changed();
  }

  changed() {
    saved["list:" + this.id] = this.rows;
    persist();
    this.render();
    this.onChange?.();
  }

  paths() {
    return this.rows.flat();
  }

  render() {
    this.root.replaceChildren(...this.rows.map((files, index) => this.renderRow(files, index)));
  }

  renderRow(files, index) {
    const chips = [];
    files.forEach((path, position) => {
      if (position) chips.push(el("span", { class: "plus" }, "+"));
      chips.push(el("span", { class: "chip", title: path },
        el("span", { class: "name" }, basename(path)),
        files.length > 1 && el("button", { title: "このファイルを外す", onclick: () => {
          files.splice(position, 1);
          this.changed();
        } }, "×")));
    });
    const row = el("li", { class: "row", draggable: "true" },
      el("span", { class: "handle", title: "ドラッグで並べ替え" }, "⋮⋮"),
      el("span", { class: "number" }, index + 1),
      el("span", { class: "files", title: files.join(" + ") }, chips),
      el("span", { class: "row-actions" },
        this.groups && el("button", { class: "small", title: "このあとに続く分割ファイルを追加",
          onclick: () => picker.openFiles(`分割ファイルを追加（${basename(files[0])} の続き）`, files.at(-1),
            (paths) => this.addToRow(index, paths)) }, "分割ファイルを追加"),
        el("button", { class: "small", title: "上へ", disabled: index === 0,
          onclick: () => this.move(index, index - 1) }, "↑"),
        el("button", { class: "small", title: "下へ", disabled: index === this.rows.length - 1,
          onclick: () => this.move(index, index + 1) }, "↓"),
        el("button", { class: "small", title: "外す", onclick: () => {
          this.rows.splice(index, 1);
          this.changed();
        } }, "×")));
    row.addEventListener("dragstart", (event) => {
      this.dragFrom = index;
      row.classList.add("dragging");
      event.dataTransfer.effectAllowed = "move";
    });
    row.addEventListener("dragend", () => row.classList.remove("dragging"));
    row.addEventListener("dragover", (event) => {
      if (this.dragFrom === null) return;
      event.preventDefault();
      row.classList.add("over");
    });
    row.addEventListener("dragleave", () => row.classList.remove("over"));
    row.addEventListener("drop", (event) => {
      event.preventDefault();
      row.classList.remove("over");
      if (this.dragFrom !== null) this.move(this.dragFrom, index);
      this.dragFrom = null;
    });
    return row;
  }

  move(from, to) {
    if (to < 0 || to >= this.rows.length || from === to) return;
    const [row] = this.rows.splice(from, 1);
    this.rows.splice(to, 0, row);
    this.changed();
  }
}

// ---------------------------------------------------------------------------
// Server-side file picker

const picker = {
  mode: "files",
  folder: null,
  selected: new Set(),
  entries: [],
  onDone: null,

  init() {
    $("#picker-go").addEventListener("click", () => this.load($("#picker-path").value));
    $("#picker-path").addEventListener("keydown", (event) => {
      if (event.key === "Enter") { event.preventDefault(); this.load($("#picker-path").value); }
    });
    $("#picker-up").addEventListener("click", () => this.parent && this.load(this.parent));
    $("#picker-media-only").addEventListener("change", () => this.render());
    $("#picker-select-all").addEventListener("click", () => {
      this.visible().filter((e) => !e.dir).forEach((e) => this.selected.add(e.path));
      this.render();
    });
    $("#picker-ok").addEventListener("click", () => this.finish());
  },

  openFiles(title, near, onDone) { this.open("files", title, near, onDone); },
  openFolder(title, near, onDone) { this.open("folder", title, near, onDone); },
  openSave(title, near, onDone) { this.open("save", title, near, onDone); },

  open(mode, title, near, onDone) {
    this.mode = mode;
    this.onDone = onDone;
    this.selected = new Set();
    $("#picker-title").textContent = title;
    $("#picker-ok").textContent = { files: "追加", folder: "このフォルダを選ぶ", save: "ここに保存" }[mode];
    $("#picker-select-all").classList.toggle("hidden", mode !== "files");
    $("#picker-save").classList.toggle("hidden", mode !== "save");
    if (mode === "save" && near) $("#picker-filename").value = basename(near);
    $("#picker").showModal();
    const start = near ? (mode === "folder" ? near : dirname(near)) : saved.pickerFolder;
    this.load(start || "");
  },

  async load(path) {
    const request = (this.request = (this.request || 0) + 1);
    const typed = $("#picker-path").value;
    try {
      const data = await api("/api/browse?path=" + encodeURIComponent(path || ""));
      if (request !== this.request) return;  // a later folder was asked for meanwhile
      if ($("#picker-path").value !== typed) return;  // the user is typing another path
      this.folder = data.path;
      this.parent = data.parent;
      this.entries = data.entries;
      saved.pickerFolder = data.path;
      persist();
      $("#picker-path").value = data.path;
      this.render();
    } catch (error) {
      toast(error.message, true);
    }
  },

  visible() {
    const mediaOnly = $("#picker-media-only").checked;
    return this.entries.filter((e) => e.dir || !mediaOnly || e.media || this.mode === "save");
  },

  render() {
    const items = this.visible().map((entry) => {
      if (entry.dir) {
        return el("li", { class: "dir", onclick: () => this.load(entry.path) },
          el("span", { class: "name" }, entry.name));
      }
      const checked = this.selected.has(entry.path);
      const box = this.mode === "files" && el("input", { type: "checkbox", checked });
      const item = el("li", {
        onclick: () => {
          if (this.mode === "files") {
            if (this.selected.has(entry.path)) this.selected.delete(entry.path);
            else this.selected.add(entry.path);
            this.render();
          } else if (this.mode === "save") {
            $("#picker-filename").value = entry.name;
          }
        },
        ondblclick: () => {
          if (this.mode === "files") { this.selected = new Set([entry.path]); this.finish(); }
        },
      }, box, el("span", { class: "name" }, entry.name),
        el("span", { class: "meta" }, formatSize(entry.size)));
      return item;
    });
    $("#picker-list").replaceChildren(...items);
    $("#picker-count").textContent = this.mode === "files" ? `${this.selected.size} 個選択中` : "";
    $("#picker-ok").disabled = this.mode === "files" && this.selected.size === 0;
  },

  finish() {
    let result;
    if (this.mode === "files") {
      // Keep folder order (names sort the way split files were recorded).
      result = this.entries.filter((e) => this.selected.has(e.path)).map((e) => e.path);
    } else if (this.mode === "folder") {
      result = this.folder;
    } else {
      const name = $("#picker-filename").value.trim();
      if (!name) return;
      result = this.folder.replace(/[\\/]$/, "") + "/" + name;
    }
    $("#picker").close();
    this.onDone?.(result);
  },
};

function formatSize(bytes) {
  if (bytes === null || bytes === undefined) return "";
  const units = ["B", "KB", "MB", "GB", "TB"];
  let value = bytes, unit = 0;
  while (value >= 1024 && unit < units.length - 1) { value /= 1024; unit++; }
  return `${value.toFixed(unit ? 1 : 0)} ${units[unit]}`;
}

// ---------------------------------------------------------------------------
// Run panels: history, progress, log, results

const panels = {};

class RunPanel {
  constructor(kind) {
    this.kind = kind;
    this.root = $(`.run[data-kind="${kind}"]`);
    this.selectedId = saved["job:" + kind] || null;
    this.detail = null;
    this.root.append(
      this.historyCard = el("div", { class: "card" }, el("h2", {}, "実行履歴"),
        this.history = el("ul", { class: "history" })),
      this.detailCard = el("div", { class: "card hidden" }));
    panels[kind] = this;
  }

  async refreshList() {
    let list;
    try { list = await api("/api/jobs?kind=" + this.kind); } catch { return; }
    this.jobs = list;
    if (!list.length) {
      this.history.replaceChildren(el("li", { class: "hint" }, "まだ実行していません。"));
      return;
    }
    if (!this.selectedId || !list.some((j) => j.id === this.selectedId)) this.select(list[0].id, false);
    this.history.replaceChildren(...list.slice(0, 20).map((job) => el("li", {
      class: job.id === this.selectedId ? "selected" : "",
      onclick: () => this.select(job.id),
    }, el("span", {}, jobTime(job)), badge(job.state),
      el("span", { class: "target", title: job.output || "" }, job.output || ""),
      job.state === "running" && el("div", { class: "bar" }, el("div", { style: `width:${job.progress * 100}%` })))));
  }

  select(id, refresh = true) {
    this.selectedId = id;
    saved["job:" + this.kind] = id;
    persist();
    this.detail = null;
    if (refresh) { this.refreshList(); this.refreshDetail(); }
  }

  get running() {
    return (this.jobs || []).some((job) => job.state === "running");
  }

  async refreshDetail() {
    if (!this.selectedId) return;
    let job;
    try { job = await api("/api/jobs/" + this.selectedId); } catch { return; }
    const previous = this.detail;
    this.detail = job;
    // Avoid re-rendering finished jobs (keeps players and open sections as they are).
    if (previous && previous.id === job.id && previous.state !== "running" && job.state !== "running") return;
    this.renderDetail(job, previous);
  }

  renderDetail(job, previous) {
    const logOpen = $("details.log", this.detailCard)?.open ?? job.state === "failed";
    const card = this.detailCard;
    card.classList.remove("hidden");
    const command = [{ sync: "sync_media.py", compose: "compose_media.py", trim: "trim_media.py" }[job.kind],
      ...job.arguments.map((a) => (/[\s"']/.test(a) ? `"${a}"` : a))].join(" ");
    const children = [
      el("div", { class: "status-line" }, el("h2", {}, "実行状況"), badge(job.state)),
      el("div", { class: "bar big" }, el("div", { style: `width:${job.progress * 100}%` })),
      el("div", { class: "status-line" }, el("span", {}, job.status),
        job.state === "running" && el("button", { onclick: () => this.stop(job) }, "中止")),
      el("details", {}, el("summary", {}, "実行したコマンド"), el("pre", {}, "python3 " + command)),
      el("details", { class: "log", open: logOpen }, el("summary", {}, "ログ"),
        el("pre", { class: "log-text" }, job.log || "（まだ出力はありません）")),
    ];
    if (job.report) children.push(renderReport(job));
    if (job.outputs.length) children.push(renderOutputs(job));
    card.replaceChildren(...children);
    const log = $(".log-text", card);
    if (log && job.state === "running") log.scrollTop = log.scrollHeight;
    if (previous?.state === "running" && job.state !== "running") {
      toast({ done: "完了しました", failed: "エラーで終了しました", stopped: "中止しました" }[job.state] || job.status,
        job.state === "failed");
    }
  }

  async stop(job) {
    if (!confirm("この処理を中止しますか？")) return;
    await api(`/api/jobs/${job.id}/stop`, { body: {} });
    this.refreshDetail();
  }

  async start(arguments_) {
    try {
      const { id } = await api("/api/jobs", { body: { kind: this.kind, arguments: arguments_ } });
      this.select(id);
      toast("開始しました");
    } catch (error) {
      toast(error.message, true);
    }
  }
}

function jobTime(job) {
  const date = new Date(job.started * 1000);
  return date.toLocaleString("ja-JP", { month: "numeric", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

function badge(state) {
  const label = { running: "実行中", done: "完了", failed: "エラー", stopped: "中止", lost: "中断" }[state];
  return el("span", { class: "badge " + state }, label);
}

function renderOutputs(job) {
  const player = el("div", { class: "player" });
  const items = job.outputs.map((path) => {
    const isVideo = /\.(mp4|mov|m4v)$/i.test(path);
    const isAudio = /\.(wav|mp3|m4a|flac)$/i.test(path);
    return el("li", {},
      el("span", { class: "name", title: path }, basename(path)),
      (isVideo || isAudio) && el("button", { class: "small", onclick: () => {
        const media = el(isVideo ? "video" : "audio", { controls: true, preload: "metadata", src: fileUrl(path) });
        media.addEventListener("error", () => toast("このブラウザでは再生できない形式です（HEVC は Safari や対応した Chrome で再生できます）", true));
        player.replaceChildren(el("div", { class: "hint" }, basename(path)), media);
        media.play().catch(() => {});
      } }, "▶ 再生"),
      el("a", { href: fileUrl(path), download: basename(path) }, el("button", { class: "small" }, "保存")));
  });
  return el("details", { open: true }, el("summary", {}, `出力（${job.outputs.length}）`),
    el("ul", { class: "outputs" }, items), player);
}

function renderReport(job) {
  const report = job.report;
  const rows = report.inputs.map((item) => {
    const warn = item.correlation_pairs_used === 0 || ["low", "timecode-only", "conversation-only"].includes(item.confidence)
      || (item.segments || []).some((s) => s.method === "timecode-only");
    const segments = (item.segments || []).map((s) =>
      `+${basename(s.input)}: ${s.method}${s.gap_before_seconds ? `（間隔 ${s.gap_before_seconds.toFixed(3)}秒）` : ""}`).join("\n");
    return el("tr", { class: warn ? "warn" : "" },
      el("td", { title: item.input }, (warn ? "⚠ " : "") + basename(item.input), segments && el("div", { class: "hint" }, segments)),
      el("td", {}, item.confidence),
      el("td", {}, item.beep_seconds?.toFixed(3) ?? ""),
      el("td", {}, item.correlation_correction_ms?.toFixed(2) ?? ""),
      el("td", {}, item.drift_ppm === null || item.drift_ppm === undefined ? "不明" : item.drift_ppm.toFixed(1)),
      el("td", {}, item.correlation_pairs_used));
  });
  const accepted = report.pairs.filter((p) => p.accepted).length;
  const done = job.state === "done" && report.check_wav;
  return el("details", { open: true }, el("summary", {}, "結果（report.json）"),
    el("div", { class: "table-scroll" }, el("table", { class: "report" },
      el("thead", {}, el("tr", {}, ["ファイル", "検出", "同期音 秒", "相関補正 ms", "ドリフト ppm", "ペア"].map((t) => el("th", {}, t)))),
      el("tbody", {}, rows))),
    el("p", { class: "hint" },
      `共通区間 ${(report.common_duration_seconds / 60).toFixed(1)} 分、採用したペア ${accepted}/${report.pairs.length}。⚠ のファイルは sync_check.wav で確認してください。`),
    done && el("div", { class: "buttons" },
      el("button", { onclick: () => sendToCompose(report) }, "合成タブに入力 →"),
      el("button", { onclick: () => sendToTrim(report) }, "切り抜きタブに入力 →")));
}

// ---------------------------------------------------------------------------
// Tabs

function showTab(name) {
  $$(".tabs button").forEach((b) => b.classList.toggle("active", b.dataset.tab === name));
  $$(".tab").forEach((t) => t.classList.toggle("hidden", t.id !== "tab-" + name));
  saved.tab = name;
  persist();
}

function sendToCompose(report) {
  const outputs = report.inputs.filter((i) => i.output);
  lists["compose-videos"].set(outputs.filter((i) => i.has_video).map((i) => i.output));
  lists["compose-audios"].set(outputs.filter((i) => !i.has_video).map((i) => i.output));
  setField("compose-output", dirname(report.check_wav.path) + "/composite.mp4");
  showTab("compose");
  toast("同期した出力を合成タブに入れました");
}

function sendToTrim(report) {
  lists["trim-inputs"].set(report.inputs.filter((i) => i.output).map((i) => i.output));
  setField("trim-output", dirname(report.check_wav.path) + "/clips");
  showTab("trim");
  toast("同期した出力を切り抜きタブに入れました");
}

function setField(id, value) {
  const field = document.getElementById(id);
  field.value = value;
  field.dispatchEvent(new Event("input"));
}

// ---------------------------------------------------------------------------
// 同期

function syncArguments() {
  const rows = lists["sync-inputs"].rows;
  const output = $("#sync-output").value.trim();
  if (!rows.length) throw new Error("入力ファイルを追加してください。");
  if (!output) throw new Error("出力フォルダを指定してください。");
  const args = ["-o", output, ...rows.map((files) => files.join("+")),
    "--duration-mode", $("#sync-duration-mode").value,
    "--video-encoder", $("#sync-encoder").value,
    "--pre-roll", $("#sync-pre-roll").value || "0.05",
    "--max-lag-ms", $("#sync-max-lag").value || "20"];
  if ($("#sync-analyze-only").checked) args.push("--analyze-only");
  if ($("#sync-fast-copy").checked) args.push("--fast-copy");
  if ($("#sync-bitrate").value.trim()) args.push("--video-bitrate", $("#sync-bitrate").value.trim());
  for (const line of $("#sync-manual").value.split("\n").map((l) => l.trim()).filter(Boolean)) {
    args.push("--manual-beep", line);
  }
  args.push(...splitArguments($("#sync-extra").value));
  return args;
}

// ---------------------------------------------------------------------------
// 合成

const probes = {};
async function probe(path) {
  if (!(path in probes)) probes[path] = api("/api/probe?path=" + encodeURIComponent(path)).catch(() => null);
  return probes[path];
}

const COLORS = ["#3B6FB6", "#E08A2E", "#3E9B5B", "#8A5BB0", "#C9483F", "#2E9BA6"];

function gridShape(count, layout) {
  if (layout === "row") return [count, 1];
  if (layout === "column") return [1, count];
  const columns = Math.ceil(Math.sqrt(count));
  return [columns, Math.ceil(count / columns)];
}

async function drawDiagram() {
  const videos = lists["compose-videos"].paths();
  const box = $("#compose-diagram");
  if (!videos.length) {
    box.replaceChildren(el("p", { class: "empty" }, "映像を追加すると配置が表示されます。"));
    return;
  }
  let aspect = 16 / 9;
  const cell = $("#compose-cell").value.trim().match(/^(\d+)x(\d+)$/i);
  if (cell) aspect = Number(cell[1]) / Number(cell[2]);
  else {
    const info = await probe(videos[0]);
    if (info?.video_size) aspect = info.video_size[0] / info.video_size[1];
  }
  const [columns, rows] = gridShape(videos.length, radioValue("layout"));
  const w = 160, h = w / aspect;
  const ns = "http://www.w3.org/2000/svg";
  const svg = document.createElementNS(ns, "svg");
  svg.setAttribute("viewBox", `0 0 ${columns * w} ${rows * h}`);
  const add = (tag, attrs, text) => {
    const node = document.createElementNS(ns, tag);
    for (const [k, v] of Object.entries(attrs)) node.setAttribute(k, v);
    if (text !== undefined) node.textContent = text;
    svg.append(node);
  };
  for (let slot = 0; slot < columns * rows; slot++) {
    const x = (slot % columns) * w, y = Math.floor(slot / columns) * h;
    if (slot >= videos.length) { add("rect", { x, y, width: w, height: h, fill: "#000" }); continue; }
    add("rect", { x, y, width: w, height: h, fill: COLORS[slot % COLORS.length], stroke: "#12161c", "stroke-width": 2 });
    add("text", { x: x + w / 2, y: y + h * 0.52, "font-size": h * 0.38, fill: "#fff", "text-anchor": "middle",
      "font-weight": "bold", "font-family": "sans-serif" }, slot + 1);
    add("text", { x: x + w / 2, y: y + h * 0.84, "font-size": Math.min(11, h * 0.11), fill: "#fff",
      "text-anchor": "middle", "font-family": "sans-serif" }, basename(videos[slot]));
  }
  box.replaceChildren(svg, el("p", { class: "hint" }, `${columns}×${rows} の配置（番号は映像の順番）`));
}

async function showPreview() {
  const videos = lists["compose-videos"].paths();
  if (!videos.length) return toast("映像を追加してください。", true);
  const atText = $("#compose-preview-at").value.trim();
  const at = atText ? parseTime(atText) : null;
  if (atText && at === null) return toast("時刻は 600 や 10:00 の形で入力してください。", true);
  const button = $("#compose-preview");
  button.disabled = true;
  button.textContent = "作成中…";
  try {
    const blob = await api("/api/preview", { binary: true, body: {
      videos, layout: radioValue("layout"), cell: $("#compose-cell").value.trim(), at } });
    if (!(blob instanceof Blob)) throw new Error("プレビューを作成できませんでした。");
    const image = el("img", { src: URL.createObjectURL(blob), alt: "プレビュー" });
    $("#compose-preview-image").replaceChildren(image);
  } catch (error) {
    toast(error.message, true);
  } finally {
    button.disabled = false;
    button.textContent = "実際の1フレームを表示";
  }
}

function composeArguments() {
  const videos = lists["compose-videos"].paths();
  const audios = lists["compose-audios"].paths();
  const output = $("#compose-output").value.trim();
  if (!videos.length && !audios.length) throw new Error("映像か音声を追加してください。");
  if (!output) throw new Error("出力ファイルを指定してください。");
  const args = ["--layout", radioValue("layout"), "--audio-mode", radioValue("audio-mode"),
    "--codec", $("#compose-codec").value, "--encoder", $("#compose-encoder").value,
    "--cell-size", $("#compose-cell").value.trim() || "source",
    "--video-bitrate", $("#compose-bitrate").value.trim() || "source",
    "--no-preview", "-o", output];
  if (videos.length) args.push("-v", ...videos);
  if (audios.length) args.push("-a", ...audios);
  return args;
}

// ---------------------------------------------------------------------------
// 切り抜き

let ranges = saved.ranges || [{ start: "", end: "", name: "" }];

function renderRanges() {
  saved.ranges = ranges;
  persist();
  $("#trim-ranges").replaceChildren(...ranges.map((range, index) => {
    const input = (key, placeholder) => el("input", { value: range[key], placeholder,
      oninput: (event) => { range[key] = event.target.value; saved.ranges = ranges; persist(); } });
    return el("tr", {},
      el("td", {}, input("start", "10:00")),
      el("td", {}, input("end", "12:30 / +90")),
      el("td", {}, input("name", "task1")),
      el("td", {}, el("button", { class: "small", title: "この範囲を消す", onclick: () => {
        ranges.splice(index, 1);
        if (!ranges.length) ranges.push({ start: "", end: "", name: "" });
        renderRanges();
      } }, "×")));
  }));
}

function trimArguments() {
  const inputs = lists["trim-inputs"].paths();
  const output = $("#trim-output").value.trim();
  if (!inputs.length) throw new Error("切り抜くメディアを追加してください。");
  if (!output) throw new Error("出力フォルダを指定してください。");
  const specs = [];
  for (const [index, range] of ranges.entries()) {
    const start = range.start.trim(), end = range.end.trim(), name = range.name.trim();
    if (!start && !end) continue;
    const length = end.startsWith("+");
    const startSeconds = parseTime(start), endSeconds = parseTime(length ? end.slice(1) : end);
    if (startSeconds === null || endSeconds === null) throw new Error(`範囲 ${index + 1} の時刻を確認してください。`);
    if (!length && endSeconds <= startSeconds) throw new Error(`範囲 ${index + 1} は終了を開始より後にしてください。`);
    if (name && !/^[\w.-]+$/.test(name)) throw new Error(`範囲 ${index + 1} の名前には英数字・_・-・. だけを使ってください。`);
    specs.push(`${start}${length ? end : "-" + end}${name ? "=" + name : ""}`);
  }
  if (!specs.length) throw new Error("範囲を入力してください。");
  const args = ["-o", output, ...inputs];
  for (const spec of specs) args.push("-r", spec);
  if ($("#trim-fast-copy").checked) args.push("--fast-copy");
  args.push("--codec", $("#trim-codec").value, "--encoder", $("#trim-encoder").value,
    "--video-bitrate", $("#trim-bitrate").value.trim() || "source");
  return args;
}

// ---------------------------------------------------------------------------
// Start-up

async function init() {
  const info = await api("/api/info").catch(() => ({ video_encoders: ["auto"] }));
  $("#sync-encoder").replaceChildren(...info.video_encoders.map((name) => el("option", { value: name }, name)));

  ["sync-output", "sync-analyze-only", "sync-fast-copy", "sync-duration-mode", "sync-encoder", "sync-bitrate",
    "sync-pre-roll", "sync-max-lag", "sync-manual", "sync-extra", "compose-codec", "compose-encoder",
    "compose-cell", "compose-bitrate", "compose-output", "compose-preview-at", "trim-fast-copy", "trim-codec",
    "trim-encoder", "trim-bitrate", "trim-output"].forEach(bindField);
  bindRadio("layout", drawDiagram);
  bindRadio("audio-mode");
  $("#compose-cell").addEventListener("input", drawDiagram);

  new FileList("sync-inputs", { groups: true });
  new FileList("compose-videos", { onChange: drawDiagram });
  new FileList("compose-audios");
  new FileList("trim-inputs");
  drawDiagram();
  renderRanges();
  picker.init();

  $$("[data-pick]").forEach((button) => button.addEventListener("click", () => {
    const list = lists[button.dataset.pick];
    picker.openFiles("ファイルを選択", list.paths().at(-1), (paths) => list.add(paths));
  }));
  $$("[data-clear]").forEach((button) => button.addEventListener("click", () => {
    if (confirm("リストを空にしますか？")) lists[button.dataset.clear].set([]);
  }));
  $$("[data-pick-folder]").forEach((button) => button.addEventListener("click", () => {
    const id = button.dataset.pickFolder;
    picker.openFolder("フォルダを選択", $("#" + id).value.trim(), (path) => setField(id, path));
  }));
  $$("[data-pick-save]").forEach((button) => button.addEventListener("click", () => {
    const id = button.dataset.pickSave;
    picker.openSave("保存先を選択", $("#" + id).value.trim() || null, (path) => setField(id, path));
  }));
  $$(".tabs button").forEach((button) => button.addEventListener("click", () => showTab(button.dataset.tab)));
  $("#compose-preview").addEventListener("click", showPreview);
  $("#trim-add-range").addEventListener("click", () => { ranges.push({ start: "", end: "", name: "" }); renderRanges(); });

  for (const kind of ["sync", "compose", "trim"]) new RunPanel(kind);
  const starters = { sync: syncArguments, compose: composeArguments, trim: trimArguments };
  for (const [kind, build] of Object.entries(starters)) {
    $(`#${kind}-start`).addEventListener("click", () => {
      let args;
      try { args = build(); } catch (error) { return toast(error.message, true); }
      if (panels[kind].running && !confirm("同じ種類の処理を実行中です。並行して開始しますか？")) return;
      panels[kind].start(args);
    });
  }
  showTab(saved.tab || "sync");

  const tick = () => {
    for (const panel of Object.values(panels)) {
      panel.refreshList();
      panel.refreshDetail();
    }
  };
  tick();
  setInterval(tick, 2500);
}

init();
