// Ambient Scribe — shared page shell and helpers (no dependencies).

// Same origin as the FastAPI app by default. Override with ?api=http://host:9000
export const API_BASE = new URLSearchParams(location.search).get("api") || "";

export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

// ---------------------------------------------------------------- icons
const ICONS = {
  mic: '<path d="M12 15a3 3 0 0 0 3-3V6a3 3 0 0 0-6 0v6a3 3 0 0 0 3 3Z"/><path d="M19 11a7 7 0 0 1-14 0M12 18v3"/>',
  stop: '<rect x="7" y="7" width="10" height="10" rx="1.5"/>',
  upload: '<path d="M12 16V4M7 9l5-5 5 5"/><path d="M20 16v3a2 2 0 0 1-2 2H6a2 2 0 0 1-2-2v-3"/>',
  plus: '<path d="M12 5v14M5 12h14"/>',
  check: '<path d="m5 12 5 5L20 7"/>',
  x: '<path d="M6 6l12 12M18 6 6 18"/>',
  list: '<path d="M8 6h13M8 12h13M8 18h13M3 6h.01M3 12h.01M3 18h.01"/>',
  copy: '<rect x="9" y="9" width="12" height="12" rx="2"/><path d="M5 15V5a2 2 0 0 1 2-2h10"/>',
  search: '<circle cx="11" cy="11" r="7"/><path d="m20 20-3.5-3.5"/>',
  info: '<circle cx="12" cy="12" r="9"/><path d="M12 8h.01M11 12h1v5h1"/>',
  alert: '<path d="M12 9v4M12 17h.01"/><path d="M10.3 3.9 1.8 18a2 2 0 0 0 1.7 3h17a2 2 0 0 0 1.7-3L13.7 3.9a2 2 0 0 0-3.4 0Z"/>',
  doc: '<path d="M14 3H7a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h10a2 2 0 0 0 2-2V8l-5-5Z"/><path d="M14 3v5h5M9 13h6M9 17h6"/>',
  wave: '<path d="M3 12h2M7 8v8M11 5v14M15 9v6M19 11v2"/>',
  arrow: '<path d="M5 12h14M13 6l6 6-6 6"/>',
  user: '<circle cx="12" cy="8" r="4"/><path d="M4 21a8 8 0 0 1 16 0"/>',
  star: '<path d="m12 3 2.8 5.8 6.2.9-4.5 4.4 1.1 6.2L12 17.4l-5.6 2.9 1.1-6.2L3 9.7l6.2-.9L12 3Z"/>',
};
export function icon(name, extra = "") {
  const fill = name === "star" ? 'fill="currentColor" stroke="none"' : 'fill="none" stroke="currentColor" stroke-width="2" stroke-linecap="round" stroke-linejoin="round"';
  return `<svg viewBox="0 0 24 24" ${fill} aria-hidden="true" ${extra}>${ICONS[name] || ""}</svg>`;
}

// ---------------------------------------------------------------- shell
export function renderShell(active) {
  const header = document.createElement("header");
  header.className = "topbar";
  header.innerHTML = `
    <a class="brand" href="/" aria-label="Ambient Scribe home">
      <img src="static/moh-hic-logo.png" alt="Republic of Rwanda, Ministry of Health, Health Intelligence Center">
    </a>
    <nav class="nav" aria-label="Main">
      <a href="/" ${active === "capture" ? 'aria-current="page"' : ""}>${icon("mic")}New consultation</a>
      <a href="/review" ${active === "review" ? 'aria-current="page"' : ""}>${icon("list")}Review <span class="count" id="navPending" hidden></span></a>
    </nav>
    <div class="topbar-right">
      <span class="status" id="sysStatus" title="Checking services…"><span class="dot"></span><span class="label">Checking…</span></span>
    </div>`;
  document.body.prepend(header);

  const toast = document.createElement("div");
  toast.className = "toast"; toast.id = "toast"; toast.setAttribute("role", "status");
  document.body.append(toast);

  checkHealth();
  setInterval(checkHealth, 30000);
  refreshPendingCount();
}

async function checkHealth() {
  const el = $("#sysStatus");
  try {
    const r = await fetch(`${API_BASE}/health`);
    const j = await r.json();
    const ok = j.status === "healthy";
    const down = [!j.triton && "speech recognition", !j.db && "database"].filter(Boolean);
    el.innerHTML = `<span class="dot ${ok ? "ok" : "bad"}"></span><span class="label">${ok ? "All systems operational" : "Service degraded"}</span>`;
    el.title = ok ? "Speech recognition and database are online" : `Unavailable: ${down.join(", ")}`;
  } catch {
    el.innerHTML = '<span class="dot bad"></span><span class="label">Server unreachable</span>';
    el.title = "The API server did not respond";
  }
}

export async function refreshPendingCount() {
  try {
    const { counts } = await api("/api/consultations?status=pending&limit=1");
    const el = $("#navPending");
    el.textContent = counts.pending;
    el.hidden = !counts.pending;
    el.title = `${counts.pending} awaiting review`;
    return counts;
  } catch { return null; }
}

// ---------------------------------------------------------------- api
export async function api(path, options = {}) {
  const r = await fetch(`${API_BASE}${path}`, options);
  const text = await r.text();
  let body;
  try { body = text ? JSON.parse(text) : null; } catch { body = null; }
  if (!r.ok) {
    const detail = body?.detail;
    const msg = typeof detail === "string" ? detail
      : Array.isArray(detail) ? detail.map((d) => d.msg).join("; ")
      : `HTTP ${r.status}${text && !body ? `: ${text.slice(0, 200)}` : ""}`;
    const err = new Error(msg);
    err.status = r.status; err.body = body;
    throw err;
  }
  return body;
}

// ---------------------------------------------------------------- formatting
export function escapeHtml(s) {
  return String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
}
function inline(s) {
  return escapeHtml(s)
    .replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>")
    .replace(/(^|[^*])\*([^*\s][^*]*?)\*(?!\*)/g, "$1<em>$2</em>");
}
// Minimal, escape-first markdown: headings, bullet / numbered lists, bold, italics.
export function renderMarkdown(md) {
  const out = [];
  let list = null;
  const close = () => { if (list) { out.push(`</${list}>`); list = null; } };
  for (const raw of String(md ?? "").split(/\r?\n/)) {
    const line = raw.trim();
    let m;
    if (!line) { close(); continue; }
    if ((m = line.match(/^#{1,6}\s+(.*)$/)) || (m = line.match(/^\*\*([^*]+?):?\*\*:?$/))) {
      close(); out.push(`<h3>${inline(m[1].replace(/:$/, ""))}</h3>`);
    } else if ((m = line.match(/^[-*•]\s+(.*)$/))) {
      if (list !== "ul") { close(); out.push("<ul>"); list = "ul"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else if ((m = line.match(/^\d+[.)]\s+(.*)$/))) {
      if (list !== "ol") { close(); out.push("<ol>"); list = "ol"; }
      out.push(`<li>${inline(m[1])}</li>`);
    } else {
      close(); out.push(`<p>${inline(line)}</p>`);
    }
  }
  close();
  return out.join("");
}

export function fmtDuration(seconds) {
  if (seconds == null || isNaN(seconds)) return "–";
  const s = Math.round(seconds);
  return `${Math.floor(s / 60)}:${String(s % 60).padStart(2, "0")}`;
}
export function fmtDate(iso, opts = {}) {
  if (!iso) return "–";
  const d = new Date(iso);
  return d.toLocaleString(undefined, { day: "2-digit", month: "short", year: opts.year ? "numeric" : undefined, hour: "2-digit", minute: "2-digit" });
}
export function fmtRelative(iso) {
  const diff = (Date.now() - new Date(iso).getTime()) / 1000;
  if (diff < 60) return "just now";
  if (diff < 3600) return `${Math.floor(diff / 60)} min ago`;
  if (diff < 86400) return `${Math.floor(diff / 3600)} h ago`;
  return fmtDate(iso);
}
export function fmtBytes(n) {
  if (!n) return "–";
  return n < 1024 * 1024 ? `${(n / 1024).toFixed(0)} KB` : `${(n / 1024 / 1024).toFixed(1)} MB`;
}
export const VERDICT_LABEL = { approved: "Approved", needs_correction: "Needs correction", rejected: "Rejected" };
export function verdictPill(verdict, status) {
  if (status === "failed") return '<span class="pill failed">Processing failed</span>';
  return verdict ? `<span class="pill ${verdict}">${VERDICT_LABEL[verdict]}</span>` : '<span class="pill">Pending review</span>';
}
export function langTag(lang) {
  return lang ? `<span class="tag">${escapeHtml(lang)}</span>` : "";
}

// ---------------------------------------------------------------- feedback
let toastTimer;
export function toast(msg) {
  const el = $("#toast");
  el.textContent = msg;
  el.classList.add("show");
  clearTimeout(toastTimer);
  toastTimer = setTimeout(() => el.classList.remove("show"), 2400);
}

export async function copyText(text) {
  if (!text) return false;
  try {
    await navigator.clipboard.writeText(text);
  } catch {
    // clipboard API needs a secure context; fall back to a hidden textarea
    const ta = Object.assign(document.createElement("textarea"), { value: text });
    ta.style.position = "fixed"; ta.style.opacity = "0";
    document.body.append(ta); ta.select(); document.execCommand("copy"); ta.remove();
  }
  toast("Copied to clipboard");
  return true;
}

// Accessible tab group: buttons with data-tab, panels with data-panel.
export function bindTabs(root, onChange) {
  const buttons = $$("[role=tab]", root);
  const select = (name) => {
    buttons.forEach((b) => b.setAttribute("aria-selected", String(b.dataset.tab === name)));
    onChange(name);
  };
  buttons.forEach((b) => b.addEventListener("click", () => select(b.dataset.tab)));
  return select;
}
