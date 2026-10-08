// Review page: list stored consultations, play audio, compare and validate notes.
import {
  $, $$, API_BASE, renderShell, icon, api, renderMarkdown, escapeHtml, fmtDuration, fmtDate,
  fmtRelative, verdictPill, langTag, VERDICT_LABEL, toast, copyText,
  bindTabs, refreshPendingCount,
} from "./common.js";

renderShell("review");
$("#searchIcon").innerHTML = icon("search");

const PAGE = 50;
let filter = "pending", query = "", offset = 0, items = [], selectedId = null, current = null;

// ---------------------------------------------------------------- list
async function loadList({ append = false } = {}) {
  if (!append) offset = 0;
  const params = new URLSearchParams({ status: filter, limit: PAGE, offset });
  if (query) params.set("q", query);
  const listEl = $("#list");
  if (!append) listEl.innerHTML = '<div class="empty"><span class="spinner dark"></span></div>';
  try {
    const { items: page, counts } = await api(`/api/consultations?${params}`);
    items = append ? items.concat(page) : page;
    $("#nPending").textContent = counts.pending;
    $("#nReviewed").textContent = counts.reviewed;
    $("#nAll").textContent = counts.all;
    renderList(page.length === PAGE);
  } catch (e) {
    listEl.innerHTML = `<div class="empty">${icon("alert")}<strong>Could not load consultations</strong><span>${escapeHtml(e.message)}</span></div>`;
  }
}

function renderList(hasMore) {
  const listEl = $("#list");
  if (!items.length) {
    const msg = query ? "No consultations match your search."
      : filter === "pending" ? "Nothing waiting for review." : "No consultations yet.";
    listEl.innerHTML = `<div class="empty">${icon(filter === "pending" && !query ? "check" : "list")}<strong>${msg}</strong>
      ${filter === "pending" && !query ? '<span>New consultations will appear here.</span>' : ""}</div>`;
    return;
  }
  listEl.innerHTML = items.map((c) => `
    <button class="list-item" type="button" data-id="${c.id}" aria-current="${c.id === selectedId}">
      <div class="top">
        <span class="id">#${c.id}</span>
        <span class="when" title="${escapeHtml(fmtDate(c.created_at, { year: true }))}">${escapeHtml(fmtRelative(c.created_at))}</span>
        ${verdictPill(c.verdict, c.status)}
      </div>
      <div class="snippet">${escapeHtml(c.snippet || "(empty transcript)")}</div>
      <div class="sub"><span class="num">${fmtDuration(c.duration_s)}</span>${langTag(c.language_detected)}
        ${c.review_count > 1 ? `<span>· ${c.review_count} reviews</span>` : ""}
        ${c.reviewer ? `<span>· ${escapeHtml(c.reviewer)}</span>` : ""}</div>
    </button>`).join("")
    + (hasMore ? '<div class="list-more"><button class="btn btn-sm" type="button" id="moreBtn">Load more</button></div>' : "");
  $("#moreBtn")?.addEventListener("click", () => { offset += PAGE; loadList({ append: true }); });
}

$("#list").addEventListener("click", (e) => {
  const item = e.target.closest(".list-item");
  if (item) select(Number(item.dataset.id));
});

$$(".filter-tabs button").forEach((b) => b.addEventListener("click", () => {
  filter = b.dataset.filter;
  $$(".filter-tabs button").forEach((x) => x.setAttribute("aria-pressed", String(x === b)));
  loadList();
}));

let searchTimer;
$("#search").addEventListener("input", (e) => {
  clearTimeout(searchTimer);
  searchTimer = setTimeout(() => { query = e.target.value.trim(); loadList(); }, 250);
});
$("#refreshBtn").addEventListener("click", () => { loadList(); if (selectedId) select(selectedId, { keepForm: true }); });

// ---------------------------------------------------------------- detail
function renderEmptyDetail() {
  $("#detail").innerHTML = `<div class="empty" style="height:100%">${icon("doc")}
    <strong>Select a consultation</strong><span>Listen to the recording, check the transcript and validate the clinical note.</span></div>`;
}

async function select(id, { keepForm = false } = {}) {
  selectedId = id;
  if (location.hash !== `#${id}`) history.replaceState(null, "", `#${id}`);
  $$(".list-item").forEach((el) => el.setAttribute("aria-current", String(Number(el.dataset.id) === id)));
  if (!keepForm) $("#detail").innerHTML = '<div class="empty" style="height:100%"><span class="spinner dark"></span></div>';
  try {
    current = await api(`/api/consultations/${id}`);
    renderDetail();
  } catch (e) {
    current = null;
    $("#detail").innerHTML = `<div class="empty" style="height:100%">${icon("alert")}<strong>${e.status === 404 ? `Consultation #${id} not found` : "Could not load consultation"}</strong><span>${escapeHtml(e.message)}</span></div>`;
  }
}

function renderDetail() {
  const c = current;
  const root = $("#detail");
  root.replaceChildren($("#detailTpl").content.cloneNode(true));
  const f = (name) => root.querySelector(`[data-f="${name}"]`);
  const latest = c.reviews[0];

  f("title").textContent = `Consultation #${c.id}`;
  f("pill").innerHTML = verdictPill(latest?.verdict, c.status);
  f("meta").innerHTML = [
    `<span><b>Recorded</b> ${escapeHtml(fmtDate(c.created_at, { year: true }))}</span>`,
    `<span><b>Duration</b> <span class="num">${fmtDuration(c.duration_s)}</span></span>`,
    c.language_detected ? `<span><b>Language</b> ${escapeHtml(c.language_detected)}</span>` : "",
    `<span><b>Models</b> ${escapeHtml(c.asr_model || "–")} · ${escapeHtml(c.llm_model || "–")}</span>`,
  ].join("");

  f("audio").src = `${API_BASE}/api/consultations/${c.id}/audio`;

  // source tabs
  const SOURCES = { cleaned: c.cleaned_transcript, raw: c.transcription };
  const showSource = (name) => {
    const text = SOURCES[name];
    f("source").innerHTML = text ? `<div class="prose">${escapeHtml(text)}</div>`
      : '<span class="muted">Not available for this consultation.</span>';
  };
  const selectSource = bindTabs(f("sourceTabs"), showSource);
  selectSource(c.cleaned_transcript ? "cleaned" : "raw");

  // note: AI version vs editable corrected version
  if (c.status === "failed") {
    f("note").innerHTML = `<div class="banner bad">${icon("alert")}<div>${escapeHtml(c.error || "The note could not be generated.")}</div></div>`;
  } else {
    f("note").innerHTML = renderMarkdown(c.structured_note || "");
  }
  const original = latest?.corrected_note ?? c.structured_note ?? "";
  f("corrected").value = original;
  f("resetNote").addEventListener("click", () => { f("corrected").value = c.structured_note || ""; });
  f("copyNote").addEventListener("click", () => copyText(c.structured_note));

  // rating stars
  let rating = 0;
  const stars = f("stars");
  stars.innerHTML = [1, 2, 3, 4, 5].map((n) =>
    `<button type="button" role="radio" aria-checked="false" aria-label="${n} of 5" data-n="${n}">${icon("star")}</button>`).join("");
  const paintStars = (n) => $$("button", stars).forEach((b) => {
    b.classList.toggle("on", Number(b.dataset.n) <= n);
    b.setAttribute("aria-checked", String(Number(b.dataset.n) === rating));
  });
  stars.addEventListener("click", (e) => {
    const b = e.target.closest("button");
    if (!b) return;
    rating = Number(b.dataset.n) === rating ? 0 : Number(b.dataset.n);
    paintStars(rating);
  });
  stars.addEventListener("mouseover", (e) => { const b = e.target.closest("button"); if (b) paintStars(Number(b.dataset.n)); });
  stars.addEventListener("mouseleave", () => paintStars(rating));

  // form
  const form = f("form");

  const submit = async (next) => {
    const verdict = form.querySelector('input[name="verdict"]:checked')?.value;
    if (!verdict) { toast("Choose a verdict first"); form.querySelector('input[name="verdict"]').focus(); return; }
    const corrected = f("corrected").value.trim();
    const body = {
      verdict, rating: rating || null,
      missing_info: form.missing_info.checked, invented_info: form.invented_info.checked,
      wrong_medication: form.wrong_medication.checked, transcription_errors: form.transcription_errors.checked,
      // only store a corrected note if the doctor actually changed it
      corrected_note: corrected && corrected !== (c.structured_note || "").trim() ? corrected : null,
      comments: form.comments.value.trim() || null,
    };
    [f("save"), f("saveNext")].forEach((b) => (b.disabled = true));
    try {
      await api(`/api/consultations/${c.id}/reviews`, {
        method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(body),
      });
      toast(`Review saved for #${c.id}`);
      refreshPendingCount();
      await loadList();
      if (next) {
        const { id } = await api(`/api/consultations/next-pending?after=${c.id}`);
        if (id) { select(id); return; }
        toast("All consultations have been reviewed");
      }
      select(c.id);
    } catch (e) {
      toast(`Could not save review: ${e.message}`);
      [f("save"), f("saveNext")].forEach((b) => (b.disabled = false));
    }
  };
  form.addEventListener("submit", (e) => { e.preventDefault(); submit(false); });
  f("saveNext").addEventListener("click", () => submit(true));

  // history
  if (c.reviews.length) {
    f("historySection").hidden = false;
    f("historyLabel").textContent = `Reviews (${c.reviews.length})`;
    f("history").innerHTML = c.reviews.map((r) => {
      const flags = [r.missing_info && "Missing information", r.invented_info && "Invented information",
        r.wrong_medication && "Wrong drug / dose", r.transcription_errors && "Transcription errors"].filter(Boolean);
      return `<li>
        <div>
          <div class="row" style="gap:8px">${r.reviewer ? `<span class="who">${escapeHtml(r.reviewer)}</span>` : ""}
            <span class="pill ${r.verdict}">${VERDICT_LABEL[r.verdict]}</span>
            ${r.rating ? `<span class="muted num">${"★".repeat(r.rating)}${"☆".repeat(5 - r.rating)}</span>` : ""}
            <span class="when">${escapeHtml(fmtDate(r.created_at, { year: true }))}</span></div>
          ${flags.length ? `<div class="flags">${flags.map((x) => `<span class="tag">${x}</span>`).join("")}</div>` : ""}
          ${r.comments ? `<div class="comment">${escapeHtml(r.comments)}</div>` : ""}
          ${r.corrected_note ? '<div class="muted" style="margin-top:6px;font-size:14px">Submitted a corrected note.</div>' : ""}
        </div></li>`;
    }).join("");
  }
}

// ---------------------------------------------------------------- start
renderEmptyDetail();
const fromHash = Number(location.hash.slice(1));
if (fromHash) {
  filter = "all";
  $$(".filter-tabs button").forEach((x) => x.setAttribute("aria-pressed", String(x.dataset.filter === "all")));
}
await loadList();
if (fromHash) select(fromHash);
else if (items.length) select(items[0].id);
