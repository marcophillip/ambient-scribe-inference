// New consultation page: record or upload, process, show result.
import {
  $, $$, API_BASE, renderShell, icon, renderMarkdown, escapeHtml, fmtDuration, fmtBytes,
  copyText, bindTabs, refreshPendingCount,
} from "./common.js";

renderShell("capture");

const recBtn = $("#recBtn"), recTime = $("#recTime"), recState = $("#recState"), level = $("#level");
const processBtn = $("#processBtn"), fileInput = $("#fileInput"), playback = $("#playback");

let mediaRecorder = null, chunks = [], stream = null, audioCtx = null, rafId = null, timerId = null, startedAt = 0;
let currentAudio = null;   // { blob, filename }
let result = null;         // last /structure response
let activeTab = "note";

$("#dropIcon").innerHTML = icon("upload");
recBtn.innerHTML = icon("mic");
$("#copyBtn").innerHTML = `${icon("copy")}Copy`;

// ---------------------------------------------------------------- source switch
$$('input[name="source"]').forEach((r) => r.addEventListener("change", () => {
  const rec = r.value === "record" && r.checked;
  if (r.checked) { $("#recordPane").hidden = !rec; $("#uploadPane").hidden = rec; }
}));

function setAudio(blob, filename, label) {
  currentAudio = { blob, filename };
  playback.src = URL.createObjectURL(blob);
  $("#audioCard").hidden = false;
  $("#audioName").textContent = label;
  $("#audioSize").textContent = fmtBytes(blob.size);
  processBtn.disabled = false;
}

// ---------------------------------------------------------------- recording
function pickMimeType() {
  const types = ["audio/webm;codecs=opus", "audio/webm", "audio/ogg;codecs=opus", "audio/mp4"];
  return types.find((t) => window.MediaRecorder && MediaRecorder.isTypeSupported(t)) || "";
}

async function startRecording() {
  if (!navigator.mediaDevices?.getUserMedia) {
    recState.innerHTML = '<span style="color:var(--bad)">Microphone needs https:// or http://localhost. Use Upload instead.</span>';
    return;
  }
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      audio: { channelCount: 1, echoCancellation: true, noiseSuppression: true },
    });
  } catch (e) {
    recState.textContent = `Microphone unavailable: ${e.message}`;
    return;
  }
  const mimeType = pickMimeType();
  mediaRecorder = new MediaRecorder(stream, mimeType ? { mimeType } : undefined);
  chunks = [];
  mediaRecorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
  mediaRecorder.onstop = () => {
    const type = mediaRecorder.mimeType || "audio/webm";
    const ext = type.includes("ogg") ? "ogg" : type.includes("mp4") ? "m4a" : "webm";
    setAudio(new Blob(chunks, { type }), `recording.${ext}`, `Recording · ${recTime.textContent}`);
    recState.textContent = "Recording ready — review it below, then process";
  };
  mediaRecorder.start(250);

  startLevel(stream);
  startedAt = Date.now();
  tick(); timerId = setInterval(tick, 250);
  recBtn.classList.add("recording");
  recBtn.innerHTML = icon("stop");
  recBtn.setAttribute("aria-label", "Stop recording");
  recState.innerHTML = '<span class="status"><span class="dot live"></span>Recording — press to stop</span>';
  processBtn.disabled = true;
}

function stopRecording() {
  if (mediaRecorder && mediaRecorder.state !== "inactive") mediaRecorder.stop();
  stream?.getTracks().forEach((t) => t.stop());
  stopLevel();
  clearInterval(timerId);
  recBtn.classList.remove("recording");
  recBtn.innerHTML = icon("mic");
  recBtn.setAttribute("aria-label", "Start recording");
}

function tick() { recTime.textContent = fmtDuration((Date.now() - startedAt) / 1000); }

recBtn.addEventListener("click", () => (mediaRecorder?.state === "recording" ? stopRecording() : startRecording()));

function startLevel(s) {
  audioCtx = new AudioContext();
  const analyser = audioCtx.createAnalyser();
  analyser.fftSize = 256;
  audioCtx.createMediaStreamSource(s).connect(analyser);
  const data = new Uint8Array(analyser.frequencyBinCount);
  const ctx = level.getContext("2d");
  const color = getComputedStyle(document.documentElement).getPropertyValue("--accent").trim();
  const bars = 48;
  const draw = () => {
    const w = (level.width = level.clientWidth * devicePixelRatio);
    const h = (level.height = level.clientHeight * devicePixelRatio);
    analyser.getByteFrequencyData(data);
    ctx.clearRect(0, 0, w, h);
    ctx.fillStyle = color;
    const bw = w / bars;
    for (let i = 0; i < bars; i++) {
      const v = data[Math.floor((i / bars) * data.length * 0.7)] / 255;
      const bh = Math.max(2 * devicePixelRatio, v * h);
      ctx.globalAlpha = 0.35 + v * 0.65;
      ctx.fillRect(i * bw + bw * 0.2, (h - bh) / 2, bw * 0.6, bh);
    }
    rafId = requestAnimationFrame(draw);
  };
  draw();
}
function stopLevel() {
  cancelAnimationFrame(rafId);
  audioCtx?.close();
  level.getContext("2d").clearRect(0, 0, level.width, level.height);
}

// ---------------------------------------------------------------- upload
fileInput.addEventListener("change", () => {
  const f = fileInput.files[0];
  if (f) setAudio(f, f.name, f.name);
});
const dz = $("#dropzone");
["dragenter", "dragover"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.add("over"); }));
["dragleave", "drop"].forEach((e) => dz.addEventListener(e, (ev) => { ev.preventDefault(); dz.classList.remove("over"); }));
dz.addEventListener("drop", (ev) => {
  const f = ev.dataTransfer.files[0];
  if (f) setAudio(f, f.name, f.name);
});

// ---------------------------------------------------------------- processing
const STEPS = ["upload", "asr", "llm", "save"];
function setStep(active, error = false) {
  $("#steps").hidden = false;
  const idx = STEPS.indexOf(active);
  $$("#steps li").forEach((li, i) => {
    li.className = i < idx ? "done" : i === idx ? (error ? "error" : "active") : "";
    const m = li.querySelector(".marker");
    m.innerHTML = li.className === "done" ? icon("check") : li.className === "error" ? icon("x") : "";
  });
}
function finishSteps() {
  $$("#steps li").forEach((li) => { li.className = "done"; li.querySelector(".marker").innerHTML = icon("check"); });
}

processBtn.addEventListener("click", async () => {
  if (!currentAudio) return;
  const form = new FormData();
  form.append("file", currentAudio.blob, currentAudio.filename);
  form.append("language", $("#language").value);

  processBtn.disabled = recBtn.disabled = true;
  processBtn.innerHTML = '<span class="spinner"></span>Processing…';
  $("#savedBanner").hidden = $("#errorBanner").hidden = true;
  result = null;
  renderResult();

  // the server does ASR + LLM in one request; advance the indicator on a timer
  // so the doctor sees progress, and settle it from the actual response
  setStep("upload");
  const timers = [setTimeout(() => setStep("asr"), 600), setTimeout(() => setStep("llm"), 4000)];

  try {
    const r = await fetch(`${API_BASE}/structure`, { method: "POST", body: form });
    const body = await r.json().catch(() => null);
    timers.forEach(clearTimeout);
    if (!body || body.transcription === undefined) {
      throw new Error(body?.detail ? (typeof body.detail === "string" ? body.detail : JSON.stringify(body.detail)) : `HTTP ${r.status}`);
    }
    result = body;
    if (r.ok) {
      finishSteps();
    } else {
      setStep("llm", true);
      showError(`The transcript was saved, but the clinical note could not be generated: ${body.detail}`);
    }
    if (body.id) {
      const b = $("#savedBanner");
      b.innerHTML = `${icon("check")}<div>Saved as <b>consultation #${body.id}</b> · ${fmtDuration(body.duration_s)} min of audio.
        <a href="/review#${body.id}">Open in Review ${icon("arrow", 'width="14" height="14" style="vertical-align:-2px"')}</a></div>`;
      b.hidden = false;
      refreshPendingCount();
    }
  } catch (e) {
    timers.forEach(clearTimeout);
    setStep($$("#steps li.active")[0]?.dataset.step || "upload", true);
    showError(`Processing failed: ${e.message}`);
  } finally {
    renderResult();
    processBtn.innerHTML = "Process consultation";
    processBtn.disabled = recBtn.disabled = false;
  }
});

function showError(msg) {
  const b = $("#errorBanner");
  b.innerHTML = `${icon("alert")}<div>${escapeHtml(msg)}</div>`;
  b.hidden = false;
}

// ---------------------------------------------------------------- results
const TAB_CONTENT = {
  note: () => result?.structured_notes,
  cleaned: () => result?.cleaned_transcript,
  raw: () => result?.transcription,
};
const TAB_EMPTY = {
  note: "The structured clinical note will appear here.",
  cleaned: "The cleaned English transcript will appear here.",
  raw: "The speech-recognition output will appear here.",
};

function renderResult() {
  const el = $("#resultContent");
  const text = TAB_CONTENT[activeTab]();
  $("#copyBtn").disabled = !text;
  if (!text) {
    el.innerHTML = `<div class="empty">${icon(activeTab === "note" ? "doc" : "wave")}
      <strong>${result ? "Nothing here for this consultation" : "No consultation processed yet"}</strong>
      <span>${result ? "" : TAB_EMPTY[activeTab]}</span></div>`;
    return;
  }
  el.innerHTML = activeTab === "note"
    ? `<div class="doc">${renderMarkdown(text)}</div>`
    : `<div class="prose">${escapeHtml(text)}</div>`;
}

bindTabs($("#resultTabs"), (name) => { activeTab = name; renderResult(); });
$("#copyBtn").addEventListener("click", () => copyText(TAB_CONTENT[activeTab]()));
renderResult();
