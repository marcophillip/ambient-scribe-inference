from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, Query
from fastapi.encoders import jsonable_encoder
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from tritonclient import http as httpclient
from schemas import ReviewIn, StatusFilter
from utils import transcribe_array, blob_bytes_to_array, structure_notes, parse_llm_output, LLM_MODEL, TARGET_SR
from datetime import datetime, timezone
import db
import os
import time
import uuid


TRITON_URL = os.environ.get("TRITON_URL", "triton:8003")
VLLM_URL = os.environ.get("VLLM_URL")
AUDIO_DIR = os.environ.get("AUDIO_DIR", "/data/audio")

ASR_MODEL = "nemotron_asr"

STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

# extensions we are willing to store, keyed by what the browser/file says
AUDIO_EXTENSIONS = {"webm", "ogg", "wav", "mp3", "m4a", "mp4", "flac", "aac", "opus"}
MIME_EXTENSIONS = {"audio/webm": "webm", "video/webm": "webm", "audio/ogg": "ogg", "audio/wav": "wav",
                   "audio/x-wav": "wav", "audio/mpeg": "mp3", "audio/mp4": "m4a", "audio/flac": "flac",
                   "audio/aac": "aac"}


app = FastAPI(
    title="Ambient Scribe API",
    description="API for Ambient Scribe ASR model"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# logo and other frontend assets
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def frontend():
    """
    serve the new-consultation page
    """
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/review", include_in_schema=False)
async def review_page():
    """
    serve the doctor validation page
    """
    return FileResponse(os.path.join(STATIC_DIR, "review.html"))


@app.get("/health")
def health_check():
    """
    service health check for this API, the Triton Inference Server and the database
    """
    db_ok = db.ping()
    try:
        client = httpclient.InferenceServerClient(url=TRITON_URL)
        triton_ok = client.is_server_live() and client.is_server_ready()
    except Exception:
        triton_ok = False

    healthy = db_ok and triton_ok
    return JSONResponse(content={"status": "healthy" if healthy else "unhealthy",
                                 "triton": triton_ok, "db": db_ok},
                        status_code=200 if healthy else 503)


@app.get("/models")
async def list_models():
    """
    List all available models on the Triton Inference Server
    """
    try:
        client = httpclient.InferenceServerClient(url=TRITON_URL)
        model_metadata = client.get_model_repository_index()  #[{'name': 'nemotron_asr', 'version': '1', 'state': 'READY'}]
        model_names = [model['name'] for model in model_metadata]
        return JSONResponse(content={"models": model_names})
    except Exception as e:
        return JSONResponse(content={"status": f"error: {str(e)}"}, status_code=503)



@app.post("/transcribe")
async def from_frontend(
    file: UploadFile = File(...),
    language: str = Form("auto"),
):
    """Transcribe an uploaded audio blob (nothing is stored).

    `language` can be fr-FR or en-US or auto to auto-detect the language
    """
    blob_bytes = await file.read()
    audio_array = blob_bytes_to_array(blob_bytes)   # already 16kHz mono float32

    text, detected = transcribe_array(
        audio_array, model_name=ASR_MODEL, url=TRITON_URL, language=language
    )

    response = {"transcription": text, "model": ASR_MODEL}
    if detected:
        response["language"] = detected
    return response


def _save_audio(blob_bytes: bytes, filename: str | None, content_type: str | None) -> str:
    """Write the upload to AUDIO_DIR and return the stored file name."""
    ext = (filename or "").rsplit(".", 1)[-1].lower() if filename and "." in filename else ""
    if ext not in AUDIO_EXTENSIONS:
        ext = MIME_EXTENSIONS.get((content_type or "").split(";")[0].strip(), "bin")
    name = f"{datetime.now(timezone.utc):%Y%m%dT%H%M%SZ}_{uuid.uuid4().hex[:8]}.{ext}"
    os.makedirs(AUDIO_DIR, exist_ok=True)
    with open(os.path.join(AUDIO_DIR, name), "wb") as f:
        f.write(blob_bytes)
    return name


# plain def: runs in the threadpool, so a long LLM call does not block the review page
@app.post("/structure")
def structure(
    file: UploadFile = File(...),
    language: str = Form("auto"),
):
    """Transcribe and structure an uploaded audio blob, and store the result.

    `language` can be fr-FR or en-US or auto to auto-detect the language
    """
    blob_bytes = file.file.read()
    audio_filename = _save_audio(blob_bytes, file.filename, file.content_type)
    audio_array = blob_bytes_to_array(blob_bytes)   # already 16kHz mono float32

    t0 = time.perf_counter()
    text, detected = transcribe_array(
        audio_array, model_name=ASR_MODEL, url=TRITON_URL, language=language
    )
    asr_seconds = time.perf_counter() - t0

    record = dict(
        audio_filename=audio_filename, audio_mime=file.content_type, audio_bytes=len(blob_bytes),
        duration_s=len(audio_array) / TARGET_SR, language_requested=language,
        language_detected=detected or None, asr_model=ASR_MODEL, llm_model=LLM_MODEL,
        transcription=text, asr_seconds=asr_seconds,
    )

    # keep the transcript even if the LLM is down: the row is stored as failed
    t0 = time.perf_counter()
    try:
        raw = structure_notes(raw_text=text, llm_url=VLLM_URL)
        blocks = parse_llm_output(raw)
        record.update(status="ready", llm_raw_output=raw,
                      cleaned_transcript=blocks["cleaned_transcript"],
                      corrections=blocks["corrections"],
                      structured_note=blocks["clinical_note"])
    except Exception as e:
        record.update(status="failed", error=f"structuring failed: {e}")
    record["llm_seconds"] = time.perf_counter() - t0

    row = db.insert_consultation(**record)

    response = {"id": row["id"], "created_at": row["created_at"],
                "transcription": text, "model": ASR_MODEL,
                "cleaned_transcript": row["cleaned_transcript"],
                "corrections": row["corrections"],
                "structured_notes": row["structured_note"],
                "duration_s": row["duration_s"]}
    if detected:
        response["language"] = detected
    if row["status"] == "failed":
        response["detail"] = row["error"]
        return JSONResponse(content=jsonable_encoder(response), status_code=503)
    return response


@app.post("/structure-notes-only")
def structure_notes_only(text: str):
    """Structure a transcript that is already text (nothing is stored)."""
    raw = structure_notes(raw_text=text, llm_url=VLLM_URL)
    return {"structured_notes": parse_llm_output(raw)["clinical_note"]}


@app.get("/api/consultations")
def list_consultations(
    status: StatusFilter = "all",
    q: str | None = None,
    limit: int = Query(50, ge=1, le=200),
    offset: int = Query(0, ge=0),
):
    """Stored consultations, newest first, with their latest review verdict."""
    return db.list_consultations(status=status, q=(q or "").strip() or None, limit=limit, offset=offset)


@app.get("/api/consultations/next-pending")
def next_pending(after: int | None = None):
    """Id of the next consultation waiting for review, or null."""
    return {"id": db.next_pending(after)}


@app.get("/api/consultations/{consultation_id}")
def get_consultation(consultation_id: int):
    row = db.get_consultation(consultation_id)
    if row is None:
        raise HTTPException(404, "consultation not found")
    row.pop("llm_raw_output", None)
    return row


@app.get("/api/consultations/{consultation_id}/audio")
def get_audio(consultation_id: int):
    row = db.get_consultation(consultation_id)
    if row is None:
        raise HTTPException(404, "consultation not found")
    # the path is built only from the stored name, and must stay inside AUDIO_DIR
    path = os.path.realpath(os.path.join(AUDIO_DIR, os.path.basename(row["audio_filename"])))
    if not path.startswith(os.path.realpath(AUDIO_DIR) + os.sep) or not os.path.isfile(path):
        raise HTTPException(404, "audio file not found")
    return FileResponse(path, media_type=row["audio_mime"] or None)


@app.post("/api/consultations/{consultation_id}/reviews", status_code=201)
def add_review(consultation_id: int, review: ReviewIn):
    if db.get_consultation(consultation_id) is None:
        raise HTTPException(404, "consultation not found")
    fields = review.model_dump()
    fields["reviewer"] = (fields["reviewer"] or "").strip() or None
    return db.add_review(consultation_id, **fields)
