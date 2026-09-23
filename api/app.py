from fastapi.responses import JSONResponse, FileResponse
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from tritonclient import http as httpclient
from utils import transcribe_array, blob_bytes_to_array
import os

TRITON_URL = os.environ.get("TRITON_URL", "triton:8003")
STATIC_DIR = os.path.join(os.path.dirname(__file__), "static")

app = FastAPI(
    title="Ambient Scribe API",
    description="API for Ambient Scribe ASR model"
)

# allow the frontend to be opened from another origin (e.g. a local file or dev server)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/", include_in_schema=False)
async def frontend():
    """
    serve the recording / upload frontend
    """
    return FileResponse(os.path.join(STATIC_DIR, "index.html"))


@app.get("/health")
async def health_check():
    """
    service health check monitoring for both this API and the Triton Inference Server
    """
    try:
        client = httpclient.InferenceServerClient(url=TRITON_URL)
        if client.is_server_live() and client.is_server_ready():
            return JSONResponse(content={"status": "healthy"})
        else:
            return JSONResponse(content={"status": "unhealthy", 
                                         "triton_live": client.is_server_live(), 
                                         "triton_ready": client.is_server_ready()}, 
                                         status_code=503)
        
    except Exception as e:
        return JSONResponse(content={"status": f"error: {str(e)}"}, status_code=503)
    

@app.get("/models")
async def list_models():
    """
    List all available models on the Triton Inference Server
    """
    try:
        client = httpclient.InferenceServerClient(url=TRITON_URL)
        model_metadata = client.get_model_repository_index()  #[{'name': 'mbaza_asr', 'version': '1', 'state': 'READY'}, ...]
        model_names = [model['name'] for model in model_metadata]
        return JSONResponse(content={"models": model_names})
    except Exception as e:
        return JSONResponse(content={"status": f"error: {str(e)}"}, status_code=503)
    

DEFAULT_MODEL = os.environ.get("DEFAULT_ASR_MODEL", "nemotron_asr")

PROMPTED_MODELS = {"nemotron_asr"}


@app.post("/transcribe")
async def from_frontend(
    file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL),
    language: str = Form("auto"),
):
    """Transcribe an uploaded audio blob.

    `language` can be fr-FR or en-EN or auto to auto-detect the language
    """
    blob_bytes = await file.read()
    audio_array = blob_bytes_to_array(blob_bytes)   # already 16kHz mono float32

    lang = language if model_name in PROMPTED_MODELS else None
    text, detected = transcribe_array(
        audio_array, model_name=model_name, url=TRITON_URL, language=lang
    )

    response = {"transcription": text, "model": model_name}
    if detected:
        response["language"] = detected
    return response