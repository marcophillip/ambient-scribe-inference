from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Form
from tritonclient import http as httpclient
from utils import transcribe_array, blob_bytes_to_array
import os

TRITON_URL = os.environ.get("TRITON_URL", "triton:8000")

app = FastAPI(
    title="Ambient Scribe API",
    description="API for Ambient Scribe ASR model"
)


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

# Models that take a language-ID prompt. Others ignore the `language` form field.
PROMPTED_MODELS = {"nemotron_asr"}


@app.post("/transcribe")
async def from_frontend(
    file: UploadFile = File(...),
    model_name: str = Form(DEFAULT_MODEL),
    language: str = Form("auto"),
):
    """Transcribe an uploaded audio blob.

    `language` applies to prompt-conditioned models such as nemotron_asr:
    pass a locale ("en-US", "fr-FR") to pin it, or "auto" to detect and
    return the detected locale alongside the text.
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