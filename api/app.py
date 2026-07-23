from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File
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
    

@app.post("/transcribe")
async def from_frontend(file: UploadFile = File(...)):
    # client = httpclient.InferenceServerClient(url="localhost:8000")
    # model_ready = client.is_model_ready(model_name=model_name)
    # if not model_ready:
    #     return JSONResponse(content={"status": "model not ready"})
    blob_bytes = await file.read()
    audio_array = blob_bytes_to_array(blob_bytes)   # already 16kHz mono float32
    text = transcribe_array(audio_array, model_name="mbaza_asr", url=TRITON_URL)  # or however you're calling it now
    return {"transcription": text}