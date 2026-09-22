FROM python:3.12-slim

# ffmpeg is required by pydub to decode webm/ogg/mp3/etc frontend blobs.
RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir \
    fastapi \
    "uvicorn[standard]" \
    python-multipart \
    pydub \
    numpy \
    tritonclient[http]\
    librosa

WORKDIR /api
COPY app.py utils.py /api/

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "9000"]