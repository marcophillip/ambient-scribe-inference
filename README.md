# Deploying the Kinyarwanda ASR Models

This repository contains production configuration setups to serve Kinyarwanda Automatic Speech Recognition (ASR) models via NVIDIA Triton Inference Server. 

## Models Included

1. **Mbaza-ASR (`mbaza_asr_nemo`)**
   - **Architecture:** NeMo Conformer-CTC (`DigitalUmuganda/Mbaza-ASR-Afrivoice-660h`).
   - **Characteristics:** CTC-based output with zero autoregressive decoding loops. Audio feature extraction (mel-spectrogram) and the CTC greedy-decode step are handled directly in Python inside Triton's execution wrapper.
   
2. **Whisper Kinyarwanda (`whisper_kinyarwanda`)**
   - **Architecture:** OpenAI Whisper Seq2Seq framework fine-tuned for Kinyarwanda speech.
   - **Characteristics:** Standard Encoder/Decoder architecture using typical autoregressive text generation parameters.

---

## Repo Layout

Triton requires a strict layout configuration structure. The repository is organized as follows:

## Repo layout

```
ambient-scribe-inference/
├── Dockerfile
├── client.py            
└── model_repository/
    ├── mbaza_asr/                  
    │   ├── config.pbtxt
    │   └── 1/
    │       ├── model.py
    │       └── model.nemo             
    ├── whisper/                  
    │   ├── config.pbtxt
    │   └── 1/
    │       ├── model.py
    │       └── model.whisper            


```

Triton requires this exact shape: a `model_repository/<model_name>/<version>/model.py`
layout, with `config.pbtxt` one level above the version folder.

---

## Run the server

Loads the model via NeMo and calls its own `.transcribe()` end-to-end —
feature extraction, encoder/decoder forward pass, and CTC decode are all
handled internally. No export step, no ONNX file to manage.

### 1. Build the image

```bash
sudo docker compose build
```

(Pick a `tritonserver` base tag in the Dockerfile matching your CUDA driver —
check https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags.
Note this image includes `nemo_toolkit[asr]`, a large dependency — expect
a slow first build.)

### 2. Run the server

```bash
sudo docker compose up
```

`model.py` calls
`nemo_asr.models.ASRModel.from_pretrained("DigitalUmuganda/Mbaza-ASR-Afrivoice-660h")`
on first load, downloading the checkpoint from the Hugging Face Hub — make
sure the container has internet access, or see "pin a local checkpoint"
below to avoid the cold-start download.

### 3. Test it

```bash
pip install tritonclient[http] soundfile
python client.py path/to/audio.wav mbaza_asr_nemo
```

Audio must be **mono, 16kHz** float32. Resample beforehand if needed, e.g.
with `librosa.resample` or `ffmpeg`.

### Optional: pin a local checkpoint

Drop a `model.nemo` file directly into `model_repository/mbaza_asr_nemo/1/`
(e.g. one you exported yourself via `asr_model.save_to(...)`, or a
fine-tuned variant) and `model.py` will load that instead of downloading
from the Hub — useful for reproducible deployments or fully offline
environments.


---
