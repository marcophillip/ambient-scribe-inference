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

## Nemotron 3.5 ASR (`nemotron_asr`) — English + French

`nvidia/nemotron-3.5-asr-streaming-0.6b`: a 600M cache-aware
FastConformer encoder with an **RNNT** decoder and language-ID prompt
conditioning. English (`en-US`, `en-GB`) and French (`fr-FR`, `fr-CA`) are
both in its top "transcription-ready" tier, and it emits punctuation and
capitalization natively. FLEURS WER: 7.91 (en) / 9.03 (fr).

### How it differs from `mbaza_asr`

It is **prompt-conditioned**, so every call carries a language ID. The
Triton wrapper exposes that as an optional `LANGUAGE` input:

- a locale — `en-US`, `fr-FR`, ... — pins the language;
- `auto` (the default) detects it and appends a `<xx-XX>` tag after the
  terminal punctuation. `model.py` strips that tag from `TRANSCRIPTION`
  and returns it separately as `LANGUAGE_DETECTED`.

For a clinic seeing both English and French speakers, `auto` is the right
default: one deployment, each utterance labeled with what was spoken.

### Latency vs. accuracy

`NEMOTRON_ATT_CONTEXT_SIZE` is `[left_frames, right_frames]` in 80ms
frames; the right value sets the streaming chunk size.

| Setting | Chunk | Notes |
| :--- | :--- | :--- |
| `[56,0]` | 80ms | lowest latency |
| `[56,3]` | 320ms | balanced |
| `[56,13]` | 1120ms | **default here** — best WER |

This deployment answers whole-utterance HTTP requests, so the extra
lookahead costs nothing and `[56,13]` is the right pick. Lower it only if
you move to true chunked streaming (see below).

### Calling it

```bash
curl -F file=@consult.wav -F model_name=nemotron_asr -F language=auto \
  http://localhost:9000/transcribe
# {"transcription": "Bonjour, comment allez-vous ?", "model": "nemotron_asr", "language": "fr-FR"}
```

`language` is ignored for models that are not prompt-conditioned, so
`model_name=mbaza_asr` still works unchanged.

### Note on the NeMo version

The class this checkpoint loads as (`EncDecRNNTBPEModelWithPrompt`) is
newer than the released `nemo_toolkit` on PyPI, so the Dockerfile installs
NeMo from git `main`. Expect a slow first build.

### Not yet wired up: real streaming

This is a *streaming* checkpoint being run in offline mode — which the
model card explicitly supports, and which is all the current
request/response API needs. Genuine low-latency streaming would mean a
WebSocket endpoint plus Triton decoupled mode holding per-connection
encoder cache, and `conformer_stream_step` in place of `transcribe`. That
is a separate piece of work.

---

### Optional: pin a local checkpoint

Drop a `model.nemo` file directly into `model_repository/mbaza_asr_nemo/1/`
(e.g. one you exported yourself via `asr_model.save_to(...)`, or a
fine-tuned variant) and `model.py` will load that instead of downloading
from the Hub — useful for reproducible deployments or fully offline
environments.


---
