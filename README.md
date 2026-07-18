# Deploying DigitalUmuganda/Mbaza-ASR-Afrivoice-660h on Triton

`DigitalUmuganda/Mbaza-ASR-Afrivoice-660h` is a **NeMo Conformer-CTC**
model for Kinyarwanda — not a `transformers` seq2seq model. Key
implications for deployment:

- Loaded via `nemo_asr.models.ASRModel.from_pretrained(...)`, not
  `AutoModelForSpeechSeq2Seq`.
- **CTC output, no autoregressive decoding.** The encoder produces one set
  of logits per audio frame; decoding is a simple greedy
  argmax-then-collapse-blanks step, not a beam-search `generate()` loop.
- Audio feature extraction (mel-spectrogram) and the CTC decode step are
  NeMo-specific and stay in Python at serving time even in the ONNX
  version below — only the encoder+decoder acoustic-scoring graph gets
  exported.

This repo has **two deployment options** for this model:

| | `mbaza_asr_onnx` | `mbaza_asr_nemo` |
|---|---|---|
| Encoder/decoder inference | ONNX Runtime (faster) | Native NeMo/PyTorch |
| Setup | Requires an offline export step | Works out of the box |
| Correctness risk | Slightly higher (relies on ONNX Runtime + a hybrid pre/post-processing split) | Lowest (exact code path NeMo's maintainers test) |
| Good for | Production, once verified | Getting started, baseline to compare ONNX output against |

If you're just getting started, `mbaza_asr_nemo` is the simplest path —
jump to that section below. Use `mbaza_asr_onnx` once you've confirmed
you need the extra speed.

There's also a **Whisper-based reference deployment** further down (a
different, earlier model this repo was originally built around) — kept
for comparison, not required for the Mbaza-ASR model.

## Repo layout

```
triton_whisper_kinyarwanda/
├── Dockerfile
├── client.py
├── export_mbaza_to_onnx.py
├── export_to_onnx.py                    (Whisper-only, see bottom section)
└── model_repository/
    ├── mbaza_asr_nemo/                  <- start here
    │   ├── config.pbtxt
    │   └── 1/
    │       ├── model.py
    │       └── model.nemo               (optional, see below)
    ├── mbaza_asr_onnx/                   <- faster, once verified
    │   ├── config.pbtxt
    │   └── 1/
    │       ├── model.py
    │       └── onnx_model/               <- created by export_mbaza_to_onnx.py
    ├── whisper_kinyarwanda/              <- reference only
    └── whisper_kinyarwanda_onnx/         <- reference only
```

Triton requires this exact shape: a `model_repository/<model_name>/<version>/model.py`
layout, with `config.pbtxt` one level above the version folder.

---

## `mbaza_asr_nemo` (pure NeMo, start here)

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
