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

## Option A: `mbaza_asr_nemo` (pure NeMo, start here)

Loads the model via NeMo and calls its own `.transcribe()` end-to-end —
feature extraction, encoder/decoder forward pass, and CTC decode are all
handled internally. No export step, no ONNX file to manage.

### 1. Build the image

```bash
docker build -t triton-whisper-kinyarwanda .
```

(Pick a `tritonserver` base tag in the Dockerfile matching your CUDA driver —
check https://catalog.ngc.nvidia.com/orgs/nvidia/containers/tritonserver/tags.
Note this image includes `nemo_toolkit[asr]`, a large dependency — expect
a slow first build.)

### 2. Run the server

```bash
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  triton-whisper-kinyarwanda \
  tritonserver --model-repository=/models
```

Drop `--gpus all` and switch `KIND_GPU` to `KIND_CPU` in `config.pbtxt` if
you don't have a GPU.

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

### Notes

- Every request round-trips through a temp WAV file on disk (see comments
  in `model.py`) rather than passing the in-memory array directly to
  `transcribe()`, since that method's accepted input types have varied
  across `nemo_toolkit` versions. This adds minor disk I/O overhead per
  request; if you've confirmed your installed version accepts raw
  arrays/tensors directly, you can simplify this.
- Batching is not wired up — one request is transcribed at a time.

---

## Option B: `mbaza_asr_onnx` (ONNX Runtime-accelerated)

NeMo's `export()` only exports the **encoder+decoder acoustic-scoring
graph** to ONNX — audio feature extraction and CTC decoding stay in
Python/NeMo at serving time, since they're cheap and easy to get subtly
wrong if hand-reimplemented.

### 1. Export to ONNX (once, offline)

```bash
pip install "nemo_toolkit[asr]"
python export_mbaza_to_onnx.py \
    --output_dir model_repository/mbaza_asr_onnx/1/onnx_model
```

This produces `model.onnx` (the encoder+decoder graph) and `model.nemo`
(the full checkpoint, used only for preprocessing + CTC decode at serving
time — both files are required, see the "what is model.nemo" note below
if that's unclear).

### 2. Build and run

Same commands as Option A — both models load into the same Triton server
from the same image:

```bash
docker build -t triton-whisper-kinyarwanda .

docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  triton-whisper-kinyarwanda \
  tritonserver --model-repository=/models
```

### 3. Test it

```bash
python client.py path/to/audio.wav mbaza_asr_onnx
```

### Confirmed implementation details

- **ONNX graph inputs**: `audio_signal` (FP32, `[batch, 80, time]` mel
  features) and `length` (INT64, `[batch]`) — verified directly against
  `session.get_inputs()`. `model.py` maps to these by name, with a
  startup check that raises clearly if a different `nemo_toolkit` version
  ever produces different names.
- **`ctc_decoder_predictions_tensor` signature**: the decode call has a
  `try`/`except` fallback for a `decoder_lengths` keyword that may not
  exist in older/newer `nemo_toolkit` releases. If both branches fail,
  check `help(self.nemo_model.decoding.ctc_decoder_predictions_tensor)`
  for your installed version's exact signature.
- **BPE vs. character vocabulary**: the decode call defers entirely to the
  model's own `decoding` object, so it should handle either tokenizer type
  correctly without extra code from us — worth double-checking output
  text looks sane on a known sample regardless.

### What `model.nemo` actually is

A tar archive (despite the custom extension) bundling the PyTorch
weights, the model config (architecture + preprocessor settings), and the
tokenizer/vocabulary — everything `ASRModel.restore_from()` needs to
reconstruct the full model object. In `mbaza_asr_onnx`, we don't run this
checkpoint's own `.forward()` for the heavy compute (that's `model.onnx`'s
job) — we lean on it only for `.preprocessor` (feature extraction) and
`.decoding` (CTC decode), since reimplementing NeMo's exact preprocessing
math and decode logic by hand risks subtle mismatches with training. You
can inspect it directly with `tar -tvf model.nemo`.

### GPU notes

`onnxruntime-gpu` requires a CUDA/cuDNN version matching what's baked into
the Triton base image. If you hit CUDA-related import or provider-loading
errors, check the compatibility table linked in the Dockerfile comments
and pin an `onnxruntime-gpu` version that matches your container's CUDA
version. If in doubt, start with CPU (`KIND_CPU` in `config.pbtxt`) to
confirm correctness before chasing GPU provider issues — or just use
Option A, which sidesteps ONNX Runtime entirely.

---

## Optimization

- **Batching**: `max_batch_size` is set to `0` (disabled) in both options
  above. To support real batching, pad audio/features to a common length
  before the forward pass.
- **Further speed**: if ONNX Runtime still isn't fast enough, options
  include NVIDIA's TensorRT/TensorRT-LLM (real engine compilation, more
  setup friction, GPU-architecture-specific builds) or exporting with
  further NeMo-specific optimizations (e.g. fused CTC decoding).
- **Health/readiness**: Triton exposes `/v2/health/ready` on port 8000 once
  the model has loaded successfully — useful for k8s readiness probes.

---

## Reference: Whisper-based deployments (earlier model, not required)

This repo was originally built around a different model,
`DigitalUmuganda/whisper_small_kinyarwanda` (a `transformers` Whisper
seq2seq model). Those deployments are kept here for reference/comparison
but aren't needed for Mbaza-ASR.

### `whisper_kinyarwanda` — PyTorch backend

Triton's Python backend loading the HuggingFace model directly and running
`generate()` inside `model.py`.

```bash
docker build -t triton-whisper-kinyarwanda .
docker run --gpus all --rm \
  -p 8000:8000 -p 8001:8001 -p 8002:8002 \
  -v $(pwd)/model_repository:/models \
  triton-whisper-kinyarwanda \
  tritonserver --model-repository=/models

python client.py path/to/audio.wav whisper_kinyarwanda
```

### `whisper_kinyarwanda_onnx` — ONNX Runtime-accelerated

Uses Hugging Face **Optimum**'s `ORTModelForSpeechSeq2Seq`, which exports
Whisper to ONNX and reimplements `generate()` on top of ONNX Runtime
sessions (KV-cache handling included, not hand-rolled).

```bash
pip install "optimum[onnxruntime]"   # or optimum[onnxruntime-gpu] with a GPU
python export_to_onnx.py \
    --output_dir model_repository/whisper_kinyarwanda_onnx/1/onnx_model

# then build/run as above, and:
python client.py path/to/audio.wav whisper_kinyarwanda_onnx
```

**What you get vs. don't get here:**
- ✅ Real ONNX Runtime-accelerated encoder + decoder inference
- ✅ Correct KV-cache handling (via Optimum, not hand-rolled)
- ❌ Not a "pure" native Triton `onnxruntime` backend + BLS orchestration —
  that would give more fine-grained batching control but requires
  manually wiring cross-attention caches between calls. Worth doing only
  if you've profiled this version and need more.

**Note on TensorRT-LLM**: NVIDIA also maintains a Whisper pipeline in
`tensorrtllm_backend`, but it expects OpenAI's original Whisper `.pt`
checkpoint naming convention, not arbitrary HuggingFace fine-tunes — using
it with a custom checkpoint like the original Whisper model here would
require an extra, somewhat fragile conversion step first.
