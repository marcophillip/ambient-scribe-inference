# Ambient Scribe — inference stack

Records a doctor verbal report after interaction with the patient, transcribes it, and extracts the clinical
content into a structured note. Everything runs on-premises with Docker Compose.

| Service | Container | Port | |
| :--- | :--- | :--- | :--- |
| `triton` | `ambient-scribe` | 8003 (HTTP) | NVIDIA Triton serving the speech model `nemotron_asr` |
| `vllm` | `vllm-ambient` | 8004 | vLLM serving the note-structuring LLM (`google/gemma-3-4b-it`) |
| `fastapi` | `fastapi-server` | 9000 | API + web frontend (`/`, `/transcribe`, `/structure`, `/health`, `/models`) |

## Speech model

**Nemotron 3.5 ASR streaming 0.6B** (`nvidia/nemotron-3.5-asr-streaming-0.6b`),
deployed in Triton as `nemotron_asr`. English and French, see below.

---

## Repo layout

Triton requires a `model_repository/<model_name>/<version>/model.py` layout,
with `config.pbtxt` one level above the version folder.

```
ambient-scribe-inference/
├── Dockerfile                 # Triton + NeMo image
├── docker-compose.yml
├── api/                       # FastAPI service
│   ├── app.py                 # endpoints
│   ├── utils.py               # audio decoding, Triton client, note structuring
│   ├── prompts.py             # clinical extraction prompts
│   └── static/index.html      # frontend
└── model_repository/
    └── nemotron_asr/
        ├── config.pbtxt
        └── 1/
            ├── model.py
            └── model.nemo     # checkpoint, downloaded on first start
```

---

## Run

```bash
docker compose up -d --build
```

On first start `model_repository/nemotron_asr/1/model.py` downloads the
checkpoint from the Hugging Face Hub (set `HF_TOKEN` in `.env`) and saves it
as `model.nemo`, so later starts load it locally. The first image build is
slow: `nemo_toolkit[asr]` is a large dependency.

Open the frontend at `http://localhost:9000/`. The microphone only works on
`https://` or `http://localhost`, so from another machine use an SSH tunnel or
VS Code port forwarding.

## Test

```bash
curl -F file=@audio.wav -F language=auto http://localhost:9000/transcribe
curl -F file=@audio.wav -F language=auto http://localhost:9000/structure
```

## Nemotron 3.5 ASR (`nemotron_asr`) — English + French

`nvidia/nemotron-3.5-asr-streaming-0.6b`: a 600M cache-aware
FastConformer encoder with an **RNNT** decoder and language-ID prompt
conditioning. English (`en-US`, `en-GB`) and French (`fr-FR`, `fr-CA`) are
both in its top "transcription-ready" tier, and it emits punctuation and
capitalization natively. FLEURS WER: 7.91 (en) / 9.03 (fr).

### Language prompting

The model is **prompt-conditioned**, so every call carries a language ID. The
Triton wrapper exposes that as an optional `LANGUAGE` input:

- a locale — `en-US`, `fr-FR`, ... — pins the language;
- `auto` (the default) detects it and appends a `<xx-XX>` tag after the
  terminal punctuation. `model.py` strips that tag from `TRANSCRIPTION`
  and returns it separately as `LANGUAGE_DETECTED`.

For a clinic seeing both English and French speakers, `auto` is the right
default: one deployment, each utterance labeled with what was spoken.


---
