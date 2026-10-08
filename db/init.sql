-- Schema for the Ambient Scribe validation database.
-- Runs once, on the first start of the db container with an empty data folder.

CREATE TABLE consultations (
  id                 BIGSERIAL PRIMARY KEY,
  created_at         TIMESTAMPTZ NOT NULL DEFAULT now(),
  audio_filename     TEXT NOT NULL,      -- file name inside AUDIO_DIR (never a full path)
  audio_mime         TEXT,
  audio_bytes        INTEGER,
  duration_s         REAL,
  language_requested TEXT,
  language_detected  TEXT,
  asr_model          TEXT,
  llm_model          TEXT,
  transcription      TEXT,               -- raw ASR output
  cleaned_transcript TEXT,               -- from <cleaned_transcript>
  corrections        TEXT,               -- from <corrections>
  structured_note    TEXT,               -- from <clinical_note>
  llm_raw_output     TEXT,               -- full model output, kept for audit
  status             TEXT NOT NULL DEFAULT 'ready' CHECK (status IN ('ready', 'failed')),
  error              TEXT,
  asr_seconds        REAL,
  llm_seconds        REAL
);

-- One consultation can be reviewed by several doctors (inter-rater agreement).
CREATE TABLE reviews (
  id                   BIGSERIAL PRIMARY KEY,
  consultation_id      BIGINT NOT NULL REFERENCES consultations(id) ON DELETE CASCADE,
  created_at           TIMESTAMPTZ NOT NULL DEFAULT now(),
  reviewer             TEXT,               -- set once logins exist; NULL = anonymous
  verdict              TEXT NOT NULL CHECK (verdict IN ('approved', 'needs_correction', 'rejected')),
  rating               SMALLINT CHECK (rating BETWEEN 1 AND 5),
  missing_info         BOOLEAN NOT NULL DEFAULT false,
  invented_info        BOOLEAN NOT NULL DEFAULT false,
  wrong_medication     BOOLEAN NOT NULL DEFAULT false,
  transcription_errors BOOLEAN NOT NULL DEFAULT false,
  corrected_note       TEXT,
  comments             TEXT
);

CREATE INDEX reviews_consultation_idx ON reviews (consultation_id, created_at DESC);
CREATE INDEX consultations_created_idx ON consultations (created_at DESC);
