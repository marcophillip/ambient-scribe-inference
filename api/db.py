"""Postgres access for stored consultations and doctor reviews.

Schema lives in db/init.sql. All functions are synchronous; FastAPI runs the
routes that call them in its threadpool.
"""

import os

from psycopg.rows import dict_row
from psycopg_pool import ConnectionPool

DATABASE_URL = os.environ.get("DATABASE_URL", "")

# opened lazily so the API still starts (and /health reports it) if the db is down
pool = ConnectionPool(DATABASE_URL, min_size=1, max_size=5, open=False,
                      kwargs={"row_factory": dict_row})

CONSULTATION_FIELDS = (
    "audio_filename", "audio_mime", "audio_bytes", "duration_s",
    "language_requested", "language_detected", "asr_model", "llm_model",
    "transcription", "cleaned_transcript", "corrections", "structured_note",
    "llm_raw_output", "status", "error", "asr_seconds", "llm_seconds",
)

REVIEW_FIELDS = (
    "reviewer", "verdict", "rating", "missing_info", "invented_info",
    "wrong_medication", "transcription_errors", "corrected_note", "comments",
)

# latest review per consultation, plus how many reviews it has
_LATEST_REVIEW = """
    LEFT JOIN LATERAL (
        SELECT verdict, reviewer, created_at AS reviewed_at, count(*) OVER () AS review_count
        FROM reviews r WHERE r.consultation_id = c.id
        ORDER BY r.created_at DESC LIMIT 1
    ) lr ON true
"""


def _conn():
    pool.open(wait=False)
    return pool.connection()


def ping() -> bool:
    try:
        with _conn() as conn:
            conn.execute("SELECT 1")
        return True
    except Exception:
        return False


def insert_consultation(**fields) -> dict:
    cols = [k for k in CONSULTATION_FIELDS if k in fields]
    sql = (f"INSERT INTO consultations ({', '.join(cols)}) "
           f"VALUES ({', '.join('%s' for _ in cols)}) RETURNING *")
    with _conn() as conn:
        return conn.execute(sql, [fields[k] for k in cols]).fetchone()


def _filters(status: str, q: str | None):
    where, params = [], []
    if status == "pending":
        where.append("NOT EXISTS (SELECT 1 FROM reviews r WHERE r.consultation_id = c.id)")
    elif status == "reviewed":
        where.append("EXISTS (SELECT 1 FROM reviews r WHERE r.consultation_id = c.id)")
    if q:
        where.append("(c.transcription ILIKE %s OR c.cleaned_transcript ILIKE %s OR c.structured_note ILIKE %s)")
        params += [f"%{q}%"] * 3
    return (" WHERE " + " AND ".join(where)) if where else "", params


def list_consultations(status: str = "all", q: str | None = None,
                       limit: int = 50, offset: int = 0) -> dict:
    where, params = _filters(status, q)
    sql = f"""
        SELECT c.id, c.created_at, c.duration_s, c.language_detected, c.status,
               left(coalesce(c.cleaned_transcript, c.transcription, ''), 160) AS snippet,
               lr.verdict, lr.reviewer, lr.reviewed_at, coalesce(lr.review_count, 0) AS review_count
        FROM consultations c {_LATEST_REVIEW} {where}
        ORDER BY c.created_at DESC LIMIT %s OFFSET %s
    """
    counts_sql = """
        SELECT count(*) AS all,
               count(*) FILTER (WHERE NOT EXISTS (SELECT 1 FROM reviews r WHERE r.consultation_id = c.id)) AS pending,
               count(*) FILTER (WHERE EXISTS (SELECT 1 FROM reviews r WHERE r.consultation_id = c.id)) AS reviewed
        FROM consultations c
    """
    with _conn() as conn:
        items = conn.execute(sql, params + [limit, offset]).fetchall()
        counts = conn.execute(counts_sql).fetchone()
    return {"items": items, "counts": counts}


def get_consultation(consultation_id: int) -> dict | None:
    with _conn() as conn:
        row = conn.execute("SELECT * FROM consultations WHERE id = %s", [consultation_id]).fetchone()
        if row is None:
            return None
        row["reviews"] = conn.execute(
            "SELECT * FROM reviews WHERE consultation_id = %s ORDER BY created_at DESC",
            [consultation_id],
        ).fetchall()
    return row


def add_review(consultation_id: int, **fields) -> dict:
    cols = ["consultation_id"] + [k for k in REVIEW_FIELDS if k in fields]
    values = [consultation_id] + [fields[k] for k in cols[1:]]
    sql = (f"INSERT INTO reviews ({', '.join(cols)}) "
           f"VALUES ({', '.join('%s' for _ in cols)}) RETURNING *")
    with _conn() as conn:
        return conn.execute(sql, values).fetchone()


def next_pending(after_id: int | None = None) -> int | None:
    """Id of the next unreviewed consultation (newest first), skipping `after_id`."""
    sql = """SELECT c.id FROM consultations c
             WHERE NOT EXISTS (SELECT 1 FROM reviews r WHERE r.consultation_id = c.id)
               AND c.id <> coalesce(%s, -1)
             ORDER BY c.created_at DESC LIMIT 1"""
    with _conn() as conn:
        row = conn.execute(sql, [after_id]).fetchone()
    return row["id"] if row else None
