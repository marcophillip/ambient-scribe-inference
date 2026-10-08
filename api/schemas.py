"""Request schemas for the Ambient Scribe API."""

from typing import Literal

from pydantic import BaseModel, Field

# a doctor's judgement of an AI-generated note
Verdict = Literal["approved", "needs_correction", "rejected"]

# filter for the consultation list: no review yet / at least one review / everything
StatusFilter = Literal["all", "pending", "reviewed"]


class ReviewIn(BaseModel):
    """A doctor's review of one consultation, posted from the Review page."""
    reviewer: str | None = Field(default=None, max_length=120)   # anonymous until logins exist
    verdict: Verdict
    rating: int | None = Field(default=None, ge=1, le=5)
    missing_info: bool = False
    invented_info: bool = False
    wrong_medication: bool = False
    transcription_errors: bool = False
    corrected_note: str | None = None
    comments: str | None = None
