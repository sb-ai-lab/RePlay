"""Pydantic schema for codex review structured output."""

from __future__ import annotations

import json
from pathlib import Path

from pydantic import BaseModel, Field, model_validator


class LineRange(BaseModel):
    """Line range for code location."""

    start: int = Field(ge=1)
    end: int = Field(ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "LineRange":
        if self.end < self.start:
            message = "line_range.end must be >= line_range.start"
            raise ValueError(message)
        return self


class CodeLocation(BaseModel):
    """Code location for a single review comment."""

    relative_file_path: str = Field(min_length=1)
    line_range: LineRange


class ReviewComment(BaseModel):
    """Single structured review comment."""

    title: str = Field(min_length=1, max_length=80)
    body: str = Field(min_length=1)
    confidence_score: float = Field(ge=0.0, le=1.0)
    priority: int = Field(ge=0, le=3)
    code_location: CodeLocation


class ReviewResult(BaseModel):
    """Structured review result."""

    comments: list[ReviewComment]


def parse_review_result_text(raw: str) -> ReviewResult:
    payload = json.loads(raw)
    return ReviewResult.model_validate(payload)


def read_review_result(path: Path) -> ReviewResult:
    return ReviewResult.model_validate_json(path.read_text(encoding="utf-8"))


def write_review_result(path: Path, result: ReviewResult) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(result.model_dump_json(indent=2), encoding="utf-8")
