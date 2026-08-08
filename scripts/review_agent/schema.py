"""Pydantic schema for review-agent structured output."""

from __future__ import annotations

from pydantic import BaseModel, Field, StrictFloat, StrictInt, model_validator


class LineRange(BaseModel):
    """Line range for code location."""

    start: StrictInt = Field(ge=1)
    end: StrictInt = Field(ge=1)

    @model_validator(mode="after")
    def validate_bounds(self) -> "LineRange":
        """Validate that the end line is not smaller than the start line."""
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
    confidence_score: StrictFloat = Field(ge=0.0, le=1.0)
    priority: StrictInt = Field(ge=0, le=3)
    code_location: CodeLocation


class ReviewResult(BaseModel):
    """Structured review result."""

    comments: list[ReviewComment]
