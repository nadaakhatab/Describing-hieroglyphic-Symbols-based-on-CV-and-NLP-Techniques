"""Validated API request and response shapes."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class Turn(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    role: Literal["user", "assistant"]
    content: str = Field(min_length=1, max_length=2000)


class ChatRequest(BaseModel):
    model_config = ConfigDict(str_strip_whitespace=True, extra="forbid")
    message: str = Field(min_length=1, max_length=2000)
    history: list[Turn] = Field(default_factory=list, max_length=10)
    glyph_codes: list[str] = Field(default_factory=list, max_length=6)

    @field_validator("glyph_codes")
    @classmethod
    def validate_codes(cls, codes: list[str]) -> list[str]:
        if any(not code.strip() or len(code) > 40 for code in codes):
            raise ValueError("Each glyph identifier must contain 1–40 characters.")
        return [code.strip() for code in codes]


class ChatResponse(BaseModel):
    response: str
    context_keys: list[str] = Field(default_factory=list)
    knowledge_loaded: bool
