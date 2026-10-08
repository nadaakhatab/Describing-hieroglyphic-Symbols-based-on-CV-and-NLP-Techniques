"""Server-side chatbot settings and local paths."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Settings:
    api_key: str = field(repr=False)
    model: str = "openai/gpt-oss-20b"
    data_path: Path = ROOT / "data" / "Semantic meaning.json"

    @classmethod
    def from_env(cls) -> "Settings":
        load_dotenv(ROOT / ".env", override=False)
        key = os.getenv("GROQ_API_KEY", "").strip()
        if key == "your_groq_api_key_here":
            key = ""
        model = os.getenv("GROQ_MODEL", "openai/gpt-oss-20b").strip()
        if not model:
            raise ValueError("GROQ_MODEL must not be empty.")
        path = Path(os.getenv("GLYPH_DATA_PATH", "data/Semantic meaning.json"))
        return cls(key, model, path if path.is_absolute() else ROOT / path)
