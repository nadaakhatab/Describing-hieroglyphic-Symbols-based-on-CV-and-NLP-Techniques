"""Shared lazy text-to-speech service for stories and chat answers."""

from __future__ import annotations

import io
import re
import wave
from functools import lru_cache


class SpeechError(RuntimeError):
    pass


def clean_text(text: str) -> str:
    return re.sub(r"\s+", " ", re.sub(r"[^\w\s.]", "", text)).strip()


@lru_cache(maxsize=1)
def pipeline():
    try:
        from kokoro import KPipeline
    except ImportError as exc:
        raise SpeechError("Speech dependencies are not installed.") from exc
    return KPipeline(lang_code="a", repo_id="hexgrad/Kokoro-82M")


def synthesize_wav(text: str) -> bytes:
    try:
        import numpy as np
        import soundfile as sf
    except ImportError as exc:
        raise SpeechError("Speech dependencies are not installed.") from exc
    pieces = [output[2] for output in pipeline()(clean_text(text), voice="af_heart") if isinstance(output, tuple) and len(output) >= 3]
    if not pieces:
        raise SpeechError("No audio data was received from the speech model.")
    buffer = io.BytesIO()
    sf.write(buffer, np.concatenate(pieces), 24000, format="WAV")
    return buffer.getvalue()
