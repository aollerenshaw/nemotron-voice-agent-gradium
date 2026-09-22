# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Gradium voice catalog for the Services / Voice Settings UI.

The blueprint's ``/api/tts-config`` endpoint discovers voices by opening a Riva
gRPC connection to the TTS NIM. Gradium is an external HTTP/WebSocket API, so it
needs its own discovery path: ``GET https://api.gradium.ai/api/voices/`` with the
``x-api-key`` header.

Returns the same shape the client already consumes:
``{"languages": [...], "voices": [{"id", "name", "language"}], "defaultVoiceId"}``.
"""

import json
import os
import urllib.error
import urllib.request

from loguru import logger

import config_store

GRADIUM_API_BASE = os.getenv("GRADIUM_API_BASE", "https://api.gradium.ai/api").rstrip("/")
GRADIUM_VOICES_TIMEOUT_SECS = float(os.getenv("GRADIUM_VOICES_TIMEOUT_SECS", "10"))
_CACHE_KEY = "tts:gradium:voices"

# Gradium reports a bare language code; the client expects BCP-47 locales.
_LANGUAGE_MAP = {
    "en": "en-US",
    "fr": "fr-FR",
    "de": "de-DE",
    "es": "es-ES",
    "pt": "pt-BR",
}
_DEFAULT_LANGUAGE = "en-US"


def _to_bcp47(code: str | None) -> str:
    """Map a Gradium language code to a BCP-47 locale."""
    if not code:
        return _DEFAULT_LANGUAGE
    normalized = str(code).strip().lower().replace("_", "-")
    if "-" in normalized:
        base, region = normalized.split("-", 1)
        return f"{base}-{region.upper()}"
    return _LANGUAGE_MAP.get(normalized, f"{normalized}-{normalized.upper()}")


def _fetch_voices() -> list[dict]:
    api_key = os.getenv("GRADIUM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GRADIUM_API_KEY is not set")
    request = urllib.request.Request(
        f"{GRADIUM_API_BASE}/voices/",
        headers={"x-api-key": api_key, "x-api-source": "nemotron-voice-agent"},
    )
    with urllib.request.urlopen(request, timeout=GRADIUM_VOICES_TIMEOUT_SECS) as response:
        payload = json.load(response)
    return payload if isinstance(payload, list) else []


def fetch_gradium_tts_config(default_voice_id: str = "") -> dict:
    """Return the Gradium voice catalog in the blueprint's tts-config shape.

    Cached in ``config_store`` after the first successful call. Blocking on
    purpose: callers run it through ``_run_blocking``.
    """
    cached = config_store.get(_CACHE_KEY)
    if isinstance(cached, dict) and cached.get("voices"):
        result = dict(cached)
        if default_voice_id:
            result["defaultVoiceId"] = default_voice_id
        config_store.set("tts", result)
        return result

    try:
        raw_voices = _fetch_voices()
    except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
        logger.warning(f"Gradium voice catalog fetch failed: {exc}")
        return {
            "languages": [_DEFAULT_LANGUAGE],
            "voices": [{"id": default_voice_id, "name": "Default", "language": _DEFAULT_LANGUAGE}]
            if default_voice_id
            else [],
            "defaultVoiceId": default_voice_id,
            "server": "gradium",
            "error": str(exc),
        }

    voices: list[dict] = []
    for voice in raw_voices:
        if not isinstance(voice, dict):
            continue
        uid = str(voice.get("uid") or "").strip()
        if not uid:
            continue
        voices.append(
            {
                "id": uid,
                "name": str(voice.get("name") or uid),
                "language": _to_bcp47(voice.get("language")),
            }
        )

    result = {
        "languages": sorted({voice["language"] for voice in voices}) or [_DEFAULT_LANGUAGE],
        "voices": sorted(voices, key=lambda voice: (voice["language"], voice["name"])),
        "defaultVoiceId": default_voice_id,
        "server": "gradium",
    }
    config_store.set(_CACHE_KEY, result)
    config_store.set("tts", result)
    logger.info(f"Gradium voice catalog loaded — {len(result['languages'])} languages, {len(voices)} voices")
    return result
