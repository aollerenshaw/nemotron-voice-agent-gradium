# SPDX-FileCopyrightText: Copyright (c) 2024–2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: BSD-2-Clause

"""Provider-aware TTS service factory.

Keeps per-vendor wiring out of the example pipelines. The active provider comes
from the ``provider`` field on the selected TTS catalog entry
(``services.cloud.yaml`` / ``services.local.yaml``). Entries with no ``provider``
keep the original NVIDIA (Riva/Magpie) behaviour, so nothing changes for the
built-in catalog entries.

Supported providers:
  - ``nvidia`` (default): ``NvidiaTTSService`` over gRPC (NVCF or local NIM).
  - ``gradium``: ``GradiumTTSService`` over Gradium's public WebSocket API.
"""

import os

from loguru import logger
from pipecat.services.nvidia.tts import NvidiaTTSService, NvidiaTTSSettings
from pipecat.services.tts_service import TTSService

from examples.shared.nemotron_speech_text_filter import NemotronSpeechTextFilter
from utils import is_nvcf, load_ipa_dictionary, normalize_lang_code

GRADIUM_DEFAULT_URL = "wss://api.gradium.ai/api/speech/tts"


def resolve_tts_provider(body: dict, default_tts: dict) -> str:
    """Return the lowercase provider id for the active TTS selection."""
    provider = body.get("tts_provider", "") or default_tts.get("provider", "") or "nvidia"
    return str(provider).strip().lower()


def build_tts_service(body: dict, default_tts: dict) -> TTSService:
    """Build the TTS service for the active selection.

    Args:
        body: The sanitized session body (``runner_args.body``).
        default_tts: The catalog entry returned by ``load_service_entry("tts", "")``.
    """
    provider = resolve_tts_provider(body, default_tts)
    if provider == "gradium":
        return _build_gradium_tts(body, default_tts)
    if provider not in ("", "nvidia"):
        logger.warning(f"Unknown TTS provider {provider!r}; falling back to the NVIDIA provider")
    return _build_nvidia_tts(body, default_tts)


def _build_gradium_tts(body: dict, default_tts: dict) -> TTSService:
    """Build a Gradium WebSocket TTS service.

    Ships with Pipecat 1.5.0 (``pipecat.services.gradium``) and needs no extra
    dependency: the ``gradium`` extra is empty and the transport is ``websockets``,
    already pulled in by the ``websocket`` extra.
    """
    from pipecat.services.gradium.tts import GradiumTTSService

    api_key = os.getenv("GRADIUM_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError("GRADIUM_API_KEY is not set; it is required by the Gradium TTS provider")

    url = body.get("tts_server", "") or default_tts.get("server", "") or GRADIUM_DEFAULT_URL
    voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    model = body.get("tts_model", "") or default_tts.get("model", "") or "default"
    # Optional JSON string of Gradium generation options (speed, temperature,
    # pronunciation_id, ...). Catalog-only: never accepted from the client body.
    json_config = default_tts.get("json_config") or None

    settings_kwargs: dict = {"model": model}
    if voice:
        settings_kwargs["voice"] = voice

    tts = GradiumTTSService(
        api_key=api_key,
        url=url,
        json_config=json_config,
        settings=GradiumTTSService.Settings(**settings_kwargs),
        text_filters=[NemotronSpeechTextFilter()],
    )

    # Gradium selects the language from the voice and the text itself, so the
    # catalog `language_code` is informational only and is not forwarded.
    logger.info(
        f"TTS: provider=gradium, url={url}, voice={voice or '(pipecat default)'}, "
        f"model={model}, json_config={'set' if json_config else '(none)'}, "
        f"sample_rate=48000, text_filters=[NemotronSpeechTextFilter]"
    )
    return tts


def _build_nvidia_tts(body: dict, default_tts: dict) -> TTSService:
    """Build the stock NVIDIA (Riva/Magpie) TTS service.

    This is the block that previously lived inline in each example pipeline,
    moved here unchanged so provider switching is a one-line call.
    """
    tts_server = body.get("tts_server", "") or default_tts.get("server", "grpc.nvcf.nvidia.com:443")
    tts_ssl = is_nvcf(tts_server)
    tts_voice = body.get("tts_voice_id", "") or default_tts.get("voice_id", "")
    tts_synthesis_mode = body.get("tts_synthesis_mode", "")
    raw_tts_function_id = body.get("tts_function_id")
    tts_function_id = (
        str(raw_tts_function_id) if raw_tts_function_id is not None else default_tts.get("function_id", "")
    )
    tts_model = body.get("tts_model", "") or default_tts.get("model", "")
    tts_zero_shot_audio_prompt_file = body.get("tts_zero_shot_audio_prompt_file", "") or default_tts.get(
        "zero_shot_audio_prompt_file", ""
    )
    tts_language_code = body.get("tts_language_code", "") or default_tts.get("language_code", "")
    if tts_language_code:
        tts_language_code = normalize_lang_code(tts_language_code)
    custom_dictionary = load_ipa_dictionary()

    tts_settings_kwargs: dict = {"voice": tts_voice}
    if tts_synthesis_mode:
        tts_settings_kwargs["synthesis_mode"] = tts_synthesis_mode
    if tts_language_code:
        tts_settings_kwargs["language"] = tts_language_code
    tts_kwargs: dict = {
        "api_key": os.getenv("NVIDIA_API_KEY"),
        "server": tts_server,
        "settings": NvidiaTTSSettings(**tts_settings_kwargs),
        "use_ssl": tts_ssl,
        "text_filters": [NemotronSpeechTextFilter()],
        "custom_dictionary": custom_dictionary,
    }
    if tts_function_id or tts_model:
        tts_kwargs["model_function_map"] = {
            "function_id": tts_function_id,
            "model_name": tts_model,
        }
    if tts_zero_shot_audio_prompt_file:
        tts_kwargs["zero_shot_audio_prompt_file"] = tts_zero_shot_audio_prompt_file
    tts = NvidiaTTSService(**tts_kwargs)

    logger.info(
        f"TTS: provider=nvidia, server={tts_server}, ssl={tts_ssl}, voice={tts_voice}, "
        f"model={tts_model or '(pipecat default)'}, function_id={tts_function_id or '(pipecat default)'}, "
        f"synthesis_mode={tts_synthesis_mode or '(pipecat default)'}, "
        f"language={tts_language_code or '(pipecat default)'}, "
        f"zero_shot_audio_prompt_file={tts_zero_shot_audio_prompt_file or '(none)'}, "
        f"text_filters=[NemotronSpeechTextFilter]"
    )
    return tts
