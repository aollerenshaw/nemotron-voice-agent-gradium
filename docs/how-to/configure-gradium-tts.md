# Configure Gradium TTS

The pipeline's TTS slot normally runs an NVIDIA Speech NIM over gRPC (Magpie, Chatterbox).
This fork adds a **`provider`** field to TTS catalog entries so a slot can instead be served by
an external API. The first such provider is [Gradium](https://gradium.ai), a hosted low-latency
TTS service reached over WebSocket.

Catalog entries **without** a `provider` field behave exactly as before, so every built-in
NVIDIA entry is unaffected.

| Provider | Pipecat service | Transport | GPU |
|---|---|---|---|
| `nvidia` (default) | `NvidiaTTSService` | gRPC — NVCF or local NIM | ~14 GB if self-hosted |
| `gradium` | `GradiumTTSService` | WebSocket — `wss://api.gradium.ai` | none |

> **Data leaves your network.** Every synthesized reply is sent to Gradium as text over TLS.
> Confirm that is acceptable for your deployment before enabling it.

## Requirements

- A Gradium API key from [studio.gradium.ai](https://studio.gradium.ai)
- Nothing else. `GradiumTTSService` ships in the pinned `pipecat-ai==1.5.0` and its `gradium`
  extra is empty, so there is **no dependency or lockfile change**. Do not add the extra to
  `pyproject.toml` — the Dockerfile runs `uv sync --frozen` and an unlocked edit breaks the build.

## Setup

### 1. Add the API key

In `.env` at the repo root:

```bash
GRADIUM_API_KEY=gd_your_key_here
```

Exporting it in your shell is not enough — the app container's environment comes from
`env_file: .env`.

### 2. Pick a voice

```bash
curl -s -H "x-api-key: $(grep '^GRADIUM_API_KEY=' .env | cut -d= -f2-)" \
  https://api.gradium.ai/api/voices/ | python3 -m json.tool | grep -E '"uid"|"name"|"language"'
```

### 3. Configure the catalog entry

In the example's `services.cloud.yaml` (for example `src/examples/generic/services.cloud.yaml`):

```yaml
tts:
  gradium-tts:
    name: "Gradium TTS (external API)"
    provider: gradium
    server: "wss://api.gradium.ai/api/speech/tts"
    voice_id: "<uid from step 2>"
    model: "default"
    function_id: ""
    # Optional Gradium generation options (catalog-only, never taken from the client):
    # json_config: '{"speed": 1.0, "temperature": 0.6}'
```

Use `services.cloud.yaml`, not `services.local.yaml`: the local catalog is filtered by a TCP
reachability probe per platform, while the cloud catalog is always loaded.

For regional pinning, use `wss://eu.api.gradium.ai/...` or `wss://us.api.gradium.ai/...`.

### 4. Make it the default (optional)

In `examples_registry.yaml`:

```yaml
  generic-assistant:
    defaults:
      tts: [gradium-tts]
```

Otherwise select it at runtime in the Services tab.

### 5. Apply

`docker-compose.yml` bind-mounts `./src` and `./examples_registry.yaml`, so no rebuild is needed:

```bash
docker compose --profile <your-profile> restart <app-service>
```

## Changing the voice

- **At runtime:** use the Voice Settings panel. `GradiumTTSService` reconnects its socket with
  the new voice mid-session.
- **Permanently:** edit `voice_id` in the catalog entry and restart the app container.
- **Several voices side by side:** add multiple entries (`gradium-tts`, `gradium-tts-male`, ...),
  each with its own `voice_id`. They appear as separate services in the Services tab.

## How it works

| File | Role |
|---|---|
| `src/examples/shared/tts_factory.py` | Reads `provider` and returns the matching Pipecat service. The NVIDIA branch is the block that previously lived inline in each pipeline. |
| `src/gradium_catalog.py` | Fetches `GET /voices/` and reshapes it into the `{languages, voices, defaultVoiceId}` payload the client's voice picker expects. |
| `src/utils.py` | Allows `tts_provider` through the session-config allowlist and catalog hydration. |
| `src/server.py` | Skips the Riva readiness probe for external providers and routes voice discovery to the Gradium catalog. |

Example pipelines call `build_tts_service(body, default_tts)` instead of constructing a TTS
service directly.

## Behaviour differences vs Magpie

- **Sample rate** is 48 kHz; Pipecat sets and resamples this automatically.
- **Languages**: Gradium covers en, fr, de, es, pt. Magpie Multilingual covers 12. The
  multilingual example's ASR∩TTS language intersection is bypassed for Gradium.
- **Word timestamps** are forwarded, so transcript sync keeps working.
- **Interruptions** drop the audio context rather than cancelling server-side, so you are billed
  for audio already generated at barge-in.
- **`TTS_IPA_FILE_PATH` does not apply** — the IPA dictionary is an `NvidiaTTSService` argument.
  Use a Gradium pronunciation dictionary and reference its id via `json_config`.
- **Zero-shot cloning** via `zero_shot_audio_prompt_file` is NVIDIA-specific. Use Gradium's own
  voice-cloning endpoint and reference the resulting voice UID as `voice_id`.

## Troubleshooting

| Symptom | Cause |
|---|---|
| Session refuses to start: `GRADIUM_API_KEY is not set` | Key exported in the shell but not in `.env`. |
| Voice picker empty | `GET /voices/` rejected the key, or egress to `api.gradium.ai:443` is blocked. Test with curl from inside the app container. |
| New Gradium voice missing | The voice list is cached for the app's lifetime. Restart the app container. |
| Session connects, no speech | Check Gradium credits — a 401/402 on `/voices/` is the quickest signal. |
| Log says `provider=nvidia` | The catalog entry isn't selected. Check `defaults.tts`, and clear the browser's `localStorage` — the UI persists service selections and overrides the server default. |

Confirm the active provider on any session start:

```bash
docker compose logs <app-service> | grep "TTS: provider"
# TTS: provider=gradium, url=wss://api.gradium.ai/api/speech/tts, voice=..., sample_rate=48000
```

## Self-hosting the local LLM alongside this

Freeing the TTS GPU makes a two-GPU deployment practical. See
[`docker/docker-compose.a100.yaml`](../../docker/docker-compose.a100.yaml) for a 2× A100 layout.
Two things that are easy to get wrong:

- **`NIM_KVCACHE_PERCENT` is a total-memory budget**, not a KV-cache reservation. It maps to
  vLLM's `gpu_memory_utilization`: weights + activations + KV cache combined. BF16 weights for
  Nemotron 3 Nano are ~60 GB of an 80 GB card, so the upstream `0.6` (tuned for ~30 GB FP8
  weights) leaves zero KV blocks and the engine aborts with *"No available memory for the cache
  blocks"*. Use `0.9`. Lowering it makes the failure worse, not safer.
- **A100 is `sm_80` and has no FP8 tensor cores.** The upstream default
  `NIM_TAGS_SELECTOR=precision=fp8,tp=1` has no A100 profile. Check what your NIM publishes with
  `docker run --rm --gpus all -e NGC_API_KEY=$NVIDIA_API_KEY <nim-image> list-model-profiles`
  and set `precision=bf16,tp=1`.
- **`defaults.llm` must name a key that exists in the catalog you are running.** The self-hosted
  LLM entry is keyed `nemotron-nano` in `services.local.yaml`, while the default is
  `nemotron-lightning`, which exists only in `services.cloud.yaml`. A default whose key is absent
  from the local catalog resolves to the **cloud** model even when your local NIM is healthy —
  the pipeline quietly calls `integrate.api.nvidia.com` while your GPU idles. Select
  "Nemotron 3 Nano" in the Services tab for on-prem runs, and confirm with
  `grep "LLM: model=" `: you want `base_url=http://nvidia-llm:8000/v1`.

## Reference

- [Gradium TTS WebSocket guide](https://docs.gradium.ai/guides/text-to-speech)
- [Gradium Pipecat integration](https://docs.gradium.ai/integrations/agent-frameworks/pipecat)
- [Gradium voice library](https://docs.gradium.ai/guides/voices/overview)
- [Configure TTS](configure-tts.md) · [Configure Services](configure-services.md)
