# Enabling the vLLM Backend

Cognithor ships with Ollama as the default local LLM backend — it "just works"
on any Windows/macOS/Linux machine without further setup. vLLM is an **opt-in**
alternative for users with an NVIDIA GPU who want faster inference, native FP4
support (Blackwell / RTX 50xx), or access to models not in the Ollama library.

## Prerequisites

You need:

1. **NVIDIA GPU** with at least 16 GB VRAM. For the best experience:
   - **RTX 5090 (32 GB)** — unlocks NVFP4 quantization, the fastest option
   - **RTX 4090 (24 GB)** — runs FP8 quantization
   - **RTX 4080 / 3090 / 4070 Ti Super (16 GB)** — runs AWQ-INT4 quantization
2. **NVIDIA driver** installed (any modern version from the last 2 years works)
3. **Docker Desktop** installed and running. Download from
   [docker.com](https://www.docker.com/products/docker-desktop). Cognithor will
   not install it for you — the installer needs admin rights and a reboot,
   which Cognithor does not handle.

## Enabling vLLM

1. Start Cognithor normally.
2. Open **Settings → LLM Backends**.
3. Tap **vLLM**. You will see four status cards:
   - **NVIDIA GPU** — detected automatically
   - **Docker Desktop** — detected automatically
   - **vLLM Docker image** — needs a one-time ~10 GB pull
   - **Model** — which model to load
4. If any card is red, fix the underlying issue first (install the missing
   driver, start Docker Desktop, etc.).
5. Tap **Pull image** on Card 3. Progress streams live.
6. Pick a model from Card 4. The star badge (⭐) marks the recommendation for
   your GPU. Models that don't fit your VRAM or require a newer GPU
   architecture are disabled with a tooltip explaining why.
7. Tap **Start vLLM**. First start takes 30–300 seconds depending on model
   size (weights download from HuggingFace).
8. Back in the list view, tap your vLLM row and select **Make active** to
   switch all future chat turns through vLLM.

## Switching Back to Ollama

Settings → LLM Backends → Ollama → **Make active**. The switch is live — no
restart required. vLLM keeps running in the background unless you enable
"Stop vLLM on app close" in settings.

## Troubleshooting

**"Version Mismatch" overlay on launch**: the installer bundled a stale
Flutter build. Install the newer Cognithor release.

**vLLM status card stays red with "No GPU detected"**: run `nvidia-smi` in a
terminal to confirm your driver works. On WSL2 you need the NVIDIA WSL driver
bundle from nvidia.com/drivers — not just the standard Windows driver.

**Docker card stays red**: open Docker Desktop, wait for the whale icon to
stop pulsing (that's the "ready" state).

**Pull fails mid-download**: partial layers are cached. Retry the pull —
Docker will resume from where it stopped.

**Container starts but /health never answers**: the model is probably still
loading. Qwen3.6-27B at FP8 takes ~60 s on an RTX 5090, up to 5 minutes on
slower cards. The setup page shows container logs below the status cards.

**Banner "vLLM offline — fallback to Ollama active"** appears mid-chat: vLLM
has crashed or become unresponsive for 3 consecutive requests. Text chats
transparently route through Ollama; image requests will error out until vLLM
recovers. Check `docker logs <container-id>` for the cause.

**I have a Qwen3.6 model selected but it fails to start**: vLLM stable
(v0.19.1) does not yet support the Qwen3.6 architecture. Workaround: set
`config.vllm.docker_image` to `vllm/vllm-openai:nightly` and restart. Cognithor
will adopt the new image on the next container start.

## Advanced Configuration

`~/.cognithor/config.yaml` section `vllm`:

| Field | Default | Purpose |
|-------|---------|---------|
| `enabled` | `false` | Master on/off |
| `model` | `""` (auto) | HF repo id. Empty → orchestrator picks best per GPU |
| `docker_image` | `vllm/vllm-openai:v0.19.1` | Override to bleed-edge |
| `port` | `8000` | Host port (falls back 8001..8009 if busy) |
| `auto_stop_on_close` | `false` | Stop container when Cognithor quits |
| `skip_hardware_check` | `false` | Override for unusual setups |
| `request_timeout_seconds` | `60` | Per-request timeout |

HF token for gated models: set `huggingface_api_key` at the top level of
`config.yaml` (or via the OS keyring) — Cognithor passes it to the container
automatically as `HF_TOKEN`.
