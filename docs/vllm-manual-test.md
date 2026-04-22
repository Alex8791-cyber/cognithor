# vLLM Backend — Manual Smoke Test

The Python + Flutter unit tests and the fake-server integration test cover
everything except the real-hardware path. Before cutting a release that
touches the vLLM backend, run this recipe once on a dev machine with a real
NVIDIA GPU and Docker Desktop.

**Time:** ~30 minutes end-to-end.

## Test Matrix

Pick the row matching your dev hardware and run those steps:

| GPU | VRAM | Expected recommended model | Test sections |
|-----|------|----------------------------|---------------|
| RTX 5090 | 32 GB | `mmangkad/Qwen3.6-27B-NVFP4` | All |
| RTX 4090 | 24 GB | `cyankiwi/Qwen3.6-27B-AWQ-INT4` or fallback | 1–5 |
| RTX 4080 / 4070 Ti Super | 16 GB | `Qwen/Qwen2.5-VL-7B-Instruct` (tested fallback) | 1–4 |

## 1. Fresh install

- Uninstall any previous Cognithor.
- Wipe `~/.cognithor/config.yaml` (or rename it as backup).
- Run the new installer.
- Launch Cognithor. Expect: Flutter UI reaches the main screen. No version-
  mismatch overlay, no config crash.

## 2. Hardware + Docker detection

- Settings → LLM Backends → tap **vLLM**.
- Expect: Cards 1 and 2 turn green within 2 seconds (the polling interval).
  Card 1 shows the correct GPU name, VRAM, and compute capability string.
- If Docker Desktop is not running, start it and wait — Card 2 turns green
  when ready.

## 3. Image pull

- Tap **Pull image**.
- Expect: progress bar advances steadily. Total download ~10 GB.
- Expect: after completion, Card 3 turns green; Card 4 enables.

## 4. Model picker + start (quick path — Qwen2.5-VL-7B)

- Card 4: expect the model dropdown to populate. Qwen2.5-VL-7B is always
  marked "tested" and should show the star badge for any non-Blackwell GPU.
- Select it. Tap **Start vLLM**.
- Expect: within 120 seconds, Card 4 turns green with "Running: Qwen/
  Qwen2.5-VL-7B-Instruct".
- Back in the list view, tap vLLM, tap **Make active**.

## 5. End-to-end chat with vision

- Open the chat screen.
- Attach an image (any PNG) via the paperclip button.
- Ask: "What do you see in this image?"
- Expect: answer comes back within 5–10 seconds, describing the image.
- Close the chat screen and reopen — state persists.

## 6. Fail-flow verification

- In a terminal: `docker stop $(docker ps -q --filter label=cognithor.managed=true)`
- In Cognithor chat, send a text-only message: "hello"
  - Expect: within 2–3 requests, the "⚠ vLLM offline — fallback to Ollama
    active" banner appears. Reply comes from Ollama.
- Send an image request.
  - Expect: red error bubble "vLLM offline — cannot process image".
- In terminal: `docker start <container-id>`
  - Expect: within ~60 seconds (the CircuitBreaker `recovery_timeout`), the
    next text request goes through vLLM again; banner dismisses.

## 7. Lifecycle toggles

- Settings → LLM Backends → vLLM → enable "Keep vLLM running after app close"
- Close Cognithor entirely.
- Verify: `docker ps | grep cognithor.managed` — container still running.
- Reopen Cognithor → status shows "Running" immediately (no restart).
- Disable the toggle. Close Cognithor.
- Verify: `docker ps | grep cognithor.managed` — no result (container was
  stopped on shutdown).

## 8. Blackwell-specific (RTX 5090 only)

- Edit `~/.cognithor/config.yaml` → set `vllm.docker_image: "vllm/vllm-openai:nightly"`.
- Back in the setup screen, pull the nightly image (replaces the old one).
- Pick `mmangkad/Qwen3.6-27B-NVFP4` (star-badged on Blackwell).
- Start. Expect: model loads in ~30–60 seconds.
- Chat with vision. Expect: tokens stream noticeably faster than FP8 on the
  same hardware (NVFP4 uses native tensor cores).

## Reporting

If any step fails, capture:
- Contents of `~/.cognithor/log/cognithor.log` (last 200 lines)
- Output of `docker logs <container-id>`
- Screenshot of the relevant Flutter screen

File bugs against the `vllm-backend` label on GitHub.
