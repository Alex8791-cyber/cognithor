# Frequently Asked Questions

## Installation

### 1. What are the minimum system requirements?

- **CPU**: 8 cores (recommended: Ryzen 9 or equivalent)
- **RAM**: 16 GB minimum, 64 GB recommended
- **GPU**: 8 GB VRAM minimum for local LLMs. Recommended: RTX 3090+ (24 GB) or RTX 5090 (32 GB)
- **Disk**: 50 GB free (100 GB+ recommended for model storage)
- **Python**: 3.12 or newer
- **OS**: Windows 10/11, Ubuntu 20.04+, macOS 12+

### 2. Can I run Cognithor without a GPU?

Yes, but performance will be significantly slower. You can use smaller models (e.g., `qwen3:8b` as planner) or switch to a cloud LLM backend (OpenAI, Anthropic, etc.) which requires no local GPU at all.

### 3. How do I install Cognithor?

```bash
git clone <repo-url> cognithor
cd cognithor
./install.sh          # Interactive installer (Linux/macOS)
# or manually:
python -m venv .venv
source .venv/bin/activate   # Linux/macOS
.venv\Scripts\activate      # Windows
pip install -e ".[all,dev]"
```

On Windows, double-click `start_cognithor.bat` after installation.

### 4. The installation fails with "build wheel failed". What do I do?

Some optional dependencies require native compilation. Install only the extras you need:
```bash
pip install -e ".[search,documents]"    # Instead of [all]
```
For SQLCipher encryption specifically: `pip install cognithor[encryption]` requires native SQLCipher libraries installed on your system.

### 5. How do I update to a newer version?

```bash
git pull
pip install -e ".[all,dev]"
python scripts/first_boot.py --quick    # Verify everything works
```

---

## LLM Backends

### 6. Which LLM backend should I use?

- **Ollama** (default): Best for privacy, runs completely local. Needs a GPU with 24+ GB VRAM for the recommended models.
- **LM Studio**: Same as Ollama but with a GUI for model management. Also fully local.
- **OpenAI / Anthropic / Gemini**: Best quality, no GPU needed, but requires internet and API costs.
- **Groq / Cerebras**: Fast cloud inference, free tiers available.
- **DeepSeek / Mistral / Together / OpenRouter**: Budget-friendly alternatives with various model options.

### 7. How do I switch from Ollama to a cloud backend?

Set your API key in `~/.cognithor/config.yaml` or as an environment variable. Cognithor auto-detects the backend:

```yaml
# Option A: In config.yaml
anthropic_api_key: "your-anthropic-api-key"

# Option B: Environment variable
# export COGNITHOR_ANTHROPIC_API_KEY=your-anthropic-api-key
```

Model names are automatically adapted to the new provider.

### 8. Which Ollama models should I download?

```bash
ollama pull qwen3:32b            # Planner (required, 20 GB VRAM)
ollama pull qwen3:8b             # Executor (required, 6 GB VRAM)
ollama pull qwen3-embedding:0.6b # Embeddings (required, 0.5 GB VRAM)
ollama pull qwen3-coder:30b      # Coder (optional, 20 GB VRAM)
```

### 9. Can I use different models than the defaults?

Yes, change them in `~/.cognithor/config.yaml`:

```yaml
models:
  planner:
    name: "llama3.3:70b"
    context_window: 128000
  executor:
    name: "llama3.1:8b"
```

### 10. Ollama is not responding. How do I fix it?

```bash
# Check if Ollama is running
curl http://localhost:11434/api/tags

# Restart Ollama
# Linux/macOS:
pkill ollama && ollama serve &
# Windows: Restart the Ollama app from system tray
```

If using a remote Ollama instance, set the URL: `export OLLAMA_HOST=http://remote-host:11434`

---

## Configuration

### 11. Where is the configuration file?

`~/.cognithor/config.yaml`. It is auto-created on first start. See `CONFIG_REFERENCE.md` for all available options.

### 12. How do I set the language?

```yaml
language: "en"   # en, de, zh, ar
```
Or via environment variable: `export COGNITHOR_LANGUAGE=en`

### 13. How do I restrict file access?

```yaml
security:
  allowed_paths:
    - "~/.cognithor/"
    - "~/Documents/"
    - "/path/to/project/"
```

The gatekeeper blocks file operations outside these paths.

### 14. Can I use environment variables instead of config.yaml?

Yes. All config keys can be overridden with `COGNITHOR_` prefixed variables:
```bash
export COGNITHOR_LANGUAGE=en
export COGNITHOR_PLANNER_MAX_ITERATIONS=15
export COGNITHOR_OLLAMA_TIMEOUT_SECONDS=180
```

---

## Channels

### 15. How do I set up the Telegram bot?

1. Create a bot with [@BotFather](https://t.me/BotFather) on Telegram
2. Set the token in `~/.cognithor/.env`: `COGNITHOR_TELEGRAM_TOKEN=123456:ABC...`
3. Enable in config: `channels.telegram_enabled: true`
4. Optionally restrict to your user ID: `channels.telegram_whitelist: ["your_user_id"]`

### 16. How do I access the Web UI?

The Flutter Web UI is automatically served by the backend on port 8741. Start Cognithor and open `http://localhost:8741` in your browser. On Windows, `start_cognithor.bat` does this automatically.

### 17. Which messaging channels are supported?

CLI, WebUI, Telegram, Discord, Slack, WhatsApp, Signal, Matrix, Microsoft Teams, Google Chat, Mattermost, Feishu/Lark, IRC, Twitch, iMessage, and Voice.

### 18. Can I use multiple channels simultaneously?

Yes. Enable each channel in the config and provide the required tokens/credentials. All channels share the same agent brain and memory.

---

## Security

### 19. Is my data sent to external servers?

By default, no. Cognithor uses Ollama (local LLM) and stores all data in `~/.cognithor/`. If you configure a cloud LLM backend (OpenAI, Anthropic, etc.), your prompts are sent to that provider's API. Web search tools access the internet when invoked.

### 20. How does the Gatekeeper work?

Every tool call goes through a risk assessment:
- **GREEN**: Auto-executed (read memory, search, calculations)
- **YELLOW**: Executed + user informed (create file, set reminder)
- **ORANGE**: Requires user confirmation (send email, delete file)
- **RED**: Blocked (destructive system commands, credential exposure)

### 21. Can I encrypt the databases?

Yes. Enable SQLCipher encryption:
```yaml
database:
  encryption_enabled: true
```
Requires `pip install cognithor[encryption]`. The encryption key is stored in your OS keyring.

### 22. How does API authentication work?

The backend generates a per-session token at startup. The Web UI fetches it via the `/api/v1/bootstrap` endpoint. All subsequent API calls require `Authorization: Bearer <token>`. WebSocket connections authenticate with an `{"type": "auth", "token": "..."}` message.

---

## Flutter UI

### 23. How do I build the Flutter UI from source?

```bash
cd flutter_app
flutter pub get
flutter run -d chrome        # Development
flutter build web --release  # Production build
```

The production build goes to `flutter_app/build/web/` and is automatically served by the backend.

### 24. The UI shows "Connection lost". What is wrong?

The backend is not running or not reachable. Check:
1. Is the backend running on port 8741? (`curl http://localhost:8741/api/v1/health`)
2. Is a firewall blocking the connection?
3. Check logs at `~/.cognithor/logs/cognithor.log`

---

## Performance

### 25. The first response is very slow. Is that normal?

Yes. On the first request, Ollama loads the model into VRAM (30--60 seconds). Subsequent responses are much faster (1--5 seconds). You can preload models:
```bash
ollama run qwen3:32b "Hello" --keepalive 30m
```

### 26. How can I reduce VRAM usage?

- Use smaller models: `qwen3:8b` as planner (instead of `qwen3:32b`)
- Reduce context window: `models.planner.context_window: 8192`
- Use a cloud backend (no local VRAM needed)

### 27. How do I monitor system health?

```bash
python scripts/first_boot.py --quick    # System validation
make smoke                              # 26 installation checks
make health                             # Runtime check (Ollama, disk, memory)
tail -f ~/.cognithor/logs/cognithor.log       # Live logs
```

---

## Memory & Knowledge

### 28. How does the memory system work?

Cognithor has three memory layers:
1. **Working Memory**: Current conversation context (auto-compacted when full)
2. **Semantic Memory**: Long-term knowledge indexed with vector embeddings + BM25 + knowledge graph
3. **Episodic Memory**: Daily interaction logs with auto-generated summaries

### 29. How do I import existing knowledge?

Place documents (Markdown, PDF, DOCX, TXT) in `~/.cognithor/memory/knowledge/`. They are automatically indexed on next startup.

### 30. How do I back up my data?

```bash
# All data lives in ~/.cognithor/
cp -r ~/.cognithor/ ~/backup/cognithor-backup/
```

See `DATABASE.md` for a complete list of database files and their locations.

---

## Troubleshooting

### 31. "No module named 'cognithor'" error

Make sure the virtual environment is activated and the package is installed:
```bash
source .venv/bin/activate    # or .venv\Scripts\activate on Windows
pip install -e ".[all]"
```

### 32. "database is locked" errors

This happens under heavy concurrent access. Cognithor automatically retries with exponential backoff. If persistent, check that no other process has an exclusive lock. Increase retries:
```yaml
database:
  sqlite_max_retries: 10
  sqlite_retry_base_delay: 0.2
```

### 33. The planner always answers directly instead of using tools

Check:
1. Are MCP tools registered? Check `~/.cognithor/logs/cognithor.log` for tool registration messages.
2. Is the system prompt too long? Reduce `memory.search_top_k`.
3. Try increasing temperature: `planner.temperature: 0.7`
4. Run `python scripts/first_boot.py` to validate the full agent loop.

### 34. Web search returns no results

DuckDuckGo may rate-limit. Options:
- Wait and retry (automatic 30s backoff)
- Configure a Brave or SearXNG backend (see `web` config section)
- Check if `web.duckduckgo_enabled` is `true`

### 35. How do I reset everything and start fresh?

```bash
# Remove all Cognithor data (CAUTION: this deletes all memories!)
rm -rf ~/.cognithor/

# Reinitialize
python scripts/first_boot.py
```

---

## Agent Quality

### 36. What is the Observer?

The Observer Audit Layer (PR #118) is an LLM-based quality check that runs after every response. It evaluates the response across 4 dimensions: hallucinations, sycophancy, laziness, and tool-ignorance. A hallucination finding triggers response regeneration inside the Planner; a tool-ignorance finding triggers a full PGE re-loop via the Gateway. The Observer always fails open — if the audit itself errors, the response is delivered unchanged. Configure it via the `observer.*` section in `config.yaml`; see `CONFIG_REFERENCE.md` for all options.
