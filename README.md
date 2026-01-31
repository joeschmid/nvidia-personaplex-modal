# PersonaPlex on Modal Labs

Real-time voice-to-voice AI with customizable personas, deployed on [Modal Labs](https://modal.com).

Built on NVIDIA's [PersonaPlex](https://github.com/NVIDIA/personaplex) model - a full-duplex speech-to-speech conversational AI that enables persona control through text prompts and voice conditioning.

## Features

- 🎙️ **Real-time voice conversation** - Talk naturally with AI, no push-to-talk needed
- 🎭 **Customizable personas** - Define AI personality via text prompts
- 🗣️ **Multiple voices** - Choose from natural and variety voice presets
- ⚡ **Low latency** - Streaming audio for responsive conversations
- 💰 **Cost efficient** - Pay only for compute time, auto-scales to zero

## Quick Start

### Prerequisites

1. **Modal account** - Sign up at [modal.com](https://modal.com)
2. **HuggingFace account** - Sign up at [huggingface.co](https://huggingface.co)
3. **Accept PersonaPlex license** - Visit [nvidia/personaplex](https://huggingface.co/nvidia/personaplex) and accept the license

### Setup

```bash
# Install dependencies with uv
uv sync

# Authenticate with Modal
uv run modal token new

# Create HuggingFace secret (get token from huggingface.co/settings/tokens)
uv run modal secret create huggingface-secret HF_TOKEN=hf_your_token_here
```

### Run Locally (Development)

```bash
# Start the server (stops on Ctrl+C)
uv run modal serve personaplex.py
```

The server will print a URL like `https://your-username--personaplex-....modal.run`. Open this in your browser.

### Deploy (Production)

```bash
# Deploy to Modal (runs until you stop it)
uv run modal deploy personaplex.py
```

## Usage

1. Open the web UI in your browser
2. Enter the access password: `REDACTED_PASSWORD`
3. Select a voice preset
4. Optionally customize the persona prompt
5. Click "Start Talking" and allow microphone access
6. Speak naturally - the AI responds in real-time!

## Voice Presets

| Voice | Description |
|-------|-------------|
| NATF2 | Natural Female voice |
| NATM2 | Natural Male voice |
| VARF0 | Variety Female voice |
| VARM0 | Variety Male voice |

## Persona Prompts

Customize the AI's personality with text prompts. Examples:

**Helpful Assistant:**
```
You are a helpful and friendly assistant. Answer questions clearly and concisely.
```

**Customer Service:**
```
You are a customer service agent for TechCorp. Your name is Alex.
Help customers with product questions and troubleshooting.
```

**Casual Conversation:**
```
You enjoy having a good conversation. You're witty and engaging.
Talk about technology, science, or whatever interests the user.
```

## Project Structure

```
v2v/
├── personaplex.py      # Main Modal application
├── pyproject.toml      # Project config (uv)
├── static/
│   └── index.html      # Web UI
├── PLAN.md             # Implementation plan
└── README.md           # This file
```

## Architecture

```
┌─────────────┐     WebSocket      ┌──────────────────┐
│   Browser   │ ◄──────────────────► │  Modal Container │
│  (Web UI)   │   Opus Audio        │                  │
└─────────────┘                     │  ┌────────────┐  │
                                    │  │   Mimi     │  │
                                    │  │  (Codec)   │  │
                                    │  └─────┬──────┘  │
                                    │        │         │
                                    │  ┌─────▼──────┐  │
                                    │  │   Moshi    │  │
                                    │  │   (LLM)    │  │
                                    │  └────────────┘  │
                                    │                  │
                                    │  GPU: A10G      │
                                    └──────────────────┘
```

## Cost Management

Modal charges only for active compute time:

| Scenario | Estimated Cost |
|----------|----------------|
| 1 hour active use | ~$0.60 |
| 5 min test | ~$0.05 |
| Deployed but idle | $0.00 |

### Commands

```bash
# Check running apps
uv run modal app list

# View logs
uv run modal app logs personaplex

# Stop the app (removes endpoint)
uv run modal app stop personaplex

# Delete the app
uv run modal app delete personaplex
```

## Configuration

Edit `personaplex.py` to customize:

```python
# GPU type (A10G recommended for cost, A100 for performance)
GPU_TYPE = "A10G"

# Container idle timeout (seconds before auto-shutdown)
CONTAINER_IDLE_TIMEOUT = 300

# Access password
AUTH_PASSWORD = "REDACTED_PASSWORD"

# Available voices
VOICE_PRESETS = {
    "NATF2": "NATF2.pt",
    "NATM2": "NATM2.pt",
    "VARF0": "VARF0.pt",
    "VARM0": "VARM0.pt",
}
```

## Troubleshooting

### "Invalid password" error
- Check that you're using the correct password in the web UI

### Cold start takes too long
- First request after idle loads models (~30-60s)
- Subsequent requests are fast while container is warm
- Increase `CONTAINER_IDLE_TIMEOUT` to keep warm longer

### Out of memory (OOM) on A10G
- The code uses `cpu_offload=True` by default
- If still failing, switch to `A100` GPU in config

### No audio playback
- Ensure you're using Chrome or Firefox
- Check browser console for errors
- Verify microphone permissions are granted

### WebSocket connection fails
- Check that the Modal app is running
- Verify the URL matches the Modal deployment
- Check for HTTPS/WSS requirements

## API Reference

### WebSocket: `/ws/chat`

Connect with query parameters:

| Parameter | Type | Description |
|-----------|------|-------------|
| `password` | string | Access password |
| `voice_prompt` | string | Voice preset name (e.g., "NATF2") |
| `text_prompt` | string | Persona system prompt |
| `seed` | int | Random seed (-1 for random) |

### Message Protocol

| Byte 0 | Direction | Meaning |
|--------|-----------|---------|
| 0x00 | Server→Client | Handshake (ready to receive audio) |
| 0x01 | Both | Opus-encoded audio data |
| 0x02 | Server→Client | Text token (UTF-8) |

### HTTP Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Web UI (requires auth) |
| `/health` | GET | Health check |
| `/voices` | GET | List available voices |

## Credits

- [NVIDIA PersonaPlex](https://github.com/NVIDIA/personaplex) - Base model
- [Kyutai Moshi](https://github.com/kyutai-labs/moshi) - Original architecture
- [Modal Labs](https://modal.com) - Serverless GPU infrastructure

## License

This deployment code is provided as-is. PersonaPlex model weights are subject to [NVIDIA's license](https://huggingface.co/nvidia/personaplex).
