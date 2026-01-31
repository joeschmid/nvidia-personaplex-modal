# PersonaPlex on Modal Labs - Implementation Plan

## Project Overview

Deploy NVIDIA's PersonaPlex voice-to-voice AI model on Modal Labs, following the patterns established in the Parakeet multi-talker example but adapted for PersonaPlex's full-duplex conversational capabilities.

## Understanding the Models

### PersonaPlex
- **Type**: Real-time, full-duplex speech-to-speech conversational AI
- **Features**: Persona control via text prompts + voice conditioning
- **Architecture**: Built on Moshi framework with Mimi audio codec + Helium LLM backbone
- **I/O**: WebSocket streaming of Opus-encoded audio chunks
- **Voices**: 18 pre-packaged voice embeddings (NATF0-3, NATM0-3, VARF0-4, VARM0-4)

### Key Differences from Parakeet Example
| Aspect | Parakeet | PersonaPlex |
|--------|----------|-------------|
| Direction | Speech-to-text (one way) | Speech-to-speech (bidirectional) |
| Output | Text transcriptions | Audio responses |
| Streaming | Audio in, text out | Audio in, audio out |
| Persona | N/A | Text prompts + voice embeddings |

---

## Implementation Plan

### Phase 1: Project Structure Setup

Create the following file structure:
```
personaplex-modal/
├── personaplex.py          # Main Modal application
├── client/
│   └── index.html          # Web UI for testing
├── requirements.txt        # Python dependencies (for reference)
├── voices/                 # Voice embedding files (downloaded at build)
└── README.md               # Documentation
```

### Phase 2: Modal Infrastructure (`personaplex.py`)

#### 2.1 Base Image Definition
```python
image = (
    modal.Image.from_registry("nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04")
    .apt_install("libopus-dev", "pkg-config", "build-essential", "git")
    .pip_install(
        "numpy>=1.26,<2.2",
        "safetensors>=0.4.0,<0.5",
        "huggingface-hub>=0.24,<0.25",
        "einops==0.7",
        "sentencepiece==0.2",
        "sounddevice==0.5",
        "sphn>=0.1.4,<0.2",
        "torch>=2.2.0,<2.5",
        "aiohttp>=3.10.5,<3.11",
        "fastapi",
        "uvicorn",
    )
    .run_commands(
        "pip install git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi"
    )
)
```

#### 2.2 Persistent Volume for Model Cache
- Create a Modal Volume to cache HuggingFace model weights
- Prevents re-downloading on each cold start
- Expected models: Mimi codec, Moshi LM, voice embeddings

#### 2.3 PersonaPlex Service Class
```python
@app.cls(
    gpu="A100",  # or A10G depending on VRAM needs
    volumes={"/cache": volume},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    image=image,
    container_idle_timeout=300,
)
class PersonaPlex:
    @modal.enter()
    def load_models(self):
        # Load Mimi, Moshi, tokenizer
        # Cache to persistent volume

    @modal.asgi_app()
    def serve(self):
        # Return FastAPI app with WebSocket endpoint
```

### Phase 3: WebSocket API Implementation

#### 3.1 Endpoints
| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/` | GET | Serve web UI |
| `/ws/chat` | WebSocket | Bidirectional audio streaming |
| `/voices` | GET | List available voice presets |
| `/health` | GET | Health check |

#### 3.2 WebSocket Protocol
Following PersonaPlex's existing protocol:
1. **Handshake**: Initial `0x00` bytes
2. **Client → Server**: Opus-encoded audio chunks (message type 1)
3. **Server → Client**: Opus-encoded response audio (message type 1)
4. **Query params**: `voice_prompt`, `text_prompt`, `seed`

#### 3.3 Async Processing Loops
Adapt from PersonaPlex's server.py:
- `recv_loop`: Receive client audio
- `opus_loop`: Process audio through Mimi→Moshi→Mimi pipeline
- `send_loop`: Stream response audio back

### Phase 4: Web Client

Create a minimal HTML/JS client for testing:
- Microphone capture using Web Audio API
- Opus encoding via `opus-recorder` or similar
- WebSocket connection management
- Voice/persona selection UI
- Audio playback of responses

### Phase 5: Deployment & Configuration

#### 5.1 Modal Secrets
```bash
uv run modal secret create huggingface-secret HF_TOKEN=<your-token>
```

#### 5.2 Running the App
```bash
# For development/testing (recommended - stops on Ctrl+C)
uv run modal serve personaplex.py

# For persistent deployment (only when ready for production)
uv run modal deploy personaplex.py
```

#### 5.3 GPU Selection
- **Selected**: A10G-24GB with `--cpu-offload` if needed
- Will test without offload first, enable if OOM occurs

---

## Required API Keys & Tokens

### 1. HuggingFace Token (Required)
- **Purpose**: Download the gated PersonaPlex model weights
- **How to get**:
  1. Create account at https://huggingface.co
  2. Go to https://huggingface.co/settings/tokens
  3. Create a new token with "Read" permissions
  4. Accept the PersonaPlex model license at https://huggingface.co/nvidia/personaplex
- **Setup in Modal**:
  ```bash
  modal secret create huggingface-secret HF_TOKEN=hf_xxxxxxxxxxxx
  ```

### 2. Modal Account (Required)
- **Purpose**: Deploy and run the application
- **How to get**:
  1. Sign up at https://modal.com
  2. Install dependencies: `uv sync`
  3. Authenticate: `uv run modal token new`
- **Cost**: Pay-as-you-go; A10G ~$0.60/hr when running

### 3. NVIDIA NGC Token (Optional)
- **Purpose**: Only needed if using NVIDIA's container registry directly
- **When needed**: If we switch to `nvcr.io` base images that require auth
- **How to get**: https://ngc.nvidia.com/setup/api-key
- **Currently**: Not required - using public CUDA image

### Summary Table

| Token | Required | Where to Get | Modal Secret Name |
|-------|----------|--------------|-------------------|
| HuggingFace | ✅ Yes | huggingface.co/settings/tokens | `huggingface-secret` |
| Modal | ✅ Yes | modal.com (CLI auth) | N/A (automatic) |
| NVIDIA NGC | ❌ No | ngc.nvidia.com | `ngc-secret` (if needed) |

---

## Modal Resource Lifecycle & Cost Management

### How Modal Billing Works
- **Pay only for compute time**: Billed per second when containers are running
- **No charge when idle**: Containers scale to zero automatically
- **A10G cost**: ~$0.000164/sec (~$0.59/hr) - only while processing requests

### Automatic Shutdown Behavior

Modal containers **automatically shut down** after a period of inactivity:

| Setting | Default | Our Config | Effect |
|---------|---------|------------|--------|
| `container_idle_timeout` | 60s | 300s (5 min) | Container stays warm for 5 min after last request |
| Scale to zero | Always | Always | No running containers = no cost |

**What this means for testing:**
1. First request after idle = **cold start** (~30-60s to load models)
2. Subsequent requests within 5 min = **warm** (fast, ~100ms)
3. After 5 min of no requests = **auto-shutdown** (no cost)

### Commands for Manual Control

```bash
# Deploy the app (creates the endpoint, doesn't start containers)
uv run modal deploy personaplex.py

# Check what's running
uv run modal app list

# View logs
uv run modal app logs personaplex

# Stop the app entirely (removes endpoint)
uv run modal app stop personaplex

# Delete the app
uv run modal app delete personaplex
```

### Cost-Efficient Testing Strategy

1. **Development**: Use `uv run modal serve personaplex.py` (hot-reload, stops when you Ctrl+C)
2. **Testing**: Use `uv run modal deploy` with short `container_idle_timeout` (e.g., 60s)
3. **Demo**: Increase timeout to 300s to avoid cold starts during demo
4. **Not using it?**: Run `uv run modal app stop personaplex` to remove the endpoint

### Estimated Costs for Testing

| Scenario | Duration | Cost |
|----------|----------|------|
| 1 hour of active testing | 60 min | ~$0.60 |
| Quick 5-min test | 5 min | ~$0.05 |
| Deployed but idle | Any | $0.00 |
| Cold start (model load) | ~45 sec | ~$0.01 |

### No Surprises

- ✅ Containers auto-terminate after idle timeout
- ✅ No background processes running when not in use
- ✅ Persistent volume storage is cheap (~$0.07/GB/month for model cache)
- ✅ You can set spending limits in Modal dashboard
- ⚠️ Only risk: leaving a WebSocket connection open keeps container alive

---

## Technical Considerations

### 1. Latency Optimization
- Keep container warm with `container_idle_timeout`
- Use persistent volume for model caching
- Consider `keep_warm=1` for production

### 2. Audio Format Handling
- Input: Browser sends Opus-encoded audio (48kHz)
- Processing: Decode to PCM → Mimi codec → Moshi LM → back to PCM
- Output: Encode to Opus for browser playback

### 3. Concurrency
- Each WebSocket connection = one conversation session
- Model state must be managed per-session
- Consider `allow_concurrent_inputs` for multiple users

### 4. Authentication
- HuggingFace token required (model gated)
- Accept PersonaPlex license at: https://huggingface.co/nvidia/personaplex

---

## Estimated Timeline

| Phase | Task | Estimate |
|-------|------|----------|
| 1 | Project setup | 15 min |
| 2 | Modal infrastructure | 1-2 hours |
| 3 | WebSocket API | 2-3 hours |
| 4 | Web client | 1-2 hours |
| 5 | Testing & deployment | 1-2 hours |

**Total**: ~6-10 hours

---

## Decisions Made

1. **GPU Tier**: A10G (24GB VRAM) - may need `--cpu-offload` flag
2. **Persistence**: Not required for MVP
3. **Authentication**: Simple password protection ("REDACTED_PASSWORD")
4. **Voice Presets**: Subset of voices (e.g., NATF2, NATM2, VARF0, VARM0)
5. **Client**: Minimal HTML/JS client for MVP

---

## Next Steps (After Plan Approval)

1. Create the Modal app skeleton with image definition
2. Implement model loading and caching
3. Port the WebSocket server logic
4. Build/adapt the web client
5. Test end-to-end
6. Deploy and document

---

*Plan created: January 2025*
