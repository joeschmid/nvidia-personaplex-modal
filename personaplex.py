# PersonaPlex on Modal Labs
# Voice-to-voice AI with persona control
#
# Based on NVIDIA's PersonaPlex: https://github.com/NVIDIA/personaplex
# Deployed on Modal Labs: https://modal.com

import modal
import os

# ------------------------------------------------------------------------------
# Configuration
# ------------------------------------------------------------------------------

APP_NAME = "personaplex"
GPU_TYPE = "L40S"  # L40S-48GB
SCALEDOWN_WINDOW = 300  # 5 minutes
MAX_CONTAINERS = 1  # Prevent autoscaling beyond a single container
MODEL_CACHE_PATH = "/cache"
VOICE_PROMPT_DIR = "/cache/voices"

# Simple password auth (set via Modal secret or local env var)
def _get_auth_password() -> str:
    password = os.environ.get("AUTH_PASSWORD")
    if not password:
        raise RuntimeError(
            "AUTH_PASSWORD env var not set. Create Modal secret "
            "'personaplex-auth' with AUTH_PASSWORD or export it locally."
        )
    return password

# Available voice presets (subset for MVP)
VOICE_PRESETS = {
    "NATF2": "NATF2.pt",  # Natural Female 2
    "NATM2": "NATM2.pt",  # Natural Male 2
    "VARF0": "VARF0.pt",  # Variety Female 0
    "VARM0": "VARM0.pt",  # Variety Male 0
}

DEFAULT_VOICE = "NATF2"
DEFAULT_TEXT_PROMPT = "You are a helpful assistant. Answer questions clearly and concisely."

# ------------------------------------------------------------------------------
# Modal Image Definition
# ------------------------------------------------------------------------------

image = (
    # modal.Image.from_registry(
    #     "nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04",
    #     add_python="3.11",
    # )
    modal.Image.from_registry(
        "nvidia/cuda:13.1.1-cudnn-devel-ubuntu22.04", add_python="3.12"
    )
    .apt_install(
        "libopus-dev",
        "pkg-config",
        "build-essential",
        "git",
        "libsndfile1",
    )
    .pip_install(
        # PersonaPlex dependencies (from pyproject.toml)
        "numpy>=1.26,<2.2",
        "safetensors>=0.4.0,<0.5",
        "huggingface-hub>=0.24,<0.28",
        "einops==0.7",
        "sentencepiece==0.2",
        "sphn>=0.1.4,<0.2",
        "torch>=2.2.0,<2.6",
        "aiohttp>=3.10.5,<3.12",
        # Additional for Modal deployment
        "fastapi[standard]",
        "uvicorn",
        "python-multipart",
        # CPU offload support
        "accelerate",
    )
    .run_commands(
        # Install PersonaPlex from GitHub
        "pip install 'git+https://github.com/NVIDIA/personaplex.git#subdirectory=moshi'"
    )
    .env({
        "HF_HOME": MODEL_CACHE_PATH,
        "TORCH_HOME": MODEL_CACHE_PATH,
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
    .add_local_dir("static", "/app/static")
)

# ------------------------------------------------------------------------------
# Modal App Setup
# ------------------------------------------------------------------------------

app = modal.App(APP_NAME)
volume = modal.Volume.from_name("personaplex-cache", create_if_missing=True)


# ------------------------------------------------------------------------------
# PersonaPlex Service Class
# ------------------------------------------------------------------------------

@app.cls(
    gpu=GPU_TYPE,
    volumes={MODEL_CACHE_PATH: volume},
    secrets=[
        modal.Secret.from_name("huggingface-secret"),
        modal.Secret.from_name("personaplex-auth"),
    ],
    image=image,
    scaledown_window=SCALEDOWN_WINDOW,
    max_containers=MAX_CONTAINERS,
    timeout=600,  # 10 min max request time
    enable_memory_snapshot=True,  # Enable memory snapshots for faster cold starts
)
class PersonaPlexService:
    """
    Modal service class for PersonaPlex voice-to-voice AI.

    Handles model loading, WebSocket connections, and audio streaming.
    """

    @modal.enter(snap=True)
    def load_models_cpu(self):
        """
        Load models to CPU - this state gets snapshotted.

        This method runs BEFORE the snapshot is captured. By loading models
        to CPU here, we avoid re-downloading and re-loading on every cold start.
        The snapshot captures the CPU memory state with all models loaded.
        """
        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        import tarfile
        from pathlib import Path

        print("=" * 60)
        print("SNAPSHOT PHASE: Loading models to CPU for snapshotting...")
        print("=" * 60)

        self.auth_password = _get_auth_password()

        # Use CPU for initial loading (will transfer to GPU after snapshot restore)
        self.cpu_device = torch.device("cpu")
        print(f"Loading models to: {self.cpu_device}")

        # HuggingFace repo for PersonaPlex
        hf_repo = loaders.DEFAULT_REPO

        # Download and load Mimi audio codec to CPU
        print("Loading Mimi audio codec to CPU...")
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, self.cpu_device)
        self.other_mimi = loaders.get_mimi(mimi_weight, self.cpu_device)
        print("Mimi loaded to CPU")

        # Download and load text tokenizer (CPU-only, no change needed)
        print("Loading text tokenizer...")
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        print("Tokenizer loaded")

        # Download and load Moshi language model to CPU
        print("Loading Moshi language model to CPU (this may take a while)...")
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
        # Load to CPU without offload (we'll move to GPU later)
        self.lm = loaders.get_moshi_lm(moshi_weight, device=self.cpu_device, cpu_offload=False)
        self.lm.eval()
        print("Moshi loaded to CPU")

        # Download and extract voice prompts
        print("Loading voice prompts...")
        voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
        voices_tgz = Path(voices_tgz)
        self.voice_prompt_dir = voices_tgz.parent / "voices"

        if not self.voice_prompt_dir.exists():
            print(f"Extracting voice prompts to {self.voice_prompt_dir}")
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_tgz.parent)
        print(f"Voice prompts ready at {self.voice_prompt_dir}")

        # Commit volume to persist cached model weights
        volume.commit()

        print("=" * 60)
        print("CPU snapshot phase complete - snapshot will be captured now")
        print("=" * 60)

    @modal.enter(snap=False)
    def transfer_to_gpu(self):
        """
        Transfer models to GPU - runs AFTER snapshot restore.

        This method runs every time a container starts (from snapshot).
        It transfers the pre-loaded models from CPU to GPU and initializes
        CUDA-dependent components. This is much faster than loading from disk.
        """
        import torch
        import asyncio

        print("=" * 60)
        print("POST-RESTORE: Transferring models to GPU...")
        print("=" * 60)

        # Set up GPU device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Target device: {self.device}")

        # Transfer models to GPU
        print("Transferring Mimi to GPU...")
        self.mimi = self.mimi.to(self.device)
        self.other_mimi = self.other_mimi.to(self.device)
        print("Mimi transferred to GPU")

        print("Transferring Moshi LM to GPU...")
        self.lm = self.lm.to(self.device)
        print("Moshi LM transferred to GPU")

        # Create LMGen (must happen after GPU transfer)
        from moshi.models import LMGen

        self.frame_size = int(self.mimi.sample_rate / self.mimi.frame_rate)
        self.lm_gen = LMGen(
            self.lm,
            audio_silence_frame_cnt=int(0.5 * self.mimi.frame_rate),
            sample_rate=self.mimi.sample_rate,
            device=self.device,
            frame_rate=self.mimi.frame_rate,
            save_voice_prompt_embeddings=False,
        )

        # Initialize streaming mode
        self.mimi.streaming_forever(1)
        self.other_mimi.streaming_forever(1)
        self.lm_gen.streaming_forever(1)

        # Create async lock for thread safety
        self.lock = asyncio.Lock()

        # Warmup to initialize CUDA kernels
        print("Warming up CUDA kernels...")
        self._warmup()

        print("=" * 60)
        print("PersonaPlex ready! (restored from snapshot + GPU transfer)")
        print("=" * 60)

    def _warmup(self):
        """Run warmup inference to initialize CUDA kernels."""
        import torch

        for _ in range(4):
            chunk = torch.zeros(1, 1, self.frame_size, dtype=torch.float32, device=self.device)
            codes = self.mimi.encode(chunk)
            _ = self.other_mimi.encode(chunk)
            for c in range(codes.shape[-1]):
                tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                if tokens is None:
                    continue
                _ = self.mimi.decode(tokens[:, 1:9])
                _ = self.other_mimi.decode(tokens[:, 1:9])

        if self.device.type == 'cuda':
            import torch
            torch.cuda.synchronize()

    @modal.asgi_app()
    def serve(self):
        """Create and return the FastAPI application."""
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Query, Cookie, Response, Form
        from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
        import secrets
        import hashlib
        import hmac
        import time
        from pathlib import Path

        fastapi_app = FastAPI(title="PersonaPlex Voice AI")
        auth_password = getattr(self, "auth_password", None) or os.environ.get("AUTH_PASSWORD")
        if not auth_password:
            raise RuntimeError(
                "AUTH_PASSWORD env var not set. Create Modal secret "
                "'personaplex-auth' with AUTH_PASSWORD or export it locally."
            )

        # Secret key for signing session tokens (in production, use a proper secret management)
        session_secret = auth_password + "_session_secret_key"

        def create_session_token() -> str:
            """Create a signed session token."""
            timestamp = str(int(time.time()))
            # Create HMAC signature of timestamp
            signature = hmac.new(
                session_secret.encode(),
                timestamp.encode(),
                hashlib.sha256
            ).hexdigest()
            return f"{timestamp}:{signature}"

        def verify_session_token(token: str) -> bool:
            """Verify a session token is valid and not expired."""
            if not token or ":" not in token:
                return False
            try:
                timestamp, signature = token.split(":", 1)
                # Verify signature
                expected_signature = hmac.new(
                    session_secret.encode(),
                    timestamp.encode(),
                    hashlib.sha256
                ).hexdigest()
                if not secrets.compare_digest(signature, expected_signature):
                    return False
                # Optional: Check if token is not too old (e.g., 24 hours)
                token_age = int(time.time()) - int(timestamp)
                max_age = 24 * 60 * 60  # 24 hours
                return token_age < max_age
            except (ValueError, TypeError):
                return False

        def check_auth(auth_cookie: str | None) -> bool:
            """Check if auth cookie contains a valid session token."""
            if auth_cookie is None:
                return False
            return verify_session_token(auth_cookie)

        @fastapi_app.get("/")
        async def root(auth: str | None = Cookie(default=None)):
            """Serve the web UI or login page."""
            if not check_auth(auth):
                return HTMLResponse(content=self._get_login_page())
            # Static files are copied to /app/static in the container
            index_path = Path("/app/static/index.html")
            if index_path.exists():
                return HTMLResponse(content=index_path.read_text())
            return HTMLResponse(content=self._get_default_ui())

        @fastapi_app.post("/login")
        async def login(password: str = Form(...)):
            """Handle login - set auth cookie with session token if password is correct."""
            if secrets.compare_digest(password, auth_password):
                response = RedirectResponse(url="/", status_code=303)
                session_token = create_session_token()
                response.set_cookie(
                    key="auth",
                    value=session_token,
                    httponly=True,
                    secure=True,
                    samesite="strict",
                    max_age=24 * 60 * 60  # 24 hours
                )
                return response
            return HTMLResponse(content=self._get_login_page(error="Invalid password"), status_code=401)

        @fastapi_app.get("/logout")
        async def logout():
            """Clear auth cookie."""
            response = RedirectResponse(url="/", status_code=303)
            response.delete_cookie(key="auth")
            return response

        @fastapi_app.get("/favicon.ico")
        async def favicon():
            """Serve favicon."""
            from fastapi.responses import FileResponse
            icon_path = Path("/app/static/favicon.ico")
            if icon_path.exists():
                return FileResponse(icon_path)
            raise HTTPException(status_code=404, detail="File not found")

        @fastapi_app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "model": "personaplex"}

        @fastapi_app.get("/static/{filename}")
        async def static_files(filename: str):
            """Serve static files (decoder worker, wasm, etc.)."""
            from fastapi.responses import FileResponse
            static_path = Path(f"/app/static/{filename}")
            if static_path.exists():
                # Set correct content type for wasm files
                content_type = None
                if filename.endswith('.wasm'):
                    content_type = 'application/wasm'
                elif filename.endswith('.js'):
                    content_type = 'application/javascript'
                return FileResponse(static_path, media_type=content_type)
            raise HTTPException(status_code=404, detail="File not found")

        @fastapi_app.get("/voices")
        async def list_voices(auth: str | None = Cookie(default=None)):
            """List available voice presets."""
            if not check_auth(auth):
                raise HTTPException(status_code=401, detail="Not authenticated")
            return JSONResponse(content={
                "voices": list(VOICE_PRESETS.keys()),
                "default": DEFAULT_VOICE,
            })

        @fastapi_app.websocket("/ws/chat")
        async def websocket_chat(
            websocket: WebSocket,
            voice_prompt: str = Query(default=DEFAULT_VOICE),
            text_prompt: str = Query(default=DEFAULT_TEXT_PROMPT),
            seed: int = Query(default=-1),
        ):
            """
            WebSocket endpoint for voice chat.

            Query params:
            - voice_prompt: Voice preset name (e.g., "NATF2")
            - text_prompt: System prompt for persona
            - seed: Random seed (-1 for random)
            """
            # Check auth cookie
            auth_cookie = websocket.cookies.get("auth")
            if not auth_cookie or not verify_session_token(auth_cookie):
                await websocket.close(code=4001, reason="Not authenticated")
                return

            await websocket.accept()
            print(f"WebSocket connection accepted from {websocket.client}")

            try:
                await self._handle_chat(websocket, voice_prompt, text_prompt, seed)
            except WebSocketDisconnect:
                print("Client disconnected")
            except Exception as e:
                print(f"WebSocket error: {e}")
                await websocket.close(code=1011, reason=str(e))

        return fastapi_app

    async def _handle_chat(self, websocket, voice_prompt_name: str, text_prompt: str, seed: int):
        """Handle a WebSocket chat session."""
        import asyncio
        import torch
        import numpy as np
        import sphn
        import time
        import random

        def seed_all(seed_val):
            torch.manual_seed(seed_val)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(seed_val)
                torch.cuda.manual_seed_all(seed_val)
            random.seed(seed_val)
            np.random.seed(seed_val)

        def wrap_with_system_tags(text: str) -> str:
            cleaned = text.strip()
            if cleaned.startswith("<system>") and cleaned.endswith("<system>"):
                return cleaned
            return f"<system> {cleaned} <system>"

        # Resolve voice prompt path
        voice_filename = VOICE_PRESETS.get(voice_prompt_name, VOICE_PRESETS[DEFAULT_VOICE])
        voice_prompt_path = str(self.voice_prompt_dir / voice_filename)

        print(f"Using voice: {voice_prompt_name} ({voice_prompt_path})")
        print(f"Text prompt: {text_prompt}")

        close = False

        async def recv_loop():
            nonlocal close
            try:
                while not close:
                    try:
                        message = await asyncio.wait_for(websocket.receive_bytes(), timeout=0.1)
                        if len(message) == 0:
                            continue
                        kind = message[0]
                        if kind == 1:  # audio
                            payload = message[1:]
                            opus_reader.append_bytes(payload)
                    except asyncio.TimeoutError:
                        continue
            except Exception as e:
                print(f"recv_loop error: {e}")
            finally:
                close = True

        async def opus_loop():
            nonlocal close
            all_pcm_data = None

            while not close:
                await asyncio.sleep(0.001)
                pcm = opus_reader.read_pcm()
                if pcm.shape[-1] == 0:
                    continue
                if all_pcm_data is None:
                    all_pcm_data = pcm
                else:
                    all_pcm_data = np.concatenate((all_pcm_data, pcm))

                while all_pcm_data.shape[-1] >= self.frame_size:
                    chunk = all_pcm_data[:self.frame_size]
                    all_pcm_data = all_pcm_data[self.frame_size:]
                    chunk = torch.from_numpy(chunk)
                    chunk = chunk.to(device=self.device)[None, None]
                    codes = self.mimi.encode(chunk)
                    _ = self.other_mimi.encode(chunk)

                    for c in range(codes.shape[-1]):
                        tokens = self.lm_gen.step(codes[:, :, c: c + 1])
                        if tokens is None:
                            continue

                        main_pcm = self.mimi.decode(tokens[:, 1:9])
                        _ = self.other_mimi.decode(tokens[:, 1:9])
                        main_pcm = main_pcm.cpu()
                        opus_writer.append_pcm(main_pcm[0, 0].detach().numpy())

                        # Send text tokens
                        text_token = tokens[0, 0, 0].item()
                        if text_token not in (0, 3):
                            _text = self.text_tokenizer.id_to_piece(text_token)
                            _text = _text.replace("▁", " ")
                            msg = b"\x02" + bytes(_text, encoding="utf8")
                            await websocket.send_bytes(msg)

        async def send_loop():
            nonlocal close
            while not close:
                await asyncio.sleep(0.001)
                msg = opus_writer.read_bytes()
                if len(msg) > 0:
                    await websocket.send_bytes(b"\x01" + msg)

        async with self.lock:
            if seed != -1:
                seed_all(seed)

            opus_writer = sphn.OpusStreamWriter(self.mimi.sample_rate)
            opus_reader = sphn.OpusStreamReader(self.mimi.sample_rate)
            self.mimi.reset_streaming()
            self.other_mimi.reset_streaming()
            self.lm_gen.reset_streaming()

            # Load voice prompt
            if voice_prompt_path.endswith('.pt'):
                self.lm_gen.load_voice_prompt_embeddings(voice_prompt_path)
            else:
                self.lm_gen.load_voice_prompt(voice_prompt_path)

            # Set text prompt
            if text_prompt:
                self.lm_gen.text_prompt_tokens = self.text_tokenizer.encode(wrap_with_system_tags(text_prompt))
            else:
                self.lm_gen.text_prompt_tokens = None

            # Process system prompts
            async def is_alive():
                return not close

            await self.lm_gen.step_system_prompts_async(self.mimi, is_alive=is_alive)
            self.mimi.reset_streaming()
            print("System prompts processed")

            # Send handshake
            await websocket.send_bytes(b"\x00")
            print("Handshake sent")

            # Run concurrent loops
            tasks = [
                asyncio.create_task(recv_loop()),
                asyncio.create_task(opus_loop()),
                asyncio.create_task(send_loop()),
            ]

            done, pending = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

            # Cleanup
            close = True
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            print("Chat session ended")

    def _get_login_page(self, error: str | None = None):
        """Return a simple login page."""
        error_html = f'<p class="error">{error}</p>' if error else ''
        return f"""
<!DOCTYPE html>
<html>
<head>
    <title>PersonaPlex - Login</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="/favicon.ico">
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            min-height: 100vh;
            display: flex;
            align-items: center;
            justify-content: center;
            margin: 0;
        }}
        .login-box {{
            background: white;
            padding: 40px;
            border-radius: 16px;
            box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
            text-align: center;
            max-width: 400px;
            width: 90%;
        }}
        h1 {{ color: #333; margin-bottom: 10px; }}
        p {{ color: #666; margin-bottom: 30px; }}
        .error {{ color: #e74c3c; margin-bottom: 20px; }}
        input[type="password"] {{
            width: 100%;
            padding: 14px;
            font-size: 16px;
            border: 2px solid #e0e0e0;
            border-radius: 8px;
            margin-bottom: 20px;
            box-sizing: border-box;
        }}
        input[type="password"]:focus {{
            outline: none;
            border-color: #667eea;
        }}
        button {{
            width: 100%;
            padding: 14px;
            font-size: 16px;
            font-weight: 600;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            border: none;
            border-radius: 8px;
            cursor: pointer;
        }}
        button:hover {{
            opacity: 0.9;
        }}
    </style>
</head>
<body>
    <div class="login-box">
        <h1>🎙️ PersonaPlex</h1>
        <p>Voice-to-voice AI</p>
        {error_html}
        <form method="POST" action="/login">
            <input type="password" name="password" placeholder="Enter password" required autofocus>
            <button type="submit">Sign In</button>
        </form>
    </div>
</body>
</html>
"""

    def _get_default_ui(self):
        """Return default HTML UI if static/index.html doesn't exist."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>PersonaPlex Voice AI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <link rel="icon" href="/favicon.ico">
    <style>
        body { font-family: system-ui; max-width: 800px; margin: 0 auto; padding: 20px; }
        h1 { color: #333; }
        .controls { margin: 20px 0; }
        button { padding: 10px 20px; font-size: 16px; margin: 5px; cursor: pointer; }
        #status { padding: 10px; background: #f0f0f0; border-radius: 5px; margin: 10px 0; }
        #transcript { min-height: 200px; border: 1px solid #ccc; padding: 10px; margin: 10px 0; }
        select, input { padding: 8px; font-size: 14px; margin: 5px; }
        .error { color: red; }
        .connected { color: green; }
    </style>
</head>
<body>
    <h1>🎙️ PersonaPlex Voice AI</h1>
    <p>Real-time voice-to-voice conversation with AI personas.</p>

    <div class="controls">
        <label>Voice: <select id="voice">
            <option value="NATF2">Natural Female</option>
            <option value="NATM2">Natural Male</option>
            <option value="VARF0">Variety Female</option>
            <option value="VARM0">Variety Male</option>
        </select></label>
        <br><br>
        <label>Persona: <input type="text" id="textPrompt" value="You are a helpful assistant." size="50"></label>
    </div>

    <div class="controls">
        <button id="startBtn" onclick="startChat()">🎤 Start Talking</button>
        <button id="stopBtn" onclick="stopChat()" disabled>⏹️ Stop</button>
    </div>

    <div id="status">Status: Ready</div>
    <div id="transcript"><em>Transcript will appear here...</em></div>

    <script>
        // Note: Full WebSocket + Opus implementation requires additional JS libraries
        // This is a placeholder - see static/index.html for full implementation

        let ws = null;

        function setStatus(msg, isError = false) {
            const el = document.getElementById('status');
            el.textContent = 'Status: ' + msg;
            el.className = isError ? 'error' : (msg.includes('Connected') ? 'connected' : '');
        }

        function startChat() {
            setStatus('Connecting...');
            // Full implementation in static/index.html
            alert('Full audio streaming requires the complete client. See static/index.html');
        }

        function stopChat() {
            if (ws) {
                ws.close();
                ws = null;
            }
            setStatus('Disconnected');
            document.getElementById('startBtn').disabled = false;
            document.getElementById('stopBtn').disabled = true;
        }
    </script>
</body>
</html>
"""


# ------------------------------------------------------------------------------
# Local entrypoint for testing
# ------------------------------------------------------------------------------

@app.local_entrypoint()
def main():
    """Local entrypoint for testing."""
    print("PersonaPlex Modal App")
    print("=====================")
    print()
    print("To run the server locally for development:")
    print("  modal serve personaplex.py")
    print()
    print("To deploy to Modal:")
    print("  modal deploy personaplex.py")
    print()
    print("Make sure you have set up the HuggingFace secret:")
    print("  modal secret create huggingface-secret HF_TOKEN=hf_xxxxx")
