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
GPU_TYPE = "A10G"  # A10G-24GB, use cpu_offload if OOM
CONTAINER_IDLE_TIMEOUT = 300  # 5 minutes
MODEL_CACHE_PATH = "/cache"
VOICE_PROMPT_DIR = "/cache/voices"

# Simple password auth (set in Modal secrets or here for testing)
AUTH_PASSWORD = "REDACTED_PASSWORD"

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
    modal.Image.from_registry(
        "nvcr.io/nvidia/cuda:12.4.1-runtime-ubuntu22.04",
        add_python="3.11",
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
    })
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
    secrets=[modal.Secret.from_name("huggingface-secret")],
    image=image,
    container_idle_timeout=CONTAINER_IDLE_TIMEOUT,
    timeout=600,  # 10 min max request time
)
class PersonaPlexService:
    """
    Modal service class for PersonaPlex voice-to-voice AI.

    Handles model loading, WebSocket connections, and audio streaming.
    """

    @modal.enter()
    def load_models(self):
        """Load models when container starts (runs once per container lifecycle)."""
        import torch
        import sentencepiece
        from huggingface_hub import hf_hub_download
        from moshi.models import loaders
        import tarfile
        from pathlib import Path
        import asyncio

        print("Loading PersonaPlex models...")

        # Set device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # HuggingFace repo for PersonaPlex
        hf_repo = loaders.DEFAULT_REPO

        # Download and load Mimi audio codec
        print("Loading Mimi audio codec...")
        mimi_weight = hf_hub_download(hf_repo, loaders.MIMI_NAME)
        self.mimi = loaders.get_mimi(mimi_weight, self.device)
        self.other_mimi = loaders.get_mimi(mimi_weight, self.device)
        print("Mimi loaded")

        # Download and load text tokenizer
        print("Loading text tokenizer...")
        tokenizer_path = hf_hub_download(hf_repo, loaders.TEXT_TOKENIZER_NAME)
        self.text_tokenizer = sentencepiece.SentencePieceProcessor(tokenizer_path)
        print("Tokenizer loaded")

        # Download and load Moshi language model
        print("Loading Moshi language model (this may take a while)...")
        moshi_weight = hf_hub_download(hf_repo, loaders.MOSHI_NAME)
        # Try with cpu_offload for A10G
        try:
            self.lm = loaders.get_moshi_lm(moshi_weight, device=self.device, cpu_offload=True)
        except Exception as e:
            print(f"CPU offload failed, trying without: {e}")
            self.lm = loaders.get_moshi_lm(moshi_weight, device=self.device, cpu_offload=False)
        self.lm.eval()
        print("Moshi loaded")

        # Download voice prompts
        print("Loading voice prompts...")
        voices_tgz = hf_hub_download(hf_repo, "voices.tgz")
        voices_tgz = Path(voices_tgz)
        self.voice_prompt_dir = voices_tgz.parent / "voices"

        if not self.voice_prompt_dir.exists():
            print(f"Extracting voice prompts to {self.voice_prompt_dir}")
            with tarfile.open(voices_tgz, "r:gz") as tar:
                tar.extractall(path=voices_tgz.parent)
        print(f"Voice prompts ready at {self.voice_prompt_dir}")

        # Create ServerState
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

        # Warmup
        print("Warming up models...")
        self._warmup()

        # Commit volume to persist cached models
        volume.commit()
        print("PersonaPlex ready!")

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
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException, Depends, Query
        from fastapi.responses import HTMLResponse, JSONResponse
        from fastapi.staticfiles import StaticFiles
        from fastapi.security import HTTPBasic, HTTPBasicCredentials
        import secrets
        from pathlib import Path

        fastapi_app = FastAPI(title="PersonaPlex Voice AI")
        security = HTTPBasic()

        def verify_password(credentials: HTTPBasicCredentials = Depends(security)):
            """Simple password verification."""
            if not secrets.compare_digest(credentials.password, AUTH_PASSWORD):
                raise HTTPException(status_code=401, detail="Invalid password")
            return True

        @fastapi_app.get("/")
        async def root(authenticated: bool = Depends(verify_password)):
            """Serve the web UI."""
            static_dir = Path(__file__).parent / "static"
            index_path = static_dir / "index.html"
            if index_path.exists():
                return HTMLResponse(content=index_path.read_text())
            return HTMLResponse(content=self._get_default_ui())

        @fastapi_app.get("/health")
        async def health():
            """Health check endpoint."""
            return {"status": "healthy", "model": "personaplex"}

        @fastapi_app.get("/voices")
        async def list_voices(authenticated: bool = Depends(verify_password)):
            """List available voice presets."""
            return JSONResponse(content={
                "voices": list(VOICE_PRESETS.keys()),
                "default": DEFAULT_VOICE,
            })

        @fastapi_app.websocket("/ws/chat")
        async def websocket_chat(
            websocket: WebSocket,
            password: str = Query(...),
            voice_prompt: str = Query(default=DEFAULT_VOICE),
            text_prompt: str = Query(default=DEFAULT_TEXT_PROMPT),
            seed: int = Query(default=-1),
        ):
            """
            WebSocket endpoint for voice chat.

            Query params:
            - password: Auth password
            - voice_prompt: Voice preset name (e.g., "NATF2")
            - text_prompt: System prompt for persona
            - seed: Random seed (-1 for random)
            """
            # Verify password
            if not secrets.compare_digest(password, AUTH_PASSWORD):
                await websocket.close(code=4001, reason="Invalid password")
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
                        opus_writer.append_pcm(main_pcm[0, 0].numpy())

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

    def _get_default_ui(self):
        """Return default HTML UI if static/index.html doesn't exist."""
        return """
<!DOCTYPE html>
<html>
<head>
    <title>PersonaPlex Voice AI</title>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
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
