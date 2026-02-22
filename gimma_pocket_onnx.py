"""
PocketTTS ONNX - Pure ONNX inference for Pocket TTS

A standalone, production-ready class for text-to-speech with voice cloning.
Supports both offline (batch) and streaming modes with adaptive chunking.

Dependencies:
    - onnxruntime (or onnxruntime-gpu for CUDA)
    - numpy
    - soundfile
    - sentencepiece
    - scipy (for resampling)

Usage:
    from pocket_tts_onnx import PocketTTSOnnx

    # Initialize with INT8 (CPU optimized - default, fastest)
    tts = PocketTTSOnnx()

    # Voice cloning from audio file
    audio = tts.generate("Hello world!", voice="samples/reference.wav")

    # Streaming with adaptive chunking
    for chunk in tts.stream("Hello world!", voice="samples/reference.wav"):
        play_audio(chunk)  # Process each chunk as it's ready
"""

import os
import queue
import threading
import time
from pathlib import Path
from typing import Generator, Optional, Union
import numpy as np
import onnxruntime as ort
import sentencepiece as spm

# Optional imports
try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False

try:
    import scipy.signal
    HAS_SCIPY = True
except ImportError:
    HAS_SCIPY = False


class PocketTTSOnnx:
    """
    Pure ONNX inference engine for Pocket TTS.

    Supports:
        - Offline (batch) generation
        - Streaming generation with adaptive chunking
        - INT8 and FP32 models
        - Voice cloning from audio files
        - Auto GPU/CPU detection
        - Temperature control for generation diversity

    Args:
        models_dir: Directory containing ONNX models
        tokenizer_path: Path to sentencepiece tokenizer.model
        precision: Model precision - "int8" (CPU optimized, fastest) or "fp32" (full precision)
        device: "auto", "cpu", or "cuda"
        temperature: Sampling temperature (0.0 = deterministic, 0.7 = default, 1.0 = more diverse)
        lsd_steps: Number of flow matching steps (default 10, lower = faster but lower quality)
    """

    SAMPLE_RATE = 24000
    SAMPLES_PER_FRAME = 1920
    FRAME_DURATION = SAMPLES_PER_FRAME / SAMPLE_RATE  # 0.08s per frame

    VALID_PRECISIONS = ("int8", "fp32")

    def __init__(
        self,
        models_dir: str = "onnx",
        tokenizer_path: str = "tokenizer.model",
        precision: str = "int8",
        device: str = "auto",
        temperature: float = 0.7,
        lsd_steps: int = 10,
    ):
        self.models_dir = Path(models_dir)

        if precision not in self.VALID_PRECISIONS:
            raise ValueError(f"precision must be one of {self.VALID_PRECISIONS}, got '{precision}'")

        self.precision = precision
        self.temperature = temperature
        self.lsd_steps = lsd_steps

        # Setup execution providers
        self.providers = self._get_providers(device)

        # Load tokenizer
        self.tokenizer = spm.SentencePieceProcessor()
        self.tokenizer.Load(str(tokenizer_path))

        # Load models
        self._load_models()

        # Pre-compute s/t buffers for flow matching
        self._precompute_flow_buffers()

        # --- OPTIMIZATION: Pre-compute state I/O mappings once ---
        # These are used in the innermost hot loops.  Computing them once at
        # init (instead of calling get_outputs() + string-parse on every frame)
        # eliminates hundreds of redundant Python-level allocations.
        self._flow_main_state_map = self._build_state_output_map(self.flow_lm_main)
        self._mimi_decoder_state_map = self._build_state_output_map(self.mimi_decoder)

        # Pre-build the zero-state prototype so _init_state just copies it
        self._flow_main_zero_state = self._build_zero_state(self.flow_lm_main)
        self._mimi_decoder_zero_state = self._build_zero_state(self.mimi_decoder)

        # Cache for voice embeddings
        self._voice_cache = {}

    # ------------------------------------------------------------------
    # Session / provider helpers
    # ------------------------------------------------------------------

    def _get_providers(self, device: str) -> list:
        """Get ONNX execution providers based on device setting."""
        if device == "cpu":
            return ["CPUExecutionProvider"]
        elif device == "cuda":
            return ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:  # auto
            available = ort.get_available_providers()
            if "CUDAExecutionProvider" in available:
                return ["CUDAExecutionProvider", "CPUExecutionProvider"]
            return ["CPUExecutionProvider"]

    def _make_session_options(self) -> ort.SessionOptions:
        """Create optimized session options for ONNX inference.

        Caps intra-op threads to avoid over-subscription overhead on the
        small sequential matmuls in the autoregressive loop.  The sweet
        spot on a 16-core machine is 3-8; we use min(cpu_count, 4) so
        low-core machines aren't over-committed and high-core machines
        don't hit the contention cliff.
        """
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = min(os.cpu_count() or 4, 4)
        opts.inter_op_num_threads = 1
        # Enable graph optimization at session creation time
        opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        return opts

    def _load_models(self):
        """Load ONNX models (dual model architecture)."""
        # Select model files based on precision
        suffix = "_int8" if self.precision == "int8" else ""
        flow_main_file = f"flow_lm_main{suffix}.onnx"
        flow_flow_file = f"flow_lm_flow{suffix}.onnx"
        mimi_file = f"mimi_decoder{suffix}.onnx"

        sess_opts = self._make_session_options()

        self.mimi_encoder = ort.InferenceSession(
            str(self.models_dir / "mimi_encoder.onnx"),
            sess_options=sess_opts, providers=self.providers
        )
        self.text_conditioner = ort.InferenceSession(
            str(self.models_dir / "text_conditioner.onnx"),
            sess_options=sess_opts, providers=self.providers
        )
        # Dual model split: main (transformer) + flow (flow network)
        self.flow_lm_main = ort.InferenceSession(
            str(self.models_dir / flow_main_file),
            sess_options=sess_opts, providers=self.providers
        )
        self.flow_lm_flow = ort.InferenceSession(
            str(self.models_dir / flow_flow_file),
            sess_options=sess_opts, providers=self.providers
        )
        self.mimi_decoder = ort.InferenceSession(
            str(self.models_dir / mimi_file),
            sess_options=sess_opts, providers=self.providers
        )

    # ------------------------------------------------------------------
    # OPTIMIZATION: Pre-compute output-index → state-key maps
    # ------------------------------------------------------------------

    def _build_state_output_map(self, session: ort.InferenceSession) -> list:
        """
        Return a list of (output_index, state_key) pairs for every output
        whose name starts with 'out_state_'.  Built once at init; used in
        hot loops instead of calling get_outputs() + string operations every
        frame.
        """
        mapping = []
        for i, out in enumerate(session.get_outputs()):
            if out.name.startswith("out_state_"):
                idx = int(out.name.replace("out_state_", ""))
                mapping.append((i, f"state_{idx}"))
        return mapping

    def _build_zero_state(self, session: ort.InferenceSession) -> dict:
        """Build a zero-initialized state dict (template for deep-copy on use)."""
        state = {}
        type_map = {
            "tensor(float)": np.float32,
            "tensor(int64)": np.int64,
            "tensor(bool)": np.bool_,
        }
        for inp in session.get_inputs():
            if inp.name.startswith("state_"):
                shape = [s if isinstance(s, int) else 0 for s in inp.shape]
                dtype = type_map.get(inp.type, np.float32)
                state[inp.name] = np.zeros(shape, dtype=dtype)
        return state

    def _precompute_flow_buffers(self):
        """Pre-compute s/t time step buffers for flow matching."""
        dt = 1.0 / self.lsd_steps
        self._st_buffers = []
        for j in range(self.lsd_steps):
            s = j / self.lsd_steps
            t = s + dt
            self._st_buffers.append((
                np.array([[s]], dtype=np.float32),
                np.array([[t]], dtype=np.float32)
            ))
        self._flow_dt = np.float32(dt)

    def _init_state(self, session: ort.InferenceSession) -> dict:
        """Initialize state tensors for a stateful model (zero-copy from template)."""
        # Use pre-built templates for the two models we care about
        if session is self.flow_lm_main:
            return {k: v.copy() for k, v in self._flow_main_zero_state.items()}
        if session is self.mimi_decoder:
            return {k: v.copy() for k, v in self._mimi_decoder_zero_state.items()}
        # Fallback for any other session
        return self._build_zero_state(session)

    # ------------------------------------------------------------------
    # Fast state-update helpers (no get_outputs() in hot path)
    # ------------------------------------------------------------------

    def _update_state_fast(self, state: dict, result: list, state_map: list):
        """
        Update state dict from model outputs using the pre-built map.
        Replaces the old _update_state_from_outputs() which called
        session.get_outputs() on every invocation.
        """
        for out_idx, state_key in state_map:
            state[state_key] = result[out_idx]

    def _update_state_from_outputs(self, state: dict, result: list, session: ort.InferenceSession):
        """Legacy wrapper — kept for external callers; internally uses fast path."""
        if session is self.flow_lm_main:
            self._update_state_fast(state, result, self._flow_main_state_map)
        elif session is self.mimi_decoder:
            self._update_state_fast(state, result, self._mimi_decoder_state_map)
        else:
            for i in range(2, len(session.get_outputs())):
                name = session.get_outputs()[i].name
                if name.startswith("out_state_"):
                    idx = int(name.replace("out_state_", ""))
                    state[f"state_{idx}"] = result[i]

    def _increment_step(self, state: dict, n: int):
        """Increment step counters in state dict."""
        for k in state:
            if "step" in k:
                state[k] = (state[k] + n).astype(np.int64)

    # ------------------------------------------------------------------
    # Audio / voice helpers
    # ------------------------------------------------------------------

    def _load_audio(self, path: Union[str, Path]) -> np.ndarray:
        """Load and preprocess audio file for voice cloning."""
        if not HAS_SOUNDFILE:
            raise ImportError("soundfile required for voice cloning. Install with: pip install soundfile")

        audio, sr = sf.read(str(path))

        # Convert to mono
        if len(audio.shape) > 1:
            audio = audio.mean(axis=1)

        # Resample to 24kHz if needed
        if sr != self.SAMPLE_RATE:
            if not HAS_SCIPY:
                raise ImportError("scipy required for resampling. Install with: pip install scipy")
            num_samples = int(len(audio) * self.SAMPLE_RATE / sr)
            audio = scipy.signal.resample(audio, num_samples)

        # Normalize and reshape
        audio = audio.astype(np.float32)
        if np.abs(audio).max() > 1.0:
            audio = audio / np.abs(audio).max()

        return audio.reshape(1, 1, -1)

    def encode_voice(self, audio_path: Union[str, Path]) -> np.ndarray:
        """
        Encode an audio file into voice embeddings for cloning.

        Args:
            audio_path: Path to audio file (wav, mp3, etc.)

        Returns:
            Voice embeddings array [1, N, 1024]
        """
        audio = self._load_audio(audio_path)
        embeddings = self.mimi_encoder.run(None, {"audio": audio})[0]

        # Normalize dimensions to [1, N, 1024]
        while embeddings.ndim > 3:
            embeddings = embeddings.squeeze(0)
        if embeddings.ndim < 3:
            embeddings = embeddings[None]

        return embeddings

    def _get_voice_embeddings(self, voice: Union[str, Path, np.ndarray]) -> np.ndarray:
        """Get voice embeddings from various input types."""
        # Already embeddings
        if isinstance(voice, np.ndarray):
            return voice

        voice_str = str(voice)

        # Check cache
        if voice_str in self._voice_cache:
            return self._voice_cache[voice_str]

        # Audio file
        if os.path.exists(voice_str):
            embeddings = self.encode_voice(voice_str)
        else:
            raise ValueError(f"Voice file '{voice_str}' not found.")

        # Cache and return
        self._voice_cache[voice_str] = embeddings
        return embeddings

    def _tokenize(self, text: str) -> np.ndarray:
        """Tokenize text for the model."""
        text = text.strip()
        if not text:
            raise ValueError("Text cannot be empty")

        # Ensure proper punctuation
        if text[-1].isalnum():
            text = text + "."
        if not text[0].isupper():
            text = text[0].upper() + text[1:]

        token_ids = self.tokenizer.Encode(text)
        return np.array(token_ids, dtype=np.int64).reshape(1, -1)

    # ------------------------------------------------------------------
    # Core generation
    # ------------------------------------------------------------------

    def _run_flow_lm(
        self,
        voice_embeddings: np.ndarray,
        text_ids: np.ndarray,
        max_frames: int = 500,
        frames_after_eos: int = 3,
    ) -> Generator[np.ndarray, None, None]:
        """
        Run flow LM autoregressive generation, yielding latents.

        Uses dual model architecture:
        - flow_lm_main: transformer/conditioner (produces conditioning vector)
        - flow_lm_flow: flow network (Euler integration for latent sampling)

        Yields individual latent frames as they're generated.

        Key optimizations vs. original:
        - State updates use pre-built index map — no get_outputs() per frame
        - Noise buffer pre-allocated once; refilled in-place each frame
        - empty_text/empty_seq are module-level constants (allocated once)
        - flow_dt is a pre-cast float32 scalar
        """
        # Text conditioning
        text_emb = self.text_conditioner.run(None, {"token_ids": text_ids})[0]
        if text_emb.ndim == 2:
            text_emb = text_emb[None]

        # Initialize state for flow_lm_main
        state = self._init_state(self.flow_lm_main)

        empty_seq = np.zeros((1, 0, 32), dtype=np.float32)
        empty_text = np.zeros((1, 0, 1024), dtype=np.float32)
        flow_main_map = self._flow_main_state_map

        # Voice conditioning pass
        res_voice = self.flow_lm_main.run(None, {
            "sequence": empty_seq,
            "text_embeddings": voice_embeddings,
            **state
        })
        self._update_state_fast(state, res_voice, flow_main_map)

        # Text conditioning pass
        res_text = self.flow_lm_main.run(None, {
            "sequence": empty_seq,
            "text_embeddings": text_emb,
            **state
        })
        self._update_state_fast(state, res_text, flow_main_map)

        # --- OPTIMIZATION: pre-allocate noise buffer ---
        # np.random.normal returns a new array every call.  Pre-allocating and
        # refilling in-place avoids one heap allocation per frame.
        std = float(np.sqrt(self.temperature)) if self.temperature > 0 else 0.0
        noise_buf = np.empty((1, 32), dtype=np.float32)

        # Autoregressive generation
        curr = np.full((1, 1, 32), np.nan, dtype=np.float32)
        dt = self._flow_dt
        st_buffers = self._st_buffers
        lsd_steps = self.lsd_steps

        eos_step = None

        for step in range(max_frames):
            # Run main model to get conditioning and EOS
            res_step = self.flow_lm_main.run(None, {
                "sequence": curr,
                "text_embeddings": empty_text,
                **state
            })

            conditioning = res_step[0]  # [1, 1, dim]
            eos_logit = res_step[1]     # [1, 1]

            # Update state using fast path (no get_outputs() call)
            self._update_state_fast(state, res_step, flow_main_map)

            # Check EOS
            if eos_logit[0][0] > -4.0 and eos_step is None:
                eos_step = step

            # Stop only after frames_after_eos additional frames
            if eos_step is not None and step >= eos_step + frames_after_eos:
                break

            # Flow matching — Euler integration
            # Re-use pre-allocated buffer instead of allocating each frame
            if std > 0:
                noise_buf[:] = np.random.normal(0, std, noise_buf.shape).astype(np.float32)
                x = noise_buf.copy()
            else:
                x = np.zeros((1, 32), dtype=np.float32)

            for j in range(lsd_steps):
                s_arr, t_arr = st_buffers[j]
                flow_out = self.flow_lm_flow.run(None, {
                    "c": conditioning,
                    "s": s_arr,
                    "t": t_arr,
                    "x": x
                })
                x += flow_out[0] * dt  # in-place add avoids one allocation per step

            latent = x.reshape(1, 1, 32)
            yield latent
            curr = latent

    def _decode_latents(self, latents: np.ndarray, chunk_size: int = 15) -> np.ndarray:
        """
        Decode latents to audio using chunked decoding with state updates.

        The MIMI decoder requires proper state updates between chunks to avoid
        garbled audio. Batch decoding without state updates causes artifacts.

        Args:
            latents: Latent array of shape [1, num_frames, 32]
            chunk_size: Number of frames to decode at once (default 15 for speed)

        Returns:
            Audio samples as numpy array
        """
        state = self._init_state(self.mimi_decoder)
        mimi_map = self._mimi_decoder_state_map
        audio_chunks = []
        num_frames = latents.shape[1]

        for i in range(0, num_frames, chunk_size):
            chunk = latents[:, i:i+chunk_size, :]
            result = self.mimi_decoder.run(None, {"latent": chunk, **state})
            audio_chunks.append(result[0].squeeze())
            self._update_state_fast(state, result, mimi_map)

        return np.concatenate(audio_chunks)

    def _decode_worker(self, latent_queue: queue.Queue, audio_chunks: list,
                       decode_chunk_size: int = 12):
        """Decode latents from a queue in a background thread."""
        mimi_state = self._init_state(self.mimi_decoder)
        mimi_map = self._mimi_decoder_state_map
        buf = []
        decoded = 0

        while True:
            item = latent_queue.get()
            if item is None:
                break
            buf.append(item)

            if len(buf) - decoded >= decode_chunk_size:
                chunk = np.concatenate(buf[decoded:decoded + decode_chunk_size], axis=1)
                result = self.mimi_decoder.run(None, {"latent": chunk, **mimi_state})
                audio_chunks.append(result[0].squeeze())
                # --- OPTIMIZATION: fast state update (no get_outputs() per chunk) ---
                self._update_state_fast(mimi_state, result, mimi_map)
                decoded += decode_chunk_size

        # Decode remaining
        if decoded < len(buf):
            remaining = np.concatenate(buf[decoded:], axis=1)
            result = self.mimi_decoder.run(None, {"latent": remaining, **mimi_state})
            audio_chunks.append(result[0].squeeze())

    def generate(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: int = 500,
    ) -> np.ndarray:
        """
        Generate audio from text (offline/batch mode).

        Runs flow LM generation and mimi decoding in parallel threads
        for maximum throughput.

        Args:
            text: Text to synthesize
            voice: Audio file path for voice cloning, or pre-computed embeddings
            max_frames: Maximum latent frames to generate

        Returns:
            Audio samples as numpy array (float32, 24kHz)
        """
        voice_emb = self._get_voice_embeddings(voice)
        text_ids = self._tokenize(text)

        # Start decode worker thread
        latent_queue = queue.Queue()
        audio_chunks = []
        decoder = threading.Thread(
            target=self._decode_worker,
            args=(latent_queue, audio_chunks),
            daemon=True,
        )
        decoder.start()

        # Generate latents and feed to decoder
        for latent in self._run_flow_lm(voice_emb, text_ids, max_frames):
            latent_queue.put(latent)
        latent_queue.put(None)  # sentinel

        decoder.join()
        return np.concatenate(audio_chunks)

    def stream(
        self,
        text: str,
        voice: Union[str, Path, np.ndarray],
        max_frames: int = 500,
        first_chunk_frames: int = 2,
        target_buffer_sec: float = 0.2,
        max_chunk_frames: int = 15,
    ) -> Generator[np.ndarray, None, None]:
        """
        Stream audio generation with adaptive chunking.

        Yields audio chunks as they become available, optimizing for:
        - Low TTFB (time to first audio)
        - Smooth real-time playback
        - High overall throughput

        Args:
            text: Text to synthesize
            voice: Audio file path for voice cloning, or pre-computed embeddings
            max_frames: Maximum latent frames to generate
            first_chunk_frames: Frames in first chunk (controls TTFB)
            target_buffer_sec: Target buffer ahead of playback
            max_chunk_frames: Maximum frames per chunk

        Yields:
            Audio chunks as numpy arrays (float32, 24kHz)
        """
        voice_emb = self._get_voice_embeddings(voice)
        text_ids = self._tokenize(text)

        # State tracking
        mimi_state = self._init_state(self.mimi_decoder)
        mimi_map = self._mimi_decoder_state_map
        generated_latents = []
        decoded_frames = 0
        playback_start_time = None
        start_time = time.time()

        for latent in self._run_flow_lm(voice_emb, text_ids, max_frames):
            generated_latents.append(latent)
            pending = len(generated_latents) - decoded_frames

            chunk_size = 0

            if playback_start_time is None:
                if pending >= first_chunk_frames:
                    chunk_size = first_chunk_frames
            else:
                elapsed = time.time() - start_time
                audio_decoded_sec = decoded_frames * self.FRAME_DURATION
                playback_elapsed = elapsed - playback_start_time
                buffer_sec = audio_decoded_sec - playback_elapsed

                if buffer_sec < target_buffer_sec and pending >= 1:
                    chunk_size = min(pending, 3)
                elif pending >= max_chunk_frames:
                    chunk_size = max_chunk_frames

            if chunk_size > 0:
                latents_chunk = np.concatenate(
                    generated_latents[decoded_frames:decoded_frames + chunk_size],
                    axis=1
                )

                res = self.mimi_decoder.run(None, {"latent": latents_chunk, **mimi_state})
                audio_chunk = res[0].squeeze()

                # --- OPTIMIZATION: fast state update ---
                self._update_state_fast(mimi_state, res, mimi_map)

                decoded_frames += chunk_size

                if playback_start_time is None:
                    playback_start_time = time.time() - start_time

                yield audio_chunk

        # Decode remaining latents
        if decoded_frames < len(generated_latents):
            remaining_latents = np.concatenate(
                generated_latents[decoded_frames:],
                axis=1
            )
            res = self.mimi_decoder.run(None, {"latent": remaining_latents, **mimi_state})
            yield res[0].squeeze()

    def save_audio(self, audio: np.ndarray, path: Union[str, Path]):
        """Save audio to file."""
        if not HAS_SOUNDFILE:
            raise ImportError("soundfile required. Install with: pip install soundfile")
        sf.write(str(path), audio, self.SAMPLE_RATE)

    @property
    def device(self) -> str:
        """Return the device being used."""
        if "CUDAExecutionProvider" in self.providers:
            return "cuda"
        return "cpu"

    def __repr__(self) -> str:
        return (
            f"PocketTTSOnnx("
            f"device={self.device!r}, "
            f"precision={self.precision!r}, "
            f"temperature={self.temperature}, "
            f"lsd_steps={self.lsd_steps}, "
            f"sample_rate={self.SAMPLE_RATE})"
        )