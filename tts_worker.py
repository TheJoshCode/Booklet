import os
import time
import logging
from typing import Optional
import torch
from pocket_tts import TTSModel
from scipy.io.wavfile import write as wav_write
from pydub import AudioSegment
import numpy as np

from state import state 

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR,"outputs")
PRESETS_DIR = os.path.join(BASE_DIR,"speaker_presets")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PRESETS_DIR, exist_ok=True)

# Logger
logger = logging.getLogger("booklet.tts_worker")
logger.setLevel(logging.INFO)
ch = logging.StreamHandler()
formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
ch.setFormatter(formatter)
logger.addHandler(ch)

# Initialize TTS model globally
tts_model = TTSModel.load_model()


def load_voice_state(voice_input: str):
    """
    Load a voice state from:
    - A preset (.pt)
    - An audio file (wav/mp3)
    - A built-in voice name
    Returns a valid voice_state dict or None
    """
    if os.path.isfile(voice_input) and voice_input.endswith(".pt"):
        try:
            # Always load to CPU for compatibility
            voice_state = torch.load(voice_input, map_location="cpu")
            logger.info(f"Loaded preset voice: {voice_input}")
            return voice_state
        except Exception as e:
            logger.error(f"Failed to load preset {voice_input}: {e}")
            return None
    else:
        # Treat as audio prompt or built-in voice
        try:
            voice_state = tts_model.get_state_for_audio_prompt(voice_input)
            logger.info(f"Generated voice state from audio/built-in: {voice_input}")
            return voice_state
        except Exception as e:
            logger.error(f"Failed to generate voice state for {voice_input}: {e}")
            return None

def create_preset_from_audio(voice_path: str, preset_name: str):
    """
    Generate a voice preset from an uploaded audio file.
    Saves it to the PRESETS_DIR and returns the voice_state.
    """
    preset_path = os.path.join(PRESETS_DIR, preset_name)
    try:
        voice_state = tts_model.get_state_for_audio_prompt(voice_path)
        # Save on CPU for portability
        voice_state_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in voice_state.items()}
        torch.save(voice_state_cpu, preset_path)
        logger.info(f"Created new preset: {preset_path}")
        return voice_state
    except Exception as e:
        logger.error(f"Failed to create preset from {voice_path}: {e}")
        return None

"""def save_audio_mp3(audio_tensor, filename: str):
    Save torch audio tensor to MP3 using pydub
    audio_np = (audio_tensor.numpy() * 32767).astype(np.int16)
    temp_wav = filename.replace(".mp3", "_temp.wav")
    wav_write(temp_wav, tts_model.sample_rate, audio_np)

    sound = AudioSegment.from_wav(temp_wav)
    sound.export(filename, format="mp3")
    os.remove(temp_wav)
    logger.info(f"Saved audio to {filename}")"""


def save_audio_wav(audio_tensor, filename: str):
    """
    Save torch audio tensor to WAV
    """
    # Ensure tensor is 1D (mono) or 2D (channels x samples)
    audio_np = audio_tensor.squeeze().numpy()
    
    # Convert from float [-1,1] to int16
    audio_np_int16 = (audio_np * 32767).astype(np.int16)
    
    # Write WAV
    wav_write(filename, tts_model.sample_rate, audio_np_int16)
    
    logger.info(f"Saved WAV audio to {filename}")


def run_tts(book_path: str, voice_input: str, output_file: str, chunk_size: int = 1000, run_id: Optional[str] = None):
    """
    TTS worker:
    - Saves each chunk immediately to individual MP3 files in /outputs/<run_id>/
    - Updates progress via state
    - Keeps all original pause/resume/stop functionality
    """
    run_id = run_id or "manual"
    chunk_size = int(chunk_size)
    logger.info(f"[{run_id}] Starting TTS | Book: {book_path} | Voice: {voice_input} | Output: {output_file} | Chunk size: {chunk_size}")

    # Load voice state
    voice_state = load_voice_state(voice_input)
    if not voice_state:
        logger.error(f"[{run_id}] Could not load a valid voice state, aborting TTS.")
        return

    if not os.path.isfile(book_path):
        logger.error(f"[{run_id}] Book path does not exist: {book_path}")
        return

    with open(book_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split text into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

    # Create run-specific output folder
    run_output_dir = os.path.join(OUTPUTS_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    # Initialize state
    state.lock.acquire()
    state.current_idx = 0
    state.total_chunks = len(chunks)
    state.eta_seconds = 0
    state.lock.release()

    start_time = time.time()

    for idx, chunk in enumerate(chunks, start=1):
        # Handle pause/resume/stop
        while getattr(state, "paused", False):
            time.sleep(0.1)
        if getattr(state, "stopped", False):
            logger.info(f"[{run_id}] TTS stopped by user.")
            return

        try:
            audio = tts_model.generate_audio(voice_state, chunk)
        except Exception as e:
            logger.error(f"[{run_id}] Error generating audio for chunk {idx}: {e}")
            continue

        # Save chunk immediately
        chunk_file = os.path.join(run_output_dir, f"chunk_{idx:03d}.wav")
        save_audio_wav(audio, chunk_file)
        logger.info(f"[{run_id}] Saved chunk {idx} -> {chunk_file}")

        # Update progress
        elapsed = time.time() - start_time
        remaining = (elapsed / idx) * (len(chunks) - idx)
        state.lock.acquire()
        state.current_idx = idx
        state.eta_seconds = int(remaining)
        state.lock.release()

        logger.info(f"[{run_id}] Chunk {idx}/{len(chunks)} complete | Elapsed: {elapsed:.1f}s | ETA: {remaining:.1f}s")

    # Optional: concatenate all chunks into one final MP3
    try:
        final_audio_path = output_file
        combined_audio = None
        for idx in range(1, len(chunks)+1):
            chunk_file = os.path.join(run_output_dir, f"chunk_{idx:03d}.mp3")
            segment = AudioSegment.from_mp3(chunk_file)
            if combined_audio is None:
                combined_audio = segment
            else:
                combined_audio += segment
        combined_audio.export(final_audio_path, format="mp3")
        logger.info(f"[{run_id}] Final MP3 saved: {final_audio_path}")
    except Exception as e:
        logger.error(f"[{run_id}] Failed to combine chunks into final MP3: {e}")

    # Finalize state
    state.lock.acquire()
    state.current_idx = state.total_chunks
    state.eta_seconds = 0
    state.lock.release()

    logger.info(f"[{run_id}] TTS complete!")
