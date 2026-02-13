import os
import logging
import gc
from typing import Optional

import torch
import numpy as np
from scipy.io.wavfile import write, read
import soundfile as sf

from state import state
from parser import load_text, chunk_text

# =====================================================
# Paths
# =====================================================

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)

logger = logging.getLogger("booklet.tts")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)


# =====================================================
# Cleanup
# =====================================================

def aggressive_cleanup():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()


# =====================================================
# Main TTS
# =====================================================

def run_tts(
    book_path: str,
    voice_path: str,
    transcription: str,
    output_file: str,
    chunk_size: int = 800,
    run_id: Optional[str] = None,
    speed: float = 0.9,
):
    run_id = run_id or "manual"

    # Load and chunk text
    text = load_text(book_path)
    chunks = list(chunk_text(text, chunk_size))

    if not chunks:
        logger.error("No text chunks generated")
        state.finish()
        return

    # Load model once
    from zipvoice.luxvoice import LuxTTS

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = LuxTTS(device=device)
    logger.info(f"Loaded LuxTTS on {device}")

    # Encode voice once
    logger.info(f"Encoding voice prompt: {voice_path}")
    encode_dict = model.encode_prompt(
        prompt_text=transcription,
        prompt_audio=voice_path,
        duration=10,  # Increased from 3 to capture more voice characteristics
        rms=0.15,     # Increased from 0.01 to strengthen voice conditioning
    )

    # Move encode dict to correct device once
    encode_dict = {
        k: v.to(device) if isinstance(v, torch.Tensor) else v
        for k, v in encode_dict.items()
    }

    run_output_dir = os.path.join(OUTPUTS_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    state.start(len(chunks))

    # =====================================================
    # Generate chunks sequentially
    # =====================================================

    for idx, chunk in enumerate(chunks, start=1):
        try:
            logger.info(f"Generating chunk {idx}/{len(chunks)}")

            audio = model.generate_speech(
                text=chunk,
                encode_dict=encode_dict,
                num_steps=6,         # Reduced from 8 - fewer steps can help maintain prompt
                guidance_scale=5.0,  # Increased from 3.0 - stronger conditioning to voice prompt
                t_shift=0.5,
                speed=speed,
            )

            if audio is None:
                raise RuntimeError("Model returned None")

            audio = audio.numpy().squeeze()

            chunk_file = os.path.join(run_output_dir, f"chunk_{idx:04d}.wav")
            sf.write(chunk_file, audio, 48000)

            state.increment_completed()
            #aggressive_cleanup()

        except Exception as e:
            logger.error(f"Chunk {idx} failed: {e}")
            state.increment_failed()
            #aggressive_cleanup()

    # =====================================================
    # Combine chunks
    # =====================================================

    logger.info("Combining chunks...")
    combined_audio = []

    for idx in range(1, len(chunks) + 1):
        chunk_file = os.path.join(run_output_dir, f"chunk_{idx:04d}.wav")
        if os.path.exists(chunk_file):
            sr, data = read(chunk_file)
            combined_audio.append(data)

    if combined_audio:
        final_audio = np.concatenate(combined_audio)
        write(output_file, 24000, final_audio)
        logger.info(f"✅ Final audio saved: {output_file}")
    else:
        logger.error("No audio chunks to combine")

    # Cleanup
    del model
    aggressive_cleanup()
    state.finish()

    # Remove temp input files
    for f in [book_path, voice_path]:
        if os.path.exists(f):
            os.remove(f)