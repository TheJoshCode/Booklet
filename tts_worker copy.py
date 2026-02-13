import os
import time
import logging
import threading
import queue
from typing import Optional
import torch
from pocket_tts import TTSModel
from scipy.io.wavfile import write as wav_write
from pydub import AudioSegment
import numpy as np

from state import state
from parser import load_text, chunk_text

# =====================================================
# Paths
# =====================================================
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PRESETS_DIR = os.path.join(BASE_DIR, "speaker_presets")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PRESETS_DIR, exist_ok=True)

# =====================================================
# Logging
# =====================================================
logger = logging.getLogger("booklet.tts_worker")
logger.setLevel(logging.INFO)

if not logger.handlers:
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

# =====================================================
# Audio Saving
# =====================================================
def save_audio_wav(audio_tensor, filename: str):
    audio_np = audio_tensor.squeeze().cpu().numpy()
    audio_np_int16 = (audio_np * 32767).astype(np.int16)
    wav_write(filename, 24000, audio_np_int16)

def save_audio_mp3(audio_tensor, filename: str):
    audio_np = audio_tensor.squeeze().cpu().numpy()
    audio_np_int16 = (audio_np * 32767).astype(np.int16)

    segment = AudioSegment(
        audio_np_int16.tobytes(),
        frame_rate=24000,
        sample_width=2,  # int16 = 2 bytes
        channels=1
    )

    segment.export(filename, format="mp3", bitrate="192k")


def create_preset_from_audio(audio_path: str, preset_name: str):
    """Create a speaker preset from an audio file."""
    preset_path = os.path.join(PRESETS_DIR, preset_name)
    
    logger.info(f"Creating preset from audio: {audio_path}")
    
    # Load a temporary model just to encode the audio
    model = TTSModel.load_model(num_threads=1)
    
    # Use save_audio_prompt to create the safetensors file
    model.save_audio_prompt(audio_path, preset_path)
    
    logger.info(f"Preset saved to: {preset_path}")
    return preset_path

# =====================================================
# Worker Thread (One Model Per Worker)
# =====================================================
class ChunkProcessor(threading.Thread):
    def __init__(self, chunk_queue, result_queue, voice_input, run_id, run_output_dir, worker_id):
        super().__init__(daemon=True)
        self.chunk_queue = chunk_queue
        self.result_queue = result_queue
        self.voice_input = voice_input  # Path to preset or audio file
        self.run_id = run_id
        self.run_output_dir = run_output_dir
        self.worker_id = worker_id
        self.should_stop = False

        logger.info(f"[{run_id}] Worker {worker_id}: loading model...")
        self.model = TTSModel.load_model(num_threads=1)
        
        logger.info(f"[{run_id}] Worker {worker_id}: loading voice...")
        self.voice_state = self.model.get_state_for_audio_prompt(voice_input)
        
        logger.info(f"[{run_id}] Worker {worker_id} ready")

    def stop(self):
        self.should_stop = True

    def run(self):
        """Process chunks from the queue."""
        chunks_processed = 0
        total_time = 0
        
        while not self.should_stop:
            try:
                if state.is_paused():
                    time.sleep(0.1)
                    continue
                
                try:
                    chunk_data = self.chunk_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if chunk_data is None:
                    break

                idx, chunk_text = chunk_data
                start_time = time.time()

                try:
                    # Each worker uses its own model and voice state
                    audio = self.model.generate_audio(
                        self.voice_state,
                        chunk_text,
                        copy_state=True
                    )

                    if audio is None:
                        raise RuntimeError("Model returned None")

                    chunk_file = os.path.join(
                        self.run_output_dir,
                        f"chunk_{idx:03d}.mp3"
                    )

                    save_audio_mp3(audio, chunk_file)

                    #save_audio_wav(audio, chunk_file)

                    # Calculate timing
                    elapsed = time.time() - start_time
                    audio_duration = audio.shape[-1] / 24000
                    
                    total_time += elapsed
                    chunks_processed += 1

                    self.result_queue.put((idx, chunk_file, "success"))
                    
                    if chunks_processed % 10 == 0:
                        avg_rtf = (chunks_processed * audio_duration) / total_time if total_time > 0 else 0
                        logger.info(f"[{self.run_id}] Worker {self.worker_id} avg RTF: {avg_rtf:.2f}x")

                except Exception as e:
                    logger.error(f"[{self.run_id}] Worker {self.worker_id} error on chunk {idx}: {e}")
                    self.result_queue.put((idx, None, f"error: {e}"))

            except Exception as e:
                logger.error(f"[{self.run_id}] Worker {self.worker_id} crash: {e}", exc_info=True)

# =====================================================
# Main TTS Runner
# =====================================================
def run_tts(
    book_path: str,
    voice_input: str,
    output_file: str,
    chunk_size: int = 1000,
    run_id: Optional[str] = None,
    num_threads: Optional[int] = None,
):
    run_id = run_id or "manual"
    chunk_size = int(chunk_size)

    logger.info(f"[{run_id}] Starting PARALLEL TTS")

    if not os.path.isfile(book_path):
        logger.error(f"[{run_id}] Book file not found")
        return

    # Validate voice input exists
    if not os.path.isfile(voice_input):
        logger.error(f"[{run_id}] Voice input not found: {voice_input}")
        return

    # Load text
    text = load_text(book_path)
    if not text:
        logger.error(f"[{run_id}] Failed to load text from book")
        return

    chunks = list(chunk_text(text, chunk_size))
    if not chunks:
        logger.error(f"[{run_id}] No chunks generated")
        return

    if num_threads is None:
        cpu_count = os.cpu_count() or 8
        # Use fewer workers, more threads per worker
        num_workers = min(8, max(1, cpu_count // 8))
        threads_per_worker = max(2, cpu_count // num_workers)
    else:
        total_threads = int(num_threads)
        num_workers = min(8, max(1, total_threads // 8))
        threads_per_worker = max(2, total_threads // num_workers)

    logger.info(f"[{run_id}] Processing {len(chunks)} chunks with {num_workers} workers ({threads_per_worker} threads each)")

    run_output_dir = os.path.join(OUTPUTS_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    state.start(len(chunks))

    chunk_queue = queue.Queue()
    result_queue = queue.Queue()

    for idx, chunk in enumerate(chunks, start=1):
        chunk_queue.put((idx, chunk))

    for _ in range(num_workers):
        chunk_queue.put(None)

    # Start workers - each loads its own model and voice state
    workers = []
    for i in range(num_workers):
        worker = ChunkProcessor(
            chunk_queue,
            result_queue,
            voice_input,  # Pass the voice path, each worker loads it
            run_id,
            run_output_dir,
            worker_id=i
        )
        worker.start()
        workers.append(worker)

    # Process results
    completed = 0
    failed = 0
    
    while completed + failed < len(chunks):
        if state.is_stopped():
            logger.info(f"[{run_id}] Stopping requested")
            break

        while state.is_paused() and not state.is_stopped():
            time.sleep(0.1)

        try:
            idx, chunk_file, status = result_queue.get(timeout=0.5)

            if status == "success":
                state.increment_completed()
                completed += 1
            else:
                state.increment_failed()
                failed += 1

        except queue.Empty:
            if not any(w.is_alive() for w in workers):
                break
            continue

    for worker in workers:
        worker.stop()
    for worker in workers:
        worker.join(timeout=5)

    # Combine audio
    combined_audio = None
    success_count = 0
    
    for idx in range(1, len(chunks) + 1):
        chunk_file = os.path.join(run_output_dir, f"chunk_{idx:03d}.mp3")
        if not os.path.exists(chunk_file):
            continue

        try:
            segment = AudioSegment.from_mp3(chunk_file)
            combined_audio = segment if combined_audio is None else combined_audio + segment
            success_count += 1
        except Exception as e:
            logger.error(f"[{run_id}] Error reading chunk {idx}: {e}")

    if combined_audio and success_count > 0:
        try:
            combined_audio.export(output_file, format="mp3")
            logger.info(f"[{run_id}] Final MP3 saved: {output_file} ({success_count}/{len(chunks)} chunks)")
        except Exception as e:
            logger.error(f"[{run_id}] Failed to export MP3: {e}")

    state.finish()
    logger.info(f"[{run_id}] Complete - Success: {success_count}, Failed: {failed}")
    
    if os.path.exists(book_path):
        try:
            os.remove(book_path)
        except Exception as e:
            logger.warning(f"[{run_id}] Failed to cleanup {book_path}: {e}")