import os
import time
import logging
import threading
import queue
from typing import Optional
import torch
import gc
from scipy.io.wavfile import write as wav_write
from pydub import AudioSegment
import numpy as np
from faster_whisper import WhisperModel

from state import state
from parser import load_text, chunk_text
from zipvoice.luxvoice import LuxTTS

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
# VRAM Monitoring
# =====================================================
def log_vram(label=""):
    """Log VRAM usage with timestamp"""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**3
        reserved = torch.cuda.memory_reserved() / 1024**3
        free = (torch.cuda.get_device_properties(0).total_memory / 1024**3) - allocated
        logger.info(f"🔍 VRAM [{label}]: Allocated={allocated:.2f}GB | Reserved={reserved:.2f}GB | Free={free:.2f}GB")

def aggressive_cleanup():
    """Aggressive VRAM cleanup"""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

# =====================================================
# Transcription
# =====================================================
_faster_whisper_model = None
_whisper_lock = threading.Lock()

def get_faster_whisper_model():
    """Get or load faster-whisper model for transcription"""
    global _faster_whisper_model
    with _whisper_lock:
        if _faster_whisper_model is None:
            logger.info("Loading faster-whisper model for transcription...")
            log_vram("BEFORE loading faster-whisper")
            
            # Load faster-whisper model with GPU acceleration
            # Using base model with float16 for best speed/accuracy balance
            _faster_whisper_model = WhisperModel(
                "base",
                device="cuda",
                compute_type="float16"  # Use fp16 for speed on GPU
            )
            
            log_vram("AFTER loading faster-whisper")
            logger.info("✅ faster-whisper model loaded")
        return _faster_whisper_model

def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio file to text using faster-whisper"""
    logger.info(f"Transcribing audio with faster-whisper: {audio_path}")
    log_vram("BEFORE transcription")
    
    # Load model
    model = get_faster_whisper_model()
    
    # Transcribe with faster-whisper
    # beam_size=5 is good balance of speed and accuracy
    # vad_filter=True improves quality by filtering silence
    segments, info = model.transcribe(
        audio_path,
        beam_size=5,
        vad_filter=True,
        vad_parameters=dict(min_silence_duration_ms=500)
    )
    
    # Extract text from segments
    transcription = " ".join([segment.text for segment in segments]).strip()
    
    # Cleanup
    aggressive_cleanup()
    log_vram("AFTER transcription")
    
    logger.info(f"Transcribed ({info.language}, {info.language_probability:.2f} confidence): '{transcription}'")
    return transcription

def cleanup_faster_whisper():
    """Clean up faster-whisper model"""
    global _faster_whisper_model
    with _whisper_lock:
        if _faster_whisper_model is not None:
            logger.info("Cleaning up faster-whisper model...")
            del _faster_whisper_model
            _faster_whisper_model = None
            aggressive_cleanup()

# =====================================================
# Audio Saving
# =====================================================
def save_audio_wav(audio_tensor, filename: str):
    # Move to CPU immediately and cleanup
    audio_np = audio_tensor.squeeze().cpu().numpy()
    del audio_tensor
    aggressive_cleanup()
    
    audio_np_int16 = (audio_np * 32767).astype(np.int16)
    wav_write(filename, 24000, audio_np_int16)
    del audio_np, audio_np_int16

def save_audio_mp3(audio_tensor, filename: str):
    # Move to CPU immediately and cleanup
    audio_np = audio_tensor.squeeze().cpu().numpy()
    del audio_tensor
    aggressive_cleanup()
    log_vram("After audio->CPU in save_audio_mp3")
    
    audio_np_int16 = (audio_np * 32767).astype(np.int16)

    segment = AudioSegment(
        audio_np_int16.tobytes(),
        frame_rate=24000,
        sample_width=2,
        channels=1
    )

    segment.export(filename, format="mp3", bitrate="192k")
    del audio_np, audio_np_int16, segment


def create_preset_from_audio(audio_path: str, preset_name: str):
    """Create a speaker preset from an audio file using LuxTTS."""
    preset_path = os.path.join(PRESETS_DIR, preset_name)
    
    logger.info(f"{'='*60}")
    logger.info(f"🎙️  Creating preset from audio: {audio_path}")
    logger.info(f"{'='*60}")
    
    # Verify CUDA is available
    if not torch.cuda.is_available():
        logger.error("⚠️  CUDA not available! Preset creation will be slow on CPU.")
    else:
        logger.info(f"✅ Using GPU: {torch.cuda.get_device_name(0)}")
    
    log_vram("BEFORE preset creation")
    
    # STEP 1: Transcribe the audio to get what's actually being said
    transcription = transcribe_audio(audio_path)
    logger.info(f"Using transcription: '{transcription}'")
    
    # STEP 2: Load LuxTTS temporarily to encode the voice
    logger.info("Loading LuxTTS model for encoding on GPU...")
    model = LuxTTS(device='cuda')  # Force CUDA
    log_vram("AFTER loading LuxTTS for preset")
    
    # STEP 3: Encode the prompt audio with the correct transcription
    logger.info("Encoding voice prompt...")
    encode_dict = model.encode_prompt(
        prompt_text=transcription,  # Use actual transcription!
        prompt_audio=audio_path,
        duration=4,
        rms=0.1
    )
    log_vram("AFTER encode_prompt()")
    
    # STEP 4: Move all tensors to CPU before saving
    logger.info("Moving tensors to CPU...")
    encode_dict_cpu = {}
    for key, value in encode_dict.items():
        if isinstance(value, torch.Tensor):
            encode_dict_cpu[key] = value.cpu()
            del value
        else:
            encode_dict_cpu[key] = value
    
    # Also save the transcription for reference
    encode_dict_cpu['_transcription'] = transcription
    
    log_vram("AFTER moving tensors to CPU")
    
    # STEP 5: Save to disk
    logger.info(f"Saving preset to: {preset_path}")
    torch.save(encode_dict_cpu, preset_path)
    
    # STEP 6: Aggressive cleanup
    logger.info("Cleaning up temporary model...")
    del encode_dict, encode_dict_cpu, model
    aggressive_cleanup()
    
    log_vram("AFTER preset cleanup")
    logger.info(f"✅ Preset saved successfully!")
    logger.info(f"{'='*60}")
    return preset_path

# =====================================================
# Shared Model Singleton (CRITICAL FIX)
# =====================================================
_shared_model = None
_model_lock = threading.Lock()

def get_shared_model():
    """Get or create shared LuxTTS model (one instance for all workers)"""
    global _shared_model
    with _model_lock:
        if _shared_model is None:
            # Verify CUDA is available
            if not torch.cuda.is_available():
                logger.error("⚠️  CUDA not available! This will be very slow on CPU.")
                logger.error("⚠️  Please ensure you have a CUDA-capable GPU and PyTorch with CUDA support.")
            else:
                logger.info(f"✅ CUDA available: {torch.cuda.get_device_name(0)}")
                logger.info(f"✅ CUDA version: {torch.version.cuda}")
            
            logger.info("🔄 Loading shared LuxTTS model on GPU...")
            log_vram("BEFORE loading LuxTTS model")
            _shared_model = LuxTTS(device='cuda')  # Force CUDA
            log_vram("AFTER loading LuxTTS model")
            logger.info("✅ Shared LuxTTS model loaded on GPU")
        return _shared_model

def cleanup_shared_model():
    """Cleanup the shared model"""
    global _shared_model
    with _model_lock:
        if _shared_model is not None:
            logger.info("🧹 Cleaning up shared model...")
            log_vram("BEFORE cleanup_shared_model()")
            del _shared_model
            _shared_model = None
            aggressive_cleanup()
            log_vram("AFTER cleanup_shared_model()")
            logger.info("✅ Shared model cleaned up")

# =====================================================
# Worker Thread (Shares One Model)
# =====================================================
class ChunkProcessor(threading.Thread):
    def __init__(self, chunk_queue, result_queue, voice_input, run_id, run_output_dir, worker_id):
        super().__init__(daemon=True)
        self.chunk_queue = chunk_queue
        self.result_queue = result_queue
        self.voice_input = voice_input
        self.run_id = run_id
        self.run_output_dir = run_output_dir
        self.worker_id = worker_id
        self.should_stop = False
        
        log_vram(f"Worker {worker_id} START initialization")
        
        # Use shared model instead of creating new one
        logger.info(f"[{run_id}] Worker {worker_id}: using shared model...")
        self.model = get_shared_model()
        log_vram(f"Worker {worker_id} after get_shared_model()")
        
        logger.info(f"[{run_id}] Worker {worker_id}: loading voice...")
        
        # Load encode_dict
        if voice_input.endswith('.safetensors') or voice_input.endswith('.pt'):
            # Load the saved encode_dict from preset
            encode_dict = torch.load(voice_input, map_location='cpu')
            
            # Extract transcription if saved (for logging)
            if '_transcription' in encode_dict:
                transcription = encode_dict['_transcription']
                logger.info(f"[{run_id}] Worker {worker_id}: preset transcription: '{transcription}'")
                # Remove it from encode_dict as it's not needed for generation
                del encode_dict['_transcription']
            
            logger.info(f"[{run_id}] Worker {worker_id}: loaded preset from {voice_input}")
        else:
            # Encode directly from raw audio file - need to transcribe it first
            logger.info(f"[{run_id}] Worker {worker_id}: transcribing raw audio file...")
            transcription = transcribe_audio(voice_input)
            logger.info(f"[{run_id}] Worker {worker_id}: transcription: '{transcription}'")
            
            # Now encode with the real transcription
            encode_dict = self.model.encode_prompt(
                prompt_text=transcription,  # Use actual transcription!
                prompt_audio=voice_input,
                duration=4,
                rms=0.1
            )
            # Move to CPU immediately
            encode_dict = {k: v.cpu() if isinstance(v, torch.Tensor) else v 
                          for k, v in encode_dict.items()}
            logger.info(f"[{run_id}] Worker {worker_id}: encoded voice from {voice_input}")
        
        log_vram(f"Worker {worker_id} after loading encode_dict")
        
        # CRITICAL: Keep encode_dict on CPU, only move to GPU when needed
        self.encode_dict_cpu = encode_dict
        aggressive_cleanup()
        
        log_vram(f"Worker {worker_id} READY (after cleanup)")
        logger.info(f"[{run_id}] Worker {worker_id} ready")

    def stop(self):
        self.should_stop = True

    def run(self):
        """Process chunks from the queue."""
        chunks_processed = 0
        total_time = 0
        
        logger.info(f"[{self.run_id}] Worker {self.worker_id} starting processing loop")
        log_vram(f"Worker {self.worker_id} BEFORE processing loop")
        
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
                    log_vram(f"Worker {self.worker_id} - Chunk {idx} START")
                    
                    # CRITICAL: Move encode_dict to GPU only for this generation
                    encode_dict_gpu = {
                        k: v.to('cuda') if isinstance(v, torch.Tensor) else v 
                        for k, v in self.encode_dict_cpu.items()
                    }
                    log_vram(f"Worker {self.worker_id} - Chunk {idx} after encode_dict->GPU")
                    
                    # Generate audio
                    # IMPORTANT: speed=0.65 to slow down (0.5 would be half speed, but model multiplies internally)
                    # This will make speech about 2x slower than current output
                    audio = self.model.generate_speech(
                        text=chunk_text,
                        encode_dict=encode_dict_gpu,
                        num_steps=4,
                        guidance_scale=3.0,
                        t_shift=0.5,
                        speed=0.65  # FIXED: Slow down to ~half speed (was 1.0)
                    )
                    log_vram(f"Worker {self.worker_id} - Chunk {idx} after generate_speech()")

                    if audio is None:
                        raise RuntimeError("Model returned None")

                    # CRITICAL: Move to CPU immediately and save
                    chunk_file = os.path.join(
                        self.run_output_dir,
                        f"chunk_{idx:03d}.mp3"
                    )
                    save_audio_mp3(audio, chunk_file)
                    log_vram(f"Worker {self.worker_id} - Chunk {idx} after save_audio_mp3()")
                    
                    # Cleanup GPU tensors
                    del audio, encode_dict_gpu
                    aggressive_cleanup()
                    log_vram(f"Worker {self.worker_id} - Chunk {idx} COMPLETE (after cleanup)")

                    # Calculate timing
                    elapsed = time.time() - start_time
                    chunks_processed += 1
                    total_time += elapsed

                    self.result_queue.put((idx, chunk_file, "success"))
                    
                    # Log every chunk for first 5, then every 5 chunks
                    if chunks_processed <= 5 or chunks_processed % 5 == 0:
                        avg_time = total_time / chunks_processed if chunks_processed > 0 else 0
                        logger.info(f"[{self.run_id}] Worker {self.worker_id} - Chunk {idx} done in {elapsed:.2f}s (avg: {avg_time:.2f}s)")
                        log_vram(f"Worker {self.worker_id} - After {chunks_processed} chunks")

                except Exception as e:
                    logger.error(f"[{self.run_id}] Worker {self.worker_id} error on chunk {idx}: {e}")
                    self.result_queue.put((idx, None, f"error: {e}"))
                    aggressive_cleanup()
                    log_vram(f"Worker {self.worker_id} - Chunk {idx} ERROR (after cleanup)")

            except Exception as e:
                logger.error(f"[{self.run_id}] Worker {self.worker_id} crash: {e}", exc_info=True)
                aggressive_cleanup()
                log_vram(f"Worker {self.worker_id} - CRASH (after cleanup)")
        
        # Cleanup when done
        logger.info(f"[{self.run_id}] Worker {self.worker_id} finished processing")
        log_vram(f"Worker {self.worker_id} - BEFORE final cleanup")
        del self.encode_dict_cpu
        aggressive_cleanup()
        log_vram(f"Worker {self.worker_id} - AFTER final cleanup")
        logger.info(f"[{self.run_id}] Worker {self.worker_id} finished")

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

    logger.info(f"{'='*80}")
    logger.info(f"[{run_id}] 🚀 Starting PARALLEL TTS with LuxTTS")
    logger.info(f"{'='*80}")
    log_vram("START of run_tts")

    if not os.path.isfile(book_path):
        logger.error(f"[{run_id}] Book file not found")
        return

    if not os.path.isfile(voice_input):
        logger.error(f"[{run_id}] Voice input not found: {voice_input}")
        return

    # Load text
    logger.info(f"[{run_id}] Loading text from book...")
    text = load_text(book_path)
    if not text:
        logger.error(f"[{run_id}] Failed to load text from book")
        return

    logger.info(f"[{run_id}] Creating text chunks...")
    chunks = list(chunk_text(text, chunk_size))
    if not chunks:
        logger.error(f"[{run_id}] No chunks generated")
        return

    # FIXED: Use only 1 worker by default to minimize VRAM
    # All workers share the same model instance anyway
    if num_threads is None:
        num_workers = 1  # Single worker uses ~6-8GB VRAM total
    else:
        # Allow 2-3 workers max for throughput
        num_workers = min(3, max(1, int(num_threads)))

    logger.info(f"[{run_id}] Processing {len(chunks)} chunks with {num_workers} GPU workers (shared model)")
    logger.info(f"[{run_id}] Chunk size: {chunk_size} characters")
    log_vram("After loading text, before workers")

    run_output_dir = os.path.join(OUTPUTS_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    state.start(len(chunks))

    chunk_queue = queue.Queue()
    result_queue = queue.Queue()

    for idx, chunk in enumerate(chunks, start=1):
        chunk_queue.put((idx, chunk))

    for _ in range(num_workers):
        chunk_queue.put(None)

    # Start workers - they all share the same model
    logger.info(f"[{run_id}] Starting {num_workers} worker(s)...")
    log_vram("BEFORE starting workers")
    
    workers = []
    for i in range(num_workers):
        worker = ChunkProcessor(
            chunk_queue,
            result_queue,
            voice_input,
            run_id,
            run_output_dir,
            worker_id=i
        )
        worker.start()
        workers.append(worker)
    
    log_vram("AFTER starting workers")
    logger.info(f"[{run_id}] Workers started, beginning processing...")

    # Process results
    completed = 0
    failed = 0
    last_log_time = time.time()
    
    while completed + failed < len(chunks):
        if state.is_stopped():
            logger.info(f"[{run_id}] ⚠️  Stopping requested")
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
            
            # Log progress every 10 seconds or every 20 chunks
            current_time = time.time()
            if completed % 20 == 0 or (current_time - last_log_time) > 10:
                progress_pct = (completed + failed) / len(chunks) * 100
                logger.info(f"[{run_id}] 📊 Progress: {completed}/{len(chunks)} ({progress_pct:.1f}%) | Failed: {failed}")
                log_vram(f"After {completed} chunks completed")
                last_log_time = current_time

        except queue.Empty:
            if not any(w.is_alive() for w in workers):
                break
            continue

    # Stop and cleanup workers
    logger.info(f"[{run_id}] Stopping workers...")
    log_vram("BEFORE stopping workers")
    
    for worker in workers:
        worker.stop()
    for worker in workers:
        worker.join(timeout=5)
    
    log_vram("AFTER workers stopped")
    
    # Cleanup shared model
    logger.info(f"[{run_id}] Cleaning up shared model...")
    cleanup_shared_model()
    log_vram("AFTER cleanup_shared_model()")
    
    # Cleanup faster-whisper if it was loaded
    cleanup_faster_whisper()
    log_vram("AFTER cleanup_faster_whisper()")

    # Combine audio
    logger.info(f"[{run_id}] Combining audio chunks into final MP3...")
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
            logger.info(f"[{run_id}] ✅ Final MP3 saved: {output_file}")
            logger.info(f"[{run_id}] 📈 Success: {success_count}/{len(chunks)} chunks")
        except Exception as e:
            logger.error(f"[{run_id}] Failed to export MP3: {e}")

    state.finish()
    
    # Cleanup
    if os.path.exists(book_path):
        try:
            os.remove(book_path)
        except Exception as e:
            logger.warning(f"[{run_id}] Failed to cleanup {book_path}: {e}")
    
    # Final cleanup
    aggressive_cleanup()
    log_vram("END of run_tts (after final cleanup)")
    
    logger.info(f"{'='*80}")
    logger.info(f"[{run_id}] ✅ Complete - Success: {success_count}, Failed: {failed}")
    logger.info(f"{'='*80}")