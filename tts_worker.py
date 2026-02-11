import os
import time
import logging
import threading
import queue
import psutil
from typing import Optional, List, Tuple
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

# Configuration
RAM_TARGET_GB = float(os.environ.get("TTS_RAM_TARGET_GB", "16"))
MODEL_SIZE_MB = int(os.environ.get("TTS_MODEL_SIZE_MB", "450"))

class ModelPool:
    """Pool of TTS model instances for parallel chunk processing"""
    
    def __init__(self, ram_target_gb: float = 16, model_size_mb: int = 450):
        self.ram_target_gb = ram_target_gb
        self.model_size_mb = model_size_mb
        self.models: List[TTSModel] = []
        self.model_queue = queue.Queue()
        self.num_models = self._calculate_num_models()
        
        logger.info(f"Initializing model pool with RAM target: {ram_target_gb}GB")
        logger.info(f"Estimated model size: {model_size_mb}MB")
        logger.info(f"Planning to load {self.num_models} model instances")
        
        self._load_models()
        
    def _calculate_num_models(self) -> int:
        """Calculate how many model instances can fit in target RAM"""
        mem = psutil.virtual_memory()
        available_gb = mem.available / (1024**3)
        
        logger.info(f"System RAM - Total: {mem.total / (1024**3):.1f}GB, Available: {available_gb:.1f}GB")
        
        # Calculate based on target
        ram_budget_mb = self.ram_target_gb * 1024
        num_models = max(1, int(ram_budget_mb / self.model_size_mb))
        
        # Cap at reasonable maximum
        num_models = min(num_models, 16)
        
        # Ensure we don't exceed 80% of available RAM
        total_model_size_gb = (num_models * self.model_size_mb) / 1024
        if total_model_size_gb > available_gb * 0.8:
            num_models = max(1, int((available_gb * 0.8 * 1024) / self.model_size_mb))
            logger.warning(f"Reduced to {num_models} models to fit available RAM")
        
        return num_models
    
    def _load_models(self):
        """Load all model instances, each with single thread"""
        for i in range(self.num_models):
            logger.info(f"Loading model {i+1}/{self.num_models}...")
            try:
                model = TTSModel.load_model(num_threads=1)
                self.models.append(model)
                self.model_queue.put(i)
                logger.info(f"✓ Model {i+1} loaded")
            except Exception as e:
                logger.error(f"✗ Failed to load model {i+1}: {e}")
                if i == 0:
                    raise
                break
        
        logger.info(f"Model pool ready: {len(self.models)} instances")
        
        # Report memory usage
        mem = psutil.virtual_memory()
        logger.info(f"RAM after init - Used: {mem.used / (1024**3):.1f}GB, Available: {mem.available / (1024**3):.1f}GB")
    
    def acquire(self, timeout: float = None) -> Tuple[int, TTSModel]:
        """Acquire an available model from pool"""
        try:
            idx = self.model_queue.get(timeout=timeout)
            return idx, self.models[idx]
        except queue.Empty:
            raise TimeoutError("No models available")
    
    def release(self, model_idx: int):
        """Return model to pool"""
        self.model_queue.put(model_idx)
    
    def size(self) -> int:
        """Total models in pool"""
        return len(self.models)
    
    def available(self) -> int:
        """Currently available models"""
        return self.model_queue.qsize()

# Initialize global model pool
logger.info("="*60)
logger.info("INITIALIZING TTS MODEL POOL")
logger.info(f"RAM Target: {RAM_TARGET_GB}GB | Model Size: {MODEL_SIZE_MB}MB")
logger.info("="*60)

model_pool = ModelPool(ram_target_gb=RAM_TARGET_GB, model_size_mb=MODEL_SIZE_MB)

logger.info("="*60)
logger.info(f"✓ Model pool ready: {model_pool.size()} instances")
logger.info("="*60)


def load_voice_state(voice_input: str, model: TTSModel):
    """Load voice state from preset, audio file, or built-in voice"""
    if os.path.isfile(voice_input) and voice_input.endswith(".pt"):
        try:
            voice_state = torch.load(voice_input, map_location="cpu")
            logger.info(f"Loaded preset: {voice_input}")
            return voice_state
        except Exception as e:
            logger.error(f"Failed to load preset {voice_input}: {e}")
            return None
    else:
        try:
            voice_state = model.get_state_for_audio_prompt(voice_input)
            logger.info(f"Generated voice state from: {voice_input}")
            return voice_state
        except Exception as e:
            logger.error(f"Failed to generate voice state for {voice_input}: {e}")
            return None

def create_preset_from_audio(voice_path: str, preset_name: str):
    """Generate a voice preset from uploaded audio file"""
    preset_path = os.path.join(PRESETS_DIR, preset_name)
    try:
        # Acquire a model temporarily to generate the preset
        model_idx, model = model_pool.acquire(timeout=10)
        try:
            voice_state = model.get_state_for_audio_prompt(voice_path)
            voice_state_cpu = {k: v.cpu() if isinstance(v, torch.Tensor) else v for k, v in voice_state.items()}
            torch.save(voice_state_cpu, preset_path)
            logger.info(f"Created preset: {preset_path}")
            return voice_state
        finally:
            model_pool.release(model_idx)
    except Exception as e:
        logger.error(f"Failed to create preset from {voice_path}: {e}")
        return None

def save_audio_wav(audio_tensor, filename: str):
    """Save torch audio tensor to WAV"""
    audio_np = audio_tensor.squeeze().numpy()
    audio_np_int16 = (audio_np * 32767).astype(np.int16)
    wav_write(filename, 24000, audio_np_int16)  # Assuming 24kHz sample rate
    logger.debug(f"Saved WAV: {filename}")


class ChunkProcessor(threading.Thread):
    """Worker thread that processes chunks using a model from the pool"""
    
    def __init__(self, chunk_queue, result_queue, voice_states, run_id, run_output_dir):
        super().__init__(daemon=True)
        self.chunk_queue = chunk_queue
        self.result_queue = result_queue
        self.voice_states = voice_states  # Dict mapping model_idx -> voice_state
        self.run_id = run_id
        self.run_output_dir = run_output_dir
        self.should_stop = False
    
    def stop(self):
        self.should_stop = True
    
    def run(self):
        """Process chunks from queue using models from pool"""
        while not self.should_stop:
            try:
                # Get next chunk to process
                chunk_data = self.chunk_queue.get(timeout=0.5)
                if chunk_data is None:  # Poison pill
                    break
                
                idx, chunk_text = chunk_data
                
                # Acquire a model from the pool
                try:
                    model_idx, model = model_pool.acquire(timeout=10)
                except TimeoutError:
                    logger.warning(f"[{self.run_id}] Timeout acquiring model for chunk {idx}")
                    self.result_queue.put((idx, None, "timeout"))
                    continue
                
                try:
                    # Use the voice state specific to this model instance
                    voice_state = self.voice_states[model_idx]
                    
                    # Generate audio
                    audio = model.generate_audio(voice_state, chunk_text)
                    
                    if audio is None:
                        logger.error(f"[{self.run_id}] Model returned None for chunk {idx}")
                        self.result_queue.put((idx, None, "generation_failed"))
                        continue
                    
                    # Save chunk
                    chunk_file = os.path.join(self.run_output_dir, f"chunk_{idx:03d}.wav")
                    save_audio_wav(audio, chunk_file)
                    
                    self.result_queue.put((idx, chunk_file, "success"))
                    logger.info(f"[{self.run_id}] Chunk {idx} complete (model {model_idx})")
                    
                except Exception as e:
                    logger.error(f"[{self.run_id}] Error processing chunk {idx}: {e}", exc_info=True)
                    self.result_queue.put((idx, None, f"error: {e}"))
                finally:
                    # Always release model back to pool
                    model_pool.release(model_idx)
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"[{self.run_id}] Worker error: {e}", exc_info=True)


def run_tts(book_path: str, voice_input: str, output_file: str, chunk_size: int = 1000, run_id: Optional[str] = None, num_threads: Optional[int] = None):
    """
    Parallel TTS generation using model pool
    num_threads parameter is ignored - parallelism comes from model pool
    """
    run_id = run_id or "manual"
    chunk_size = int(chunk_size)
    
    logger.info(f"[{run_id}] Starting parallel TTS")
    logger.info(f"[{run_id}] Book: {book_path}")
    logger.info(f"[{run_id}] Voice: {voice_input}")
    logger.info(f"[{run_id}] Chunk size: {chunk_size}")
    logger.info(f"[{run_id}] Model pool size: {model_pool.size()}")

    # Generate voice state for EACH model in the pool
    voice_states = {}
    logger.info(f"[{run_id}] Generating voice states for all models...")
    for i in range(model_pool.size()):
        model_idx, model = model_pool.acquire(timeout=30)
        try:
            voice_state = load_voice_state(voice_input, model)
            if not voice_state:
                logger.error(f"[{run_id}] Could not load voice state for model {model_idx}, aborting")
                return
            voice_states[model_idx] = voice_state
            logger.info(f"[{run_id}] Voice state loaded for model {model_idx}")
        finally:
            model_pool.release(model_idx)

    if not os.path.isfile(book_path):
        logger.error(f"[{run_id}] Book file not found: {book_path}")
        return

    with open(book_path, "r", encoding="utf-8") as f:
        text = f.read()

    # Split into chunks
    chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
    logger.info(f"[{run_id}] Split into {len(chunks)} chunks")

    # Create output directory
    run_output_dir = os.path.join(OUTPUTS_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    # Initialize state
    state.lock.acquire()
    state.current_idx = 0
    state.total_chunks = len(chunks)
    state.eta_seconds = 0
    state.lock.release()

    # Create queues
    chunk_queue = queue.Queue()
    result_queue = queue.Queue()

    # Populate chunk queue
    for idx, chunk in enumerate(chunks, start=1):
        chunk_queue.put((idx, chunk))

    # Start worker threads (one per model in pool)
    num_workers = model_pool.size()
    workers = []
    for _ in range(num_workers):
        worker = ChunkProcessor(chunk_queue, result_queue, voice_states, run_id, run_output_dir)
        worker.start()
        workers.append(worker)
    
    logger.info(f"[{run_id}] Started {num_workers} worker threads")

    # Monitor progress
    start_time = time.time()
    completed = 0
    failed = 0

    while completed + failed < len(chunks):
        # Check for pause/stop
        if getattr(state, "stopped", False):
            logger.info(f"[{run_id}] Stopping workers...")
            for worker in workers:
                worker.stop()
            break
        
        while getattr(state, "paused", False):
            time.sleep(0.1)
        
        try:
            idx, chunk_file, status = result_queue.get(timeout=1)
            
            if status == "success":
                completed += 1
            else:
                failed += 1
                logger.warning(f"[{run_id}] Chunk {idx} failed: {status}")
            
            # Update progress
            elapsed = time.time() - start_time
            if completed > 0:
                remaining = (elapsed / completed) * (len(chunks) - completed)
            else:
                remaining = 0
            
            state.lock.acquire()
            state.current_idx = completed
            state.eta_seconds = int(remaining)
            state.lock.release()
            
            logger.info(f"[{run_id}] Progress: {completed}/{len(chunks)} | Failed: {failed} | ETA: {remaining:.1f}s")
            
        except queue.Empty:
            continue

    # Wait for workers to finish
    for worker in workers:
        worker.stop()
    for worker in workers:
        worker.join(timeout=5)

    logger.info(f"[{run_id}] Generation complete - {completed} succeeded, {failed} failed")

    # Combine chunks into final MP3
    if completed > 0:
        try:
            combined_audio = None
            for idx in range(1, len(chunks) + 1):
                chunk_file = os.path.join(run_output_dir, f"chunk_{idx:03d}.wav")
                if not os.path.exists(chunk_file):
                    logger.warning(f"[{run_id}] Missing chunk {idx}, skipping")
                    continue
                
                segment = AudioSegment.from_wav(chunk_file)
                if combined_audio is None:
                    combined_audio = segment
                else:
                    combined_audio += segment
            
            if combined_audio:
                combined_audio.export(output_file, format="mp3")
                logger.info(f"[{run_id}] Final MP3 saved: {output_file}")
        except Exception as e:
            logger.error(f"[{run_id}] Failed to combine chunks: {e}", exc_info=True)

    # Finalize state
    state.lock.acquire()
    state.current_idx = state.total_chunks
    state.eta_seconds = 0
    state.lock.release()

    logger.info(f"[{run_id}] TTS job complete!")