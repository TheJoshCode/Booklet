import os
import time
import logging
import threading
import queue
from typing import Optional
import torch
from pocket_tts import TTSModel
from pocket_tts.modules.stateful_module import init_states  # Import this
from scipy.io.wavfile import write as wav_write
from pydub import AudioSegment
import numpy as np
import copy

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

# =====================================================
# Voice Handling
# =====================================================
def load_voice_conditioning(voice_input: str, model: TTSModel):
    """
    Load voice conditioning (speaker embedding) from preset file or audio file.
    Returns the audio conditioning tensor, NOT the model state.
    """
    if os.path.isfile(voice_input) and voice_input.endswith(".pt"):
        try:
            # Load the saved conditioning tensor
            data = torch.load(voice_input, map_location="cpu")
            if isinstance(data, dict) and 'conditioning' in data:
                conditioning = data['conditioning']
            elif isinstance(data, torch.Tensor):
                conditioning = data
            else:
                # Fallback: try to extract from old format
                conditioning = data
            logger.info(f"Loaded preset: {voice_input}")
            return conditioning
        except Exception as e:
            logger.error(f"Failed to load preset {voice_input}: {e}")
            return None
    else:
        try:
            # Generate conditioning from audio file
            if isinstance(voice_input, str):
                from pocket_tts.utils.utils import download_if_necessary
                from pocket_tts.data.audio import audio_read
                from pocket_tts.data.audio_utils import convert_audio
                
                voice_input = download_if_necessary(voice_input)
                audio, sample_rate = audio_read(voice_input)
                audio = convert_audio(audio, sample_rate, model.config.mimi.sample_rate, 1)
                audio = audio.unsqueeze(0).to(model.device)
            
            # Encode to get conditioning
            with torch.no_grad():
                encoded = model.mimi.encode_to_latent(audio)
                latents = encoded.transpose(-1, -2).to(torch.float32)
                conditioning = torch.nn.functional.linear(latents, model.flow_lm.speaker_proj_weight)
            
            logger.info(f"Generated voice conditioning from: {voice_input}")
            return conditioning
        except Exception as e:
            logger.error(f"Failed to generate voice conditioning: {e}")
            return None


def create_preset_from_audio(voice_path: str, preset_name: str):
    """Create a preset file from an audio file - saves the conditioning tensor."""
    preset_path = os.path.join(PRESETS_DIR, preset_name)
    try:
        temp_model = TTSModel.load_model(num_threads=1)
        
        # Generate conditioning
        from pocket_tts.utils.utils import download_if_necessary
        from pocket_tts.data.audio import audio_read
        from pocket_tts.data.audio_utils import convert_audio
        
        voice_path = download_if_necessary(voice_path)
        audio, sample_rate = audio_read(voice_path)
        audio = convert_audio(audio, sample_rate, temp_model.config.mimi.sample_rate, 1)
        audio = audio.unsqueeze(0).to(temp_model.device)
        
        with torch.no_grad():
            encoded = temp_model.mimi.encode_to_latent(audio)
            latents = encoded.transpose(-1, -2).to(torch.float32)
            conditioning = torch.nn.functional.linear(latents, temp_model.flow_lm.speaker_proj_weight)
        
        # Save just the conditioning tensor
        torch.save({'conditioning': conditioning.cpu()}, preset_path)
        logger.info(f"Created preset: {preset_path}")
        return conditioning
    except Exception as e:
        logger.error(f"Failed to create preset: {e}")
        return None

# =====================================================
# Worker Thread (One Model Per Worker)
# =====================================================
class ChunkProcessor(threading.Thread):
    def __init__(self, chunk_queue, result_queue, voice_conditioning, run_id, run_output_dir):
        super().__init__(daemon=True)
        self.chunk_queue = chunk_queue
        self.result_queue = result_queue
        self.voice_conditioning = voice_conditioning  # This is the speaker embedding tensor
        self.run_id = run_id
        self.run_output_dir = run_output_dir
        self.should_stop = False

        logger.info(f"[{run_id}] Loading model for worker {self.name}")
        self.model = TTSModel.load_model(num_threads=1)
        
        # Initialize the model state properly using init_states
        self.model_state = self._init_model_state()
        logger.info(f"[{run_id}] Worker {self.name} ready")

    def _init_model_state(self):
        """Initialize fresh model state with voice conditioning."""
        # Create fresh state
        state = init_states(self.model.flow_lm, batch_size=1, sequence_length=self.voice_conditioning.shape[1])
        
        # Run the audio conditioning through the model to set up the state
        with torch.no_grad():
            # This mimics what get_state_for_audio_prompt does
            text_embeddings = torch.zeros((1, 0, self.model.flow_lm.dim), 
                                         dtype=self.model.flow_lm.dtype, 
                                         device=self.model.flow_lm.device)
            
            # Prepare conditioning
            audio_conditioning = self.voice_conditioning.to(self.model.flow_lm.device)
            
            # Run through flow_lm to initialize state properly
            self.model.flow_lm._sample_next_latent(
                torch.empty((1, 0, self.model.flow_lm.ldim), 
                           dtype=self.model.flow_lm.dtype, 
                           device=self.model.flow_lm.device),
                torch.cat([text_embeddings, audio_conditioning], dim=1),
                model_state=state,
                lsd_decode_steps=self.model.lsd_decode_steps,
                temp=self.model.temp,
                noise_clamp=self.model.noise_clamp,
                eos_threshold=self.model.eos_threshold,
            )
        
        return state

    def stop(self):
        self.should_stop = True

    def run(self):
        """Process chunks from the queue."""
        if self.voice_conditioning is None:
            logger.error(f"[{self.run_id}] Worker {self.name} has no voice conditioning")
            return
            
        while not self.should_stop:
            try:
                # Check paused state
                if state.is_paused():
                    time.sleep(0.1)
                    continue
                
                # Get next chunk with timeout
                try:
                    chunk_data = self.chunk_queue.get(timeout=0.5)
                except queue.Empty:
                    continue
                
                if chunk_data is None:
                    break

                idx, chunk_text = chunk_data

                try:
                    # Use the pre-initialized model state
                    # copy_state=True ensures we don't mutate the original state
                    audio = self.model.generate_audio(
                        model_state=copy.deepcopy(self.model_state),  # Deep copy to be safe
                        text_to_generate=chunk_text,
                        copy_state=True  # This creates a fresh copy for generation
                    )

                    if audio is None:
                        raise RuntimeError("Model returned None")

                    chunk_file = os.path.join(
                        self.run_output_dir,
                        f"chunk_{idx:03d}.wav"
                    )

                    save_audio_wav(audio, chunk_file)

                    self.result_queue.put((idx, chunk_file, "success"))
                    logger.info(f"[{self.run_id}] Chunk {idx} complete ({self.name})")

                except Exception as e:
                    logger.error(
                        f"[{self.run_id}] Error processing chunk {idx}: {e}",
                        exc_info=True
                    )
                    self.result_queue.put((idx, None, f"error: {e}"))

            except Exception as e:
                logger.error(f"[{self.run_id}] Worker crash: {e}", exc_info=True)

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

    # Load voice conditioning once
    temp_model = TTSModel.load_model(num_threads=1)
    voice_conditioning = load_voice_conditioning(voice_input, temp_model)
    if voice_conditioning is None:
        logger.error(f"[{run_id}] Voice validation failed")
        return
    del temp_model  # Free memory

    # Load text using parser
    text = load_text(book_path)
    if not text:
        logger.error(f"[{run_id}] Failed to load text from book")
        return

    # Chunk text properly
    chunks = list(chunk_text(text, chunk_size))
    if not chunks:
        logger.error(f"[{run_id}] No chunks generated")
        return

    logger.info(f"[{run_id}] Processing {len(chunks)} chunks with {num_threads or 'auto'} threads")

    run_output_dir = os.path.join(OUTPUTS_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    state.start(len(chunks))

    chunk_queue = queue.Queue()
    result_queue = queue.Queue()

    # Queue all chunks
    for idx, chunk in enumerate(chunks, start=1):
        chunk_queue.put((idx, chunk))

    # Determine number of workers
    if num_threads is None:
        num_workers = max(1, min(os.cpu_count() or 1, 4))
    else:
        num_workers = int(num_threads)

    # Add sentinel values to stop workers
    for _ in range(num_workers):
        chunk_queue.put(None)

    # Start workers - each gets the voice conditioning
    workers = []
    for _ in range(num_workers):
        worker = ChunkProcessor(
            chunk_queue,
            result_queue,
            voice_conditioning,  # Pass the voice conditioning tensor
            run_id,
            run_output_dir
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

        # Handle pause
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
            # Check if workers are done
            if not any(w.is_alive() for w in workers):
                break
            continue

    # Cleanup workers
    for worker in workers:
        worker.stop()
    for worker in workers:
        worker.join(timeout=5)

    # Combine audio files in order
    combined_audio = None
    success_count = 0
    
    for idx in range(1, len(chunks) + 1):
        chunk_file = os.path.join(run_output_dir, f"chunk_{idx:03d}.wav")
        if not os.path.exists(chunk_file):
            logger.warning(f"[{run_id}] Missing chunk file: {chunk_file}")
            continue

        try:
            segment = AudioSegment.from_wav(chunk_file)
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
    else:
        logger.error(f"[{run_id}] No audio generated")

    state.finish()
    logger.info(f"[{run_id}] TTS job complete - Success: {success_count}, Failed: {failed}")
    
    # Cleanup input file
    if os.path.exists(book_path):
        try:
            os.remove(book_path)
            logger.info(f"[{run_id}] Cleaned up input file: {book_path}")
        except Exception as e:
            logger.warning(f"[{run_id}] Failed to cleanup {book_path}: {e}")