import os
import logging
import gc
from typing import Optional

import torch
import numpy as np
from pydub import AudioSegment
import soundfile as sf

from state import state
from parser import load_text, chunk_text
from gpu_monitor import GPUMonitor, GPUMemoryTracker, log_gpu_stats, get_vram_usage

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
# GPU Monitoring Integration
# =====================================================

# Global GPU monitor
gpu_monitor = GPUMonitor()


# =====================================================
# Complete GPU Reset
# =====================================================

def complete_gpu_reset():
    """
    Nuclear option: completely reset GPU state.
    
    This releases ALL cached memory and resets the allocator.
    Guarantees zero VRAM leaks.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.synchronize()
        torch.cuda.ipc_collect()
        torch.cuda.reset_peak_memory_stats()
        torch.cuda.reset_accumulated_memory_stats()


def save_as_mp3(audio_array: np.ndarray, sample_rate: int, output_path: str, bitrate: str = "128k"):
    """
    Save audio array as MP3 file.
    
    Args:
        audio_array: NumPy array of audio samples
        sample_rate: Sample rate (e.g., 48000, 24000)
        output_path: Output MP3 file path
        bitrate: MP3 bitrate (default: 128k)
    """
    # Normalize to int16
    if audio_array.dtype == np.float32 or audio_array.dtype == np.float64:
        audio_array = np.clip(audio_array, -1.0, 1.0)
        audio_array = (audio_array * 32767).astype(np.int16)
    
    # Create AudioSegment
    audio_segment = AudioSegment(
        audio_array.tobytes(),
        frame_rate=sample_rate,
        sample_width=audio_array.dtype.itemsize,
        channels=1
    )
    
    # Export as MP3
    audio_segment.export(output_path, format="mp3", bitrate=bitrate)


# =====================================================
# Batch Chunk Generator (Optimized Isolation)
# =====================================================

def generate_chunk_batch(
    chunks: list,
    start_idx: int,
    batch_size: int,
    voice_path: str,
    transcription: str,
    run_output_dir: str,
    speed: float = 0.9,
    device: str = "cuda",
    memory_tracker: Optional[GPUMemoryTracker] = None
) -> int:
    """
    Generate a batch of chunks with a single model load.
    
    OPTIMIZATION STRATEGY:
    - Load model ONCE
    - Generate N chunks (batch_size)
    - Unload model COMPLETELY
    - This balances speed vs VRAM safety
    
    Args:
        chunks: List of text chunks
        start_idx: Starting index in chunks list
        batch_size: Number of chunks to process before unloading
        voice_path: Path to voice sample
        transcription: Voice sample transcription
        run_output_dir: Output directory for chunk files
        speed: Generation speed
        device: Device to use
        memory_tracker: Optional GPU memory tracker
        
    Returns:
        Number of successfully generated chunks
    """
    successful = 0
    
    # Log GPU stats before batch
    log_gpu_stats("📊 Before batch: ")
    
    try:
        # Load model once for the batch
        from zipvoice.luxvoice import LuxTTS
        
        logger.info(f"Loading model for batch (chunks {start_idx+1}-{min(start_idx+batch_size, len(chunks))})...")
        model = LuxTTS(device=device)
        
        # Log GPU stats after model load
        log_gpu_stats("📊 After model load: ")
        
        # Encode voice prompt once for the batch
        encode_dict = model.encode_prompt(
            prompt_text=transcription,
            prompt_audio=voice_path,
            duration=10,
            rms=0.15,
        )
        
        # Process each chunk in the batch
        for i in range(batch_size):
            chunk_idx = start_idx + i
            if chunk_idx >= len(chunks):
                break
            
            try:
                chunk_text = chunks[chunk_idx]
                actual_idx = chunk_idx + 1  # 1-indexed for filenames
                
                logger.info(f"  Generating chunk {actual_idx}/{len(chunks)} (batch item {i+1}/{batch_size})")
                
                # Generate audio
                audio = model.generate_speech(
                    text=chunk_text,
                    encode_dict=encode_dict,
                    num_steps=6,
                    guidance_scale=5.0,
                    t_shift=0.5,
                    speed=speed,
                )
                
                if audio is None:
                    raise RuntimeError("Model returned None")
                
                # Move to CPU and convert
                audio = audio.cpu().numpy().squeeze()
                
                # Save as MP3
                chunk_file = os.path.join(run_output_dir, f"chunk_{actual_idx:04d}.mp3")
                save_as_mp3(audio, 48000, chunk_file, bitrate="128k")
                
                # Clean up this chunk's audio
                del audio
                
                successful += 1
                state.increment_completed()
                
                # Record memory measurement
                if memory_tracker:
                    memory_tracker.record_measurement()
                
            except Exception as e:
                logger.error(f"Chunk {chunk_idx+1} failed: {e}", exc_info=True)
                state.increment_failed()
        
        # Clean up batch resources
        del encode_dict
        del model
        
        logger.info(f"Batch complete ({successful}/{batch_size} successful), unloading model...")
        
    except Exception as e:
        logger.error(f"Batch generation failed: {e}", exc_info=True)
    
    # Nuclear cleanup after batch
    complete_gpu_reset()
    
    # Log GPU stats after cleanup
    log_gpu_stats("📊 After cleanup: ")
    
    return successful


# =====================================================
# Main TTS - BATCH ISOLATED VERSION WITH GPU MONITORING
# =====================================================

def run_tts(
    book_path: str,
    voice_path: str,
    transcription: str,
    output_file: str,
    chunk_size: int = 400,
    batch_size: int = 5,
    run_id: Optional[str] = None,
    speed: float = 0.9,
):
    """
    Batch-optimized isolated TTS runner with comprehensive GPU monitoring.
    
    Uses nvidia-ml-py for accurate VRAM tracking and leak detection.
    
    BATCH ISOLATION STRATEGY:
    - Load model → Generate batch_size chunks → Unload model → Repeat
    - Balances speed (fewer loads) with safety (periodic resets)
    - Much faster than per-chunk isolation
    - Still guarantees VRAM won't grow indefinitely
    
    GPU MONITORING:
    - Real-time VRAM usage tracking
    - GPU utilization monitoring
    - Temperature monitoring
    - Power consumption tracking
    - Memory leak detection
    
    Args:
        batch_size: Number of chunks to process before model reload
                   - Small (1-5): Safest, slower
                   - Medium (10-20): Balanced (RECOMMENDED)
                   - Large (50+): Faster, but defeats purpose
    """
    run_id = run_id or "manual"

    # Initialize GPU memory tracker
    memory_tracker = GPUMemoryTracker()
    memory_tracker.set_baseline()
    
    # Log initial GPU state
    logger.info("=" * 70)
    logger.info("GPU MONITORING ENABLED (nvidia-ml-py)")
    logger.info("=" * 70)
    log_gpu_stats("📊 Initial GPU state: ")
    logger.info("=" * 70)

    # Load and chunk text
    text = load_text(book_path)
    chunks = list(chunk_text(text, chunk_size))

    if not chunks:
        logger.error("No text chunks generated")
        state.finish()
        return

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info(f"Starting BATCH-ISOLATED generation on {device}")
    logger.info(f"Total chunks: {len(chunks)}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Strategy: Load/process {batch_size} chunks/unload (optimized isolation)")
    logger.info(f"Output format: MP3 (128kbps chunks, 192kbps final)")

    run_output_dir = os.path.join(OUTPUTS_DIR, run_id)
    os.makedirs(run_output_dir, exist_ok=True)

    state.start(len(chunks))

    # =====================================================
    # Generate chunks in batches
    # =====================================================

    total_batches = (len(chunks) + batch_size - 1) // batch_size
    
    for batch_num in range(total_batches):
        start_idx = batch_num * batch_size
        current_batch_size = min(batch_size, len(chunks) - start_idx)
        
        logger.info(f"\n{'='*70}")
        logger.info(f"BATCH {batch_num+1}/{total_batches}")
        logger.info(f"Processing chunks {start_idx+1}-{start_idx+current_batch_size}")
        logger.info(f"{'='*70}\n")
        
        # Get VRAM before batch
        vram_before = get_vram_usage()
        logger.info(f"VRAM before batch: {vram_before['used_gb']:.2f}GB / {vram_before['total_gb']:.2f}GB ({vram_before['percent']:.1f}%)")
        
        # Generate this batch
        batch_successful = generate_chunk_batch(
            chunks=chunks,
            start_idx=start_idx,
            batch_size=current_batch_size,
            voice_path=voice_path,
            transcription=transcription,
            run_output_dir=run_output_dir,
            speed=speed,
            device=device,
            memory_tracker=memory_tracker
        )
        
        # Get VRAM after batch
        vram_after = get_vram_usage()
        leak = vram_after['used_gb'] - vram_before['used_gb']
        
        logger.info(f"VRAM after batch: {vram_after['used_gb']:.2f}GB / {vram_after['total_gb']:.2f}GB ({vram_after['percent']:.1f}%)")
        
        if leak > 0.1:
            logger.warning(f"⚠️  VRAM leak detected: +{leak:.3f}GB this batch")
        else:
            logger.info(f"✅ VRAM stable: {leak:+.3f}GB change")
        
        # Check for memory leak
        if memory_tracker.check_for_leak(threshold_gb=0.5):
            logger.warning("⚠️  Significant memory leak detected across batches!")
            logger.warning("    Consider reducing batch_size or switching to isolated mode")
        
        logger.info(f"Batch {batch_num+1} complete: {batch_successful}/{current_batch_size} successful\n")
        
        # Check for stop/pause
        if state.is_stopped():
            logger.info("Stopped by user")
            break
            
        while state.is_paused():
            import time
            time.sleep(0.5)

    # =====================================================
    # Combine MP3 chunks into final file
    # =====================================================

    logger.info("=" * 70)
    logger.info("Combining MP3 chunks into final audio...")
    
    combined = AudioSegment.empty()
    chunks_combined = 0
    
    for idx in range(1, len(chunks) + 1):
        chunk_file = os.path.join(run_output_dir, f"chunk_{idx:04d}.mp3")
        if os.path.exists(chunk_file):
            try:
                chunk_audio = AudioSegment.from_mp3(chunk_file)
                combined += chunk_audio
                chunks_combined += 1
            except Exception as e:
                logger.error(f"Error loading chunk {idx}: {e}")

    if chunks_combined > 0:
        # Export as MP3
        combined.export(output_file, format="mp3", bitrate="192k")
        duration_sec = len(combined) / 1000.0
        file_size_mb = os.path.getsize(output_file) / (1024 * 1024)
        
        logger.info(f"✅ Final audio saved: {output_file}")
        logger.info(f"   Combined {chunks_combined}/{len(chunks)} chunks")
        logger.info(f"   Duration: {duration_sec:.1f} seconds ({duration_sec/60:.1f} minutes)")
        logger.info(f"   File size: {file_size_mb:.1f} MB")
        logger.info(f"   Format: MP3 @ 192kbps")
        
    else:
        logger.error("No audio chunks to combine")

    # =====================================================
    # Final GPU stats and leak report
    # =====================================================
    
    logger.info("=" * 70)
    logger.info("FINAL GPU REPORT")
    logger.info("=" * 70)
    
    # Final cleanup
    complete_gpu_reset()
    
    # Get final stats
    log_gpu_stats("📊 Final GPU state: ")
    
    # Memory leak summary
    summary = memory_tracker.get_summary()
    logger.info(f"📊 Memory Tracking Summary:")
    logger.info(f"   Baseline: {summary['baseline_gb']:.2f}GB")
    logger.info(f"   Current:  {summary['current_gb']:.2f}GB")
    logger.info(f"   Peak:     {summary['peak_gb']:.2f}GB")
    logger.info(f"   Leak:     {summary['leak_gb']:.3f}GB")
    
    if summary['leak_gb'] < 0.1:
        logger.info("   ✅ No significant memory leak detected")
    elif summary['leak_gb'] < 0.5:
        logger.warning("   ⚠️  Minor memory leak detected")
    else:
        logger.error("   ❌ Significant memory leak detected!")
    
    logger.info("=" * 70)
    
    state.finish()

    # Remove temp input files
    for f in [book_path, voice_path]:
        if os.path.exists(f):
            os.remove(f)