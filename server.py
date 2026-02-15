import os
import uuid
import threading
import logging
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from state import state
from gpu_monitor import get_gpu_monitor, GPUMonitorThread
import config

# Import the appropriate worker based on config
if config.WORKER_MODE == "fixed":
    from tts_worker import run_tts
    WORKER_NAME = "Fixed Cleanup"
elif config.WORKER_MODE == "isolated":
    from tts_worker import run_tts
    WORKER_NAME = "Isolated Runner"
elif config.WORKER_MODE == "batch_isolated":
    try:
        # Try the monitored version first
        from tts_worker import run_tts
        WORKER_NAME = "Batch Isolated (GPU Monitored)"
    except ImportError:
        # Fall back to non-monitored
        from tts_worker import run_tts
        WORKER_NAME = "Batch Isolated (Optimized)"
elif config.WORKER_MODE == "subprocess":
    from tts_worker import run_tts
    WORKER_NAME = "Subprocess Isolation"
else:
    # Fallback to original
    from tts_worker import run_tts
    WORKER_NAME = "Original (Legacy)"
from tts_worker import run_tts
WORKER_NAME = "Batch Isolated (GPU Monitored)"
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("tts_booklet.log", encoding="utf-8")]
)
logger = logging.getLogger("booklet.server")

logger.info(f"=== TTS Server Starting ===")
logger.info(f"Worker Mode: {config.WORKER_MODE.upper()} ({WORKER_NAME})")
if config.WORKER_MODE in config.WORKER_DESCRIPTIONS:
    desc = config.WORKER_DESCRIPTIONS[config.WORKER_MODE]
    logger.info(f"Description: {desc['description']}")
    logger.info(f"VRAM Usage: {desc['vram_usage']}")
    logger.info(f"Speed: {desc['speed']}")

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

# Initialize GPU monitor
gpu_monitor = get_gpu_monitor()

# Optional: Start background monitoring thread
gpu_monitor_thread = None
if getattr(config, 'ENABLE_GPU_MONITORING_THREAD', False):
    gpu_monitor_thread = GPUMonitorThread(interval=5.0)
    gpu_monitor_thread.start()
    logger.info("Background GPU monitoring thread started")

@app.get("/")
def serve_ui():
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="Frontend missing")
    return FileResponse(index_path)

@app.get("/config")
def get_config():
    """Return current worker configuration"""
    return {
        "worker_mode": config.WORKER_MODE,
        "worker_name": WORKER_NAME,
        "description": config.WORKER_DESCRIPTIONS.get(config.WORKER_MODE, {}),
        "vram_monitoring": config.ENABLE_VRAM_MONITORING,
        "default_batch_size": getattr(config, 'DEFAULT_BATCH_SIZE', 10),
        "gpu_monitoring_available": gpu_monitor.initialized
    }

@app.post("/generate")
async def generate(
    book: UploadFile, 
    voice: UploadFile, 
    transcription: str = Form(...),
    chunk_size: int = Form(400),
    batch_size: int = Form(10),
    cpu_threads: int = Form(None),
    speed: float = Form(0.9)
):
    if state.is_running():
        raise HTTPException(409, "Generation in progress")

    run_id = uuid.uuid4().hex[:8]
    book_path = f"/tmp/{run_id}_{book.filename}"
    voice_path = f"/tmp/{run_id}_{voice.filename}"

    # Validate speed
    if speed < 0.5 or speed > 2.0:
        speed = max(0.5, min(2.0, speed))

    # Validate chunk_size based on config
    if chunk_size > config.MAX_CHUNK_SIZE:
        logger.warning(f"Chunk size {chunk_size} exceeds max {config.MAX_CHUNK_SIZE}, adjusting")
        chunk_size = config.MAX_CHUNK_SIZE
    if chunk_size < config.MIN_CHUNK_SIZE:
        logger.warning(f"Chunk size {chunk_size} below min {config.MIN_CHUNK_SIZE}, adjusting")
        chunk_size = config.MIN_CHUNK_SIZE

    # Validate batch_size
    if batch_size < 1:
        batch_size = 1
    elif batch_size > 100:
        logger.warning(f"Batch size {batch_size} is very large, adjusting to 100")
        batch_size = 100

    # Validate transcription
    if not transcription or not transcription.strip():
        raise HTTPException(400, "Transcription is required - provide what is said in the voice sample")

    try:
        # Save uploaded files temporarily
        with open(book_path, "wb") as f:
            f.write(await book.read())
        with open(voice_path, "wb") as f:
            f.write(await voice.read())

        logger.info(f"=== New Generation Request ===")
        logger.info(f"Worker Mode: {config.WORKER_MODE.upper()}")
        logger.info(f"Voice: {voice.filename}")
        logger.info(f"Speed: {speed}")
        logger.info(f"Chunk size: {chunk_size}")
        if config.WORKER_MODE == "batch_isolated":
            logger.info(f"Batch size: {batch_size}")
        logger.info(f"Transcription: {len(transcription)} chars")

        output_file = os.path.join(OUTPUTS_DIR, f"{run_id}.mp3")

        # Prepare kwargs based on worker mode
        worker_kwargs = {
            "book_path": book_path,
            "voice_path": voice_path,
            "transcription": transcription.strip(),
            "chunk_size": chunk_size,
            "run_id": run_id,
            "output_file": output_file,
            "speed": speed
        }
        
        # Add batch_size only for batch_isolated mode
        if config.WORKER_MODE == "batch_isolated":
            worker_kwargs["batch_size"] = batch_size

        threading.Thread(
            target=run_tts,
            kwargs=worker_kwargs,
            daemon=True,
            name=f"tts-run-{run_id}"
        ).start()

        response = {
            "status": "started", 
            "run_id": run_id, 
            "speed": speed,
            "worker_mode": config.WORKER_MODE,
            "chunk_size": chunk_size
        }
        
        if config.WORKER_MODE == "batch_isolated":
            response["batch_size"] = batch_size

        return response

    except Exception as e:
        for f in [book_path, voice_path]:
            if os.path.exists(f):
                os.remove(f)
        logger.error(f"Error starting generation: {e}")
        raise HTTPException(500, str(e))


@app.post("/pause")
def pause(): 
    state.pause()
    return {"status": "paused"}

@app.post("/resume")
def resume(): 
    state.resume()
    return {"status": "resumed"}

@app.post("/stop")
def stop(): 
    state.stop()
    return {"status": "stopped"}

@app.get("/progress")
def get_progress():
    if state.total_chunks == 0:
        return {"current": 0, "total": 0, "percent": 0.0, "eta": 0, "failed": 0}
    percent = round((state.current_idx / state.total_chunks * 100), 1)
    return {
        "current": state.current_idx, 
        "total": state.total_chunks, 
        "percent": percent, 
        "eta": state.eta_seconds,
        "failed": state.failed_chunks
    }

@app.get("/download/{run_id}")
def download(run_id: str):
    mp3_path = os.path.join(OUTPUTS_DIR, f"{run_id}.mp3")
    if os.path.exists(mp3_path):
        return FileResponse(mp3_path, filename=f"booklet_{run_id}.mp3", media_type="audio/mpeg")
    raise HTTPException(404, "File not found")


# =====================================================
# GPU Monitoring Endpoints (nvidia-ml-py)
# =====================================================

@app.get("/gpu/stats")
def get_gpu_stats():
    """
    Get comprehensive GPU statistics.
    
    Returns detailed info including:
    - VRAM usage (used, free, total)
    - GPU utilization %
    - Temperature
    - Power consumption
    - Clock speeds
    - Device info
    """
    stats = gpu_monitor.get_stats()
    
    if stats is None:
        return {
            "available": False,
            "message": "GPU monitoring not available. Install: pip install nvidia-ml-py3"
        }
    
    return {
        "available": True,
        "stats": stats.to_dict()
    }


@app.get("/gpu/memory")
def get_gpu_memory():
    """
    Get GPU memory info (lightweight).
    
    Returns just VRAM usage without other metrics.
    Faster than /gpu/stats for frequent polling.
    """
    memory = gpu_monitor.get_memory_info()
    
    if memory["total_gb"] == 0:
        return {
            "available": False,
            "message": "GPU monitoring not available"
        }
    
    return {
        "available": True,
        "memory": memory
    }


@app.get("/gpu/summary")
def get_gpu_summary():
    """Get human-readable GPU summary"""
    stats = gpu_monitor.get_stats()
    
    if stats is None:
        return {
            "available": False,
            "summary": "GPU monitoring not available"
        }
    
    return {
        "available": True,
        "summary": stats.format_summary()
    }


@app.get("/gpu/history")
def get_gpu_history():
    """Get GPU monitoring history (if background thread enabled)"""
    if gpu_monitor_thread is None:
        return {
            "available": False,
            "message": "Background monitoring not enabled. Set ENABLE_GPU_MONITORING_THREAD=True in config"
        }
    
    history = gpu_monitor_thread.get_history(last_n=20)
    
    return {
        "available": True,
        "count": len(history),
        "history": [s.to_dict() for s in history]
    }


@app.get("/vram")
def get_vram_stats():
    """
    Legacy endpoint for backwards compatibility.
    
    Now uses nvidia-ml-py for accurate measurements.
    """
    memory = gpu_monitor.get_memory_info()
    stats = gpu_monitor.get_stats()
    
    if memory["total_gb"] == 0:
        return {
            "available": False, 
            "message": "GPU not available"
        }
    
    response = {
        "available": True,
        "allocated_gb": round(memory["used_gb"], 2),
        "total_gb": round(memory["total_gb"], 2),
        "free_gb": round(memory["free_gb"], 2),
        "usage_percent": round(memory["percent"], 1)
    }
    
    # Add extra stats if available
    if stats:
        response["temperature_c"] = stats.temperature
        response["gpu_utilization"] = stats.gpu_utilization
        response["power_usage_w"] = round(stats.power_usage, 1)
        response["device"] = stats.device_name
    
    return response


@app.on_event("shutdown")
def shutdown_event():
    """Cleanup on shutdown"""
    if gpu_monitor_thread:
        gpu_monitor_thread.stop()
    logger.info("Server shutdown complete")