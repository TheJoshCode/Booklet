import os
import uuid
import threading
import logging
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tts_worker import run_tts, create_preset_from_audio
from state import state

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s | %(levelname)-7s | %(threadName)-12s | %(message)s',
    handlers=[logging.StreamHandler(), logging.FileHandler("tts_booklet.log", encoding="utf-8")]
)
logger = logging.getLogger("booklet.server")

app = FastAPI()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
PRESETS_DIR = os.path.join(BASE_DIR, "speaker_presets")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PRESETS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

@app.get("/")
def serve_ui():
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="Frontend missing")
    return FileResponse(index_path)

@app.post("/generate")
async def generate(book: UploadFile, voice: UploadFile = None, preset: str = Form(None), chunk_size: int = Form(1000), cpu_threads: int = Form(None)):
    # FIXED: Use is_running() instead of lock.locked()
    if state.is_running():
        raise HTTPException(409, "Generation in progress")

    run_id = uuid.uuid4().hex[:8]
    book_path = f"/tmp/{run_id}_{book.filename}"
    voice_path = None
    temp_files = [book_path]  # Track files for cleanup
    
    try:
        with open(book_path, "wb") as f:
            f.write(await book.read())

        # Clean up preset value - treat empty string as None
        if preset == "" or preset is None:
            preset = None
        
        logger.info(f"Generate request - preset: {preset}, voice uploaded: {voice is not None}")

        if preset:
            # Use existing preset - pass the full path to the .pt file
            voice_path = os.path.join(PRESETS_DIR, preset)
            if not os.path.isfile(voice_path):
                raise HTTPException(404, f"Preset not found: {preset}")
            logger.info(f"Using existing preset: {voice_path}")
        elif voice:
            # Save uploaded voice temporarily and create a preset from it
            temp_voice_path = f"/tmp/{run_id}_{voice.filename}"
            temp_files.append(temp_voice_path)
            with open(temp_voice_path, "wb") as f:
                f.write(await voice.read())
            # Generate a new preset from this audio
            preset_name = f"{run_id}.pt"
            logger.info(f"Creating preset from uploaded voice: {temp_voice_path}")
            create_preset_from_audio(temp_voice_path, preset_name)
            voice_path = os.path.join(PRESETS_DIR, preset_name)
            logger.info(f"Created new preset: {voice_path}")
        else:
            raise HTTPException(400, "No voice or preset provided")

        output_file = os.path.join(OUTPUTS_DIR, f"{run_id}.mp3")

        threading.Thread(
            target=run_tts,
            kwargs={
                "book_path": book_path,
                "voice_input": voice_path,
                "chunk_size": chunk_size,
                "run_id": run_id,
                "output_file": output_file,
                "num_threads": cpu_threads
            },
            daemon=True,
            name=f"tts-run-{run_id}"
        ).start()

        return {"status": "started", "run_id": run_id}
        
    except HTTPException:
        # FIXED: Cleanup on error
        for f in temp_files:
            if os.path.exists(f):
                os.remove(f)
        raise
    except Exception as e:
        # FIXED: Cleanup on error
        for f in temp_files:
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

@app.get("/history")
def history():
    dirs = [d for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR, d))]
    dirs.sort(key=lambda d: os.path.getmtime(os.path.join(OUTPUTS_DIR, d)), reverse=True)
    return {"history": dirs}

@app.get("/presets")
def list_presets():
    presets = sorted([f for f in os.listdir(PRESETS_DIR) if f.endswith(".pt")])
    return {"presets": presets}

@app.get("/download/{run_id}")
def download(run_id: str):
    """Download the generated MP3 file"""
    mp3_path = os.path.join(OUTPUTS_DIR, f"{run_id}.mp3")
    if os.path.exists(mp3_path):
        return FileResponse(mp3_path, filename=f"booklet_{run_id}.mp3", media_type="audio/mpeg")
    raise HTTPException(404, "File not found")