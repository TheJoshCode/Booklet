import os
import uuid
import threading
import logging
from fastapi import FastAPI, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from tts_worker import run_tts
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
os.makedirs(OUTPUTS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

@app.get("/")
def serve_ui():
    index_path = os.path.join(BASE_DIR, "index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="Frontend missing")
    return FileResponse(index_path)

@app.post("/generate")
async def generate(
    book: UploadFile, 
    voice: UploadFile, 
    transcription: str = Form(...),  # REQUIRED - what the voice sample says
    chunk_size: int = Form(400),  # Reduced from 800 - smaller chunks maintain voice better
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

    # Validate transcription
    if not transcription or not transcription.strip():
        raise HTTPException(400, "Transcription is required - provide what is said in the voice sample")

    try:
        # Save uploaded files temporarily
        with open(book_path, "wb") as f:
            f.write(await book.read())
        with open(voice_path, "wb") as f:
            f.write(await voice.read())

        logger.info(f"Generate request - voice: {voice.filename}, speed: {speed}, transcription provided: {len(transcription)} chars")

        output_file = os.path.join(OUTPUTS_DIR, f"{run_id}.mp3")

        threading.Thread(
            target=run_tts,
            kwargs={
                "book_path": book_path,
                "voice_path": voice_path,
                "transcription": transcription.strip(),
                "chunk_size": chunk_size,
                "run_id": run_id,
                "output_file": output_file,
                "speed": speed
            },
            daemon=True,
            name=f"tts-run-{run_id}"
        ).start()

        return {"status": "started", "run_id": run_id, "speed": speed}

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