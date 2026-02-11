import os, uuid, threading, logging
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
OUTPUTS_DIR = os.path.join(BASE_DIR,"outputs")
PRESETS_DIR = os.path.join(BASE_DIR,"speaker_presets")
os.makedirs(OUTPUTS_DIR, exist_ok=True)
os.makedirs(PRESETS_DIR, exist_ok=True)
app.mount("/outputs", StaticFiles(directory=OUTPUTS_DIR), name="outputs")

@app.get("/")
def serve_ui():
    index_path = os.path.join(BASE_DIR,"index.html")
    if not os.path.exists(index_path):
        raise HTTPException(status_code=500, detail="Frontend missing")
    return FileResponse(index_path)

@app.post("/generate")
async def generate(book: UploadFile, voice: UploadFile = None, preset: str = Form(None), chunk_size: int = Form(1000)):
    if state.lock.locked():
        raise HTTPException(409, "Generation in progress")

    run_id = uuid.uuid4().hex[:8]
    book_path = f"/tmp/{run_id}_{book.filename}"
    with open(book_path, "wb") as f:
        f.write(await book.read())

    if preset:
        # Use existing preset
        voice_path = os.path.join(PRESETS_DIR, preset)
        if not os.path.isfile(voice_path):
            raise HTTPException(404, f"Preset not found: {preset}")
    elif voice:
        # Save uploaded voice temporarily
        voice_path = f"/tmp/{run_id}_{voice.filename}"
        with open(voice_path, "wb") as f:
            f.write(await voice.read())
        # Optionally generate a new preset from this audio
        preset_name = f"{run_id}.pt"
        create_preset_from_audio(voice_path, preset_name)
        voice_path = os.path.join(PRESETS_DIR, preset_name)
    else:
        raise HTTPException(400, "No voice provided")

    output_file = os.path.join(OUTPUTS_DIR, f"{run_id}.mp3")

    threading.Thread(
        target=run_tts,
        kwargs={
            "book_path": book_path,
            "voice_input": voice_path,
            "chunk_size": chunk_size,
            "run_id": run_id,
            "output_file": output_file
        },
        daemon=True,
        name=f"tts-run-{run_id}"
    ).start()

    return {"status": "started", "run_id": run_id}


@app.post("/pause")
def pause(): state.pause(); return {"status":"paused"}
@app.post("/resume")
def resume(): state.resume(); return {"status":"resumed"}
@app.post("/stop")
def stop(): state.stop(); return {"status":"stopped"}

@app.get("/progress")
def get_progress():
    if state.total_chunks==0: return {"current":0,"total":0,"percent":0.0,"eta":0}
    percent = round((state.current_idx/state.total_chunks*100),1)
    return {"current":state.current_idx,"total":state.total_chunks,"percent":percent,"eta":state.eta_seconds}

@app.get("/history")
def history():
    dirs = [d for d in os.listdir(OUTPUTS_DIR) if os.path.isdir(os.path.join(OUTPUTS_DIR,d))]
    dirs.sort(key=lambda d: os.path.getmtime(os.path.join(OUTPUTS_DIR,d)), reverse=True)
    return {"history": dirs}

@app.get("/presets")
def list_presets():
    presets = sorted([f for f in os.listdir(PRESETS_DIR) if f.endswith(".pt")])
    return {"presets": presets}
