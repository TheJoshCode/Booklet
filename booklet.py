import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from flask import Flask, request, jsonify, send_from_directory
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
import threading
import re
import numpy as np
import os
import time
from datetime import datetime
import uuid
import logging
from urllib.parse import urlparse
import socket
import ipaddress
import requests
import webbrowser
from threading import Lock

# ────────────────────────────────────────────────
# Logging Stuff
# ────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('booklet.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# ────────────────────────────────────────────────
# Flask App Conf Stuff
# ────────────────────────────────────────────────
app = Flask(__name__)

app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', os.urandom(32).hex())
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MiB max request size

app.config['WTF_CSRF_ENABLED'] = False
logger.warning("CSRF protection DISABLED – local development mode only")

csrf = CSRFProtect(app)  # still initialize, but disabled via config

limiter = Limiter(
    app=app,
    key_func=get_remote_address,
    default_limits=["999999999 per day", "999999999 per hour"],
    storage_uri="memory://"
)

# ────────────────────────────────────────────────
# Directories
# ────────────────────────────────────────────────
OUTPUT_DIRECTORY = os.path.abspath('outputs')
TEMP_DIRECTORY   = os.path.abspath('temp')

os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
os.makedirs(TEMP_DIRECTORY,   exist_ok=True)

# ────────────────────────────────────────────────
# Constants & Limits
# ────────────────────────────────────────────────
MAX_TEXT_CHARACTERS         = 999999          # ~500KB of text (unlimited in practice)
MAX_REFERENCE_TEXT_LENGTH   = 1000
MAX_REFERENCE_AUDIO_SIZE    = 10 * 1024 * 1024   # 10 MiB
MAX_BOOK_FILE_SIZE          = 10 * 1024 * 1024   # 10 MiB
FILE_RETENTION_SECONDS      = 24 * 3600   # 24 hours

SUPPORTED_LANGUAGES = ['English', 'Spanish', 'French', 'German', 'Chinese', 'Japanese']

ALLOWED_URL_SCHEMES = {'http', 'https'}
BLOCKED_HOSTNAMES   = {'localhost', '127.0.0.1', '0.0.0.0', '::1'}

PRIVATE_IP_NETWORKS = [
    ipaddress.ip_network('10.0.0.0/8'),
    ipaddress.ip_network('172.16.0.0/12'),
    ipaddress.ip_network('192.168.0.0/16'),
    ipaddress.ip_network('169.254.0.0/16'),
    ipaddress.ip_network('127.0.0.0/8'),
    ipaddress.ip_network('::1/128'),
    ipaddress.ip_network('fe80::/10'),
    ipaddress.ip_network('fc00::/7'),
]

# ────────────────────────────────────────────────
# Loadin Da Model
# ────────────────────────────────────────────────
tts_model = None
try:
    logger.info("Loading Qwen3-TTS model...")
    tts_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16
    )
    logger.info("Model loaded successfully")
except Exception as exc:
    logger.error(f"Failed to load model: {exc}")
    tts_model = None

# ────────────────────────────────────────────────
# Session Management Stuff
# ────────────────────────────────────────────────
generation_sessions = {}
sessions_lock = Lock()


class GenerationSession:
    def __init__(self):
        self.id = str(uuid.uuid4())
        self.progress_percentage = 0
        self.output_filename = None
        self.is_generating = False
        self.error_message = None
        self.created_timestamp = time.time()
        self.start_time = None
        self.estimated_time_remaining = None
        self.total_chunks = 0
        self.processed_chunks = 0


def remove_expired_sessions():
    cutoff_time = time.time() - 3600  # 1 hour
    with sessions_lock:
        expired = [
            sid for sid, session in generation_sessions.items()
            if session.created_timestamp < cutoff_time
        ]
        for sid in expired:
            del generation_sessions[sid]
    if expired:
        logger.info(f"Cleaned up {len(expired)} expired sessions")


def remove_old_files():
    cutoff_time = time.time() - FILE_RETENTION_SECONDS
    deleted_count = 0

    for directory in (OUTPUT_DIRECTORY, TEMP_DIRECTORY):
        try:
            for filename in os.listdir(directory):
                path = os.path.join(directory, filename)
                if os.path.isfile(path) and os.path.getmtime(path) < cutoff_time:
                    try:
                        os.remove(path)
                        deleted_count += 1
                    except Exception as exc:
                        logger.error(f"Failed to delete {path}: {exc}")
        except Exception as exc:
            logger.error(f"Error during cleanup of {directory}: {exc}")

    if deleted_count:
        logger.info(f"Deleted {deleted_count} old files")


def periodic_cleanup_task():
    while True:
        time.sleep(3600)  # every hour
        try:
            remove_old_files()
            remove_expired_sessions()
        except Exception as exc:
            logger.error(f"Periodic cleanup error: {exc}")


threading.Thread(target=periodic_cleanup_task, daemon=True).start()


# ────────────────────────────────────────────────
# Sec Def is here
# ────────────────────────────────────────────────

def is_private_ip_address(ip_str: str) -> bool:
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in net for net in PRIVATE_IP_NETWORKS)
    except ValueError:
        return True  # treat invalid as unsafe


def is_safe_url(url: str) -> tuple[bool, str | None]:
    try:
        parsed = urlparse(url)

        if parsed.scheme not in ALLOWED_URL_SCHEMES:
            return False, "Only http/https schemes allowed"

        hostname = parsed.hostname
        if not hostname:
            return False, "No hostname in URL"

        if hostname.lower() in BLOCKED_HOSTNAMES:
            return False, "Localhost / loopback access blocked"

        try:
            resolved_ip = socket.gethostbyname(hostname)
            if is_private_ip_address(resolved_ip):
                return False, "Private / internal IP addresses blocked"
        except socket.gaierror:
            return False, "Hostname resolution failed"

        return True, None

    except Exception as exc:
        logger.error(f"URL safety check failed: {exc}")
        return False, "Invalid URL format"


def download_reference_audio(url: str, destination_path: str) -> tuple[bool, str | None]:
    try:
        safe, reason = is_safe_url(url)
        if not safe:
            return False, reason

        response = requests.get(
            url,
            timeout=30,
            stream=True,
            allow_redirects=True,
            headers={'User-Agent': 'Booklet/1.0'}
        )
        response.raise_for_status()

        content_type = response.headers.get('Content-Type', '')
        if not content_type.startswith('audio/'):
            return False, "Resource is not an audio file"

        content_length = response.headers.get('Content-Length')
        if content_length and int(content_length) > MAX_REFERENCE_AUDIO_SIZE:
            return False, f"File too large (max {MAX_REFERENCE_AUDIO_SIZE // 1024 // 1024} MiB)"

        downloaded = 0
        with open(destination_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                downloaded += len(chunk)
                if downloaded > MAX_REFERENCE_AUDIO_SIZE:
                    os.remove(destination_path)
                    return False, "Downloaded file exceeds size limit"
                f.write(chunk)

        logger.info(f"Successfully downloaded reference audio from {url}")
        return True, None

    except requests.RequestException as exc:
        logger.error(f"Download failed: {exc}")
        return False, f"Download error: {str(exc)}"
    except Exception as exc:
        logger.error(f"Unexpected download error: {exc}")
        return False, "Unexpected error while downloading audio"


def is_valid_uploaded_file(file, max_size_bytes: int, required_extension: str | None = None) -> tuple[bool, str | None]:
    if not file or not file.filename:
        return False, "No file received"

    file.seek(0, os.SEEK_END)
    size = file.tell()
    file.seek(0)

    if size == 0:
        return False, "File is empty"

    if size > max_size_bytes:
        return False, f"File too large (max {max_size_bytes // 1024 // 1024} MiB)"

    if required_extension and not file.filename.lower().endswith(required_extension):
        return False, f"File must have .{required_extension.lstrip('.')} extension"

    return True, None


def validate_text(text: str, max_length: int, field_name: str) -> tuple[bool, str | None]:
    if not text or not text.strip():
        return False, f"{field_name} is required"

    cleaned = text.strip()

    if len(cleaned) > max_length:
        return False, f"{field_name} exceeds maximum length ({max_length} characters)"

    return True, cleaned


# ────────────────────────────────────────────────
# Background Audio Generation
# ────────────────────────────────────────────────

def generate_audiobook_in_background(
    book_text: str,
    reference_audio_path: str | None,
    reference_transcript: str,
    language: str,
    session: GenerationSession
):
    try:
        session.progress_percentage = 0
        session.is_generating = True
        session.start_time = time.time()

        if tts_model is None:
            session.error_message = "TTS model not available"
            session.is_generating = False
            logger.error("Generation attempted but model is not loaded")
            return

        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<=\.|\?|\!)\s', book_text)
        sentences = [s.strip() for s in sentences if s.strip()]

        if not sentences:
            session.error_message = "No valid sentences found in text"
            session.is_generating = False
            return

        logger.info(f"Session {session.id}: processing {len(sentences)} sentences")
        
        # Initialize chunk counters
        batch_size = 5
        session.total_chunks = (len(sentences) + batch_size - 1) // batch_size
        session.processed_chunks = 0

        audio_chunks = []

        for i in range(0, len(sentences), batch_size):
            if not session.is_generating:
                logger.info(f"Session {session.id}: generation cancelled by user")
                return

            batch = sentences[i:i + batch_size]
            languages = [language] * len(batch)

            try:
                wavs, sample_rate = tts_model.generate_voice_clone(
                    text=batch,
                    language=languages,
                    ref_audio=reference_audio_path,
                    ref_text=reference_transcript,
                )
                audio_chunks.extend(wavs)
                processed = i + len(batch)
                session.processed_chunks += 1
                session.progress_percentage = min(round((processed / len(sentences)) * 100), 99)
                
                elapsed_time = time.time() - session.start_time
                if session.processed_chunks > 0:
                    avg_time_per_chunk = elapsed_time / session.processed_chunks
                    chunks_remaining = session.total_chunks - session.processed_chunks
                    session.estimated_time_remaining = round(avg_time_per_chunk * chunks_remaining)
            except Exception as exc:
                logger.error(f"Batch generation failed: {exc}")
                session.error_message = f"Generation error: {str(exc)}"
                session.is_generating = False
                return

        full_audio = np.concatenate(audio_chunks, axis=0)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        short_id = session.id[:8]
        filename = f"audiobook_{timestamp}_{short_id}.wav"
        full_path = os.path.join(OUTPUT_DIRECTORY, filename)

        sf.write(full_path, full_audio, sample_rate)
        logger.info(f"Session {session.id}: saved audio → {filename}")

        session.output_filename = filename
        session.progress_percentage = 100
        session.is_generating = False
        if reference_audio_path and os.path.exists(reference_audio_path) and reference_audio_path.startswith(TEMP_DIRECTORY):
            try:
                os.remove(reference_audio_path)
            except Exception as exc:
                logger.error(f"Failed to remove temp reference file: {exc}")

    except Exception as exc:
        logger.error(f"Unexpected generation error: {exc}", exc_info=True)
        session.error_message = "Unexpected error during generation"
        session.is_generating = False


# ────────────────────────────────────────────────
# Routes
# ────────────────────────────────────────────────

@app.after_request
def add_security_headers(response):
    response.headers['X-Content-Type-Options'] = 'nosniff'
    response.headers['X-Frame-Options'] = 'DENY'
    response.headers['X-XSS-Protection'] = '1; mode=block'
    response.headers['Referrer-Policy'] = 'strict-origin-when-cross-origin'
    return response


@app.route('/')
def serve_main_page():
    return """
<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>Booklet</title>
<meta name="viewport" content="width=device-width, initial-scale=1">

<style>
:root {
    --bg: #0f1115;
    --card: #161a22;
    --text: #e6e8eb;
    --muted: #9aa0a6;
    --accent: #4f8cff;
    --border: #242a36;
    --error: #ff4444;
    --success: #44ff88;
}

* { box-sizing: border-box; }

body {
    margin: 0;
    background: var(--bg);
    color: var(--text);
    font-family: system-ui, sans-serif;
    line-height: 1.5;
}

.container {
    max-width: 1200px;
    margin: 40px auto;
    padding: 0 20px;
}

.header-box {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 20px;
    text-align: center;
    margin-bottom: 24px;
}

h1 { margin: 0 0 8px; font-size: 1.8rem; }

.subtitle { margin: 0; color: var(--muted); font-size: 1rem; }

.main-content {
    display: flex;
    gap: 24px;
    flex-wrap: wrap;
    align-items: stretch;
    min-height: 60vh;
}

.card, .outputs-panel {
    background: var(--card);
    border: 1px solid var(--border);
    border-radius: 14px;
    padding: 24px;
    flex: 1;
    min-width: 320px;
    display: flex;
    flex-direction: column;
}

.outputs-panel {
    max-height: none;
    overflow-y: auto;
}

h2 { margin-top: 0; font-size: 1.3rem; color: var(--text); }

form { display: grid; gap: 16px; flex: 1; }

label { 
    font-size: 0.85rem; 
    color: var(--muted); 
    display: block;
    margin-bottom: 4px;
}

.help-text {
    font-size: 0.75rem;
    color: var(--muted);
    margin-top: 4px;
}

input[type="text"], input[type="file"], textarea, select {
    width: 100%;
    background: #0c0f14;
    color: var(--text);
    border: 1px solid var(--border);
    border-radius: 10px;
    padding: 10px 12px;
    font-size: 0.9rem;
}

textarea { min-height: 140px; resize: vertical; }

input[type="file"] { padding: 8px; }

select {
    cursor: pointer;
}

button {
    margin-top: 8px;
    padding: 12px;
    font-size: 0.95rem;
    font-weight: 600;
    color: white;
    background: var(--accent);
    border: none;
    border-radius: 12px;
    cursor: pointer;
    transition: opacity 0.2s;
}

button:hover { opacity: 0.9; }

button:disabled {
    opacity: 0.5;
    cursor: not-allowed;
}

progress {
    width: 100%;
    height: 14px;
    border-radius: 8px;
    overflow: hidden;
    appearance: none;
}

progress::-webkit-progress-bar { background: #0c0f14; }
progress::-webkit-progress-value { background: var(--accent); }
progress::-moz-progress-bar { background: var(--accent); }

.status { 
    margin-top: 10px; 
    font-size: 0.85rem; 
    color: var(--muted); 
    min-height: 1.2em; 
}

.status.error { color: var(--error); }
.status.success { color: var(--success); }

.footer { 
    margin-top: 25px; 
    text-align: center; 
    font-size: 0.75rem; 
    color: var(--muted); 
}

.output-item {
    margin-bottom: 16px;
    padding: 12px;
    background: #0c0f14;
    border-radius: 10px;
    border: 1px solid var(--border);
}

.output-item p { margin: 0 0 8px; font-size: 0.9rem; }

.output-actions {
    display: flex;
    gap: 8px;
}

.toggle-btn, .download-btn {
    padding: 8px 16px;
    background: var(--accent);
    border: none;
    border-radius: 8px;
    color: white;
    cursor: pointer;
    font-size: 0.9rem;
    flex: 1;
}

.toggle-btn.playing { background: #ff4d4d; }

.download-btn { background: #44aa66; }

.empty-state {
    text-align: center;
    padding: 40px 20px;
    color: var(--muted);
}

@media (max-width: 800px) {
    .main-content { flex-direction: column; }
}
</style>
</head>

<body>
<div class="container">
    <div class="header-box">
        <h1>Booklet</h1>
        <div class="subtitle">AUDIOBOOKS FROM 5 SECOND VOICE CLIPS</div>
    </div>

    <div class="main-content">
        <div class="card">
            <h2>MAIN</h2>
            <form id="form">
                <div>
                    <label>WEB LINK FOR VOICE</label>
                    <input type="text" name="ref_audio_url" placeholder="https://example.com/voice.wav">
                </div>

                <div>
                    <label>VOICE TO CLONE (WAV FILE)</label>
                    <input type="file" name="ref_audio_file" accept=".wav">
                    <div class="help-text">Upload a WAV file (max 10MB)</div>
                </div>

                <div>
                    <label>REF TRANSCRIPTION</label>
                    <input type="text" name="ref_text" placeholder="Exact words spoken in the voice clip" required>
                    <div class="help-text">Max 1000 characters</div>
                </div>

                <div>
                    <label>BOOK TEXT</label>
                    <textarea name="novel_text" placeholder="Enter the book or novel text here..." required></textarea>
                </div>

                <div>
                    <label>BOOK FILE (TXT)</label>
                    <input type="file" name="novel_file" accept=".txt">
                    <div class="help-text">Upload a text file (max 10MB)</div>
                </div>

                <div>
                    <label>LANGUAGE</label>
                    <select name="language" required>
                        <option value="English">English</option>
                        <option value="Spanish">Spanish</option>
                        <option value="French">French</option>
                        <option value="German">German</option>
                        <option value="Chinese">Chinese</option>
                        <option value="Japanese">Japanese</option>
                    </select>
                </div>

                <button type="submit" id="submitBtn">Generate</button>
            </form>

            <div class="progress-wrapper">
                <progress id="prog" value="0" max="100"></progress>
                <div id="status" class="status"></div>
            </div>
        </div>

        <div class="outputs-panel">
            <h2>History</h2>
            <div id="outputs-list">
                <div class="empty-state">No audiobooks generated yet</div>
            </div>
        </div>
    </div>

    <div class="footer">
        Credit @TheJoshCode on 
        <a href="https://github.com/TheJoshCode" style="color: white; text-decoration: underline;">
            https://github.com/TheJoshCode
        </a>
    </div>
</div>

<script>
const form = document.getElementById('form');
const prog = document.getElementById('prog');
const status = document.getElementById('status');
const outputsList = document.getElementById('outputs-list');
const submitBtn = document.getElementById('submitBtn');

let currentPlayingAudio = null;
let currentSessionId = null;
let progressInterval = null;

function setStatus(message, type = '') {
    status.textContent = message;
    status.className = 'status ' + type;
}

function loadOutputs() {
    fetch('/outputs')
        .then(res => res.json())
        .then(data => {
            if (!data.files || data.files.length === 0) {
                outputsList.innerHTML = '<div class="empty-state">No audiobooks generated yet</div>';
                return;
            }

            outputsList.innerHTML = '';
            data.files.forEach(file => {
                const item = document.createElement('div');
                item.className = 'output-item';

                const p = document.createElement('p');
                p.textContent = file;
                item.appendChild(p);

                const actions = document.createElement('div');
                actions.className = 'output-actions';

                const playBtn = document.createElement('button');
                playBtn.className = 'toggle-btn';
                playBtn.textContent = '▶ Play';

                const downloadBtn = document.createElement('button');
                downloadBtn.className = 'download-btn';
                downloadBtn.textContent = '⬇ Download';

                const audio = document.createElement('audio');
                audio.src = `/outputs/${file}`;
                audio.preload = 'none';
                audio.style.display = 'none';
                item.appendChild(audio);

                playBtn.addEventListener('click', () => {
                    if (currentPlayingAudio && currentPlayingAudio !== audio && !currentPlayingAudio.paused) {
                        currentPlayingAudio.pause();
                        const prevBtn = document.querySelector('.toggle-btn.playing');
                        if (prevBtn) {
                            prevBtn.textContent = '▶ Play';
                            prevBtn.classList.remove('playing');
                        }
                    }

                    if (audio.paused) {
                        audio.play().catch(err => {
                            console.error('Playback error:', err);
                            setStatus('Playback failed', 'error');
                        });
                        playBtn.textContent = '⏸ Pause';
                        playBtn.classList.add('playing');
                        currentPlayingAudio = audio;
                    } else {
                        audio.pause();
                        playBtn.textContent = '▶ Play';
                        playBtn.classList.remove('playing');
                        currentPlayingAudio = null;
                    }
                });

                audio.addEventListener('ended', () => {
                    playBtn.textContent = '▶ Play';
                    playBtn.classList.remove('playing');
                    currentPlayingAudio = null;
                });

                downloadBtn.addEventListener('click', () => {
                    window.location.href = `/outputs/${file}`;
                });

                actions.appendChild(playBtn);
                actions.appendChild(downloadBtn);
                item.appendChild(actions);
                outputsList.appendChild(item);
            });
        })
        .catch(err => {
            console.error('Failed to load outputs:', err);
            outputsList.innerHTML = '<div class="empty-state">Error loading audiobooks</div>';
        });
}

loadOutputs();

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    
    setStatus('Starting generation...', '');
    prog.value = 0;
    submitBtn.disabled = true;

    const fd = new FormData(form);
    
    try {
        const res = await fetch('/generate', { method: 'POST', body: fd });
        const data = await res.json();

        if (!res.ok || data.error) {
            setStatus(data.error || 'Generation failed', 'error');
            submitBtn.disabled = false;
            return;
        }

        if (data.status === 'started' && data.session_id) {
            currentSessionId = data.session_id;
            setStatus('Processing...', '');
            
            progressInterval = setInterval(async () => {
                try {
                    const pRes = await fetch(`/progress/${currentSessionId}`);
                    const pData = await pRes.json();
                    
                    if (!pRes.ok) {
                        clearInterval(progressInterval);
                        setStatus(pData.error || 'Session error', 'error');
                        submitBtn.disabled = false;
                        return;
                    }
                    
                    prog.value = pData.progress;
                    
                    if (pData.error) {
                        clearInterval(progressInterval);
                        setStatus(pData.error, 'error');
                        submitBtn.disabled = false;
                        return;
                    }

                    if (pData.progress >= 100 && pData.audio_file) {
                        clearInterval(progressInterval);
                        setStatus('✓ Complete! Audiobook ready.', 'success');
                        submitBtn.disabled = false;
                        loadOutputs();
                        
                        document.querySelector('.outputs-panel').scrollIntoView({ 
                            behavior: 'smooth', 
                            block: 'start' 
                        });
                    }
                } catch (err) {
                    console.error('Progress check error:', err);
                    clearInterval(progressInterval);
                    setStatus('Error checking progress', 'error');
                    submitBtn.disabled = false;
                }
            }, 1000);
        }
    } catch (err) {
        console.error('Submission error:', err);
        setStatus('Network error. Please try again.', 'error');
        submitBtn.disabled = false;
    }
});
</script>
</body>
</html>
    """


@app.route('/generate', methods=['POST'])
@limiter.limit("5 per hour")
def start_audiobook_generation():
    if tts_model is None:
        logger.error("Generation requested but model not loaded")
        return jsonify({'error': 'Service unavailable - model not loaded'}), 503

    try:
        # Reference transcript
        ref_text_raw = request.form.get('ref_text', '').strip()
        valid, ref_text_clean = validate_text(ref_text_raw, MAX_REFERENCE_TEXT_LENGTH, "Reference text")
        if not valid:
            return jsonify({'error': ref_text_clean}), 400
        ref_text = ref_text_clean

        # Language
        language = request.form.get('language', 'English').strip()
        if language not in SUPPORTED_LANGUAGES:
            return jsonify({'error': f'Language must be one of: {", ".join(SUPPORTED_LANGUAGES)}'}), 400

        # Book text ── from textarea or uploaded file
        book_text = ""

        if novel_text := request.form.get('novel_text', '').strip():
            valid, cleaned = validate_text(novel_text, MAX_TEXT_CHARACTERS, "Book text")
            if not valid:
                return jsonify({'error': cleaned}), 400
            book_text = cleaned

        elif 'novel_file' in request.files and request.files['novel_file'].filename:
            file = request.files['novel_file']
            valid, msg = is_valid_uploaded_file(file, MAX_BOOK_FILE_SIZE, '.txt')
            if not valid:
                return jsonify({'error': msg}), 400

            try:
                content = file.read().decode('utf-8', errors='ignore').strip()
                if not content:
                    return jsonify({'error': 'Uploaded book file is empty'}), 400
                if len(content) > MAX_TEXT_CHARACTERS:
                    return jsonify({'error': f'Book text too long (max ~500KB)'}), 400
                book_text = content
            except Exception as exc:
                logger.error(f"Failed to read uploaded book file: {exc}")
                return jsonify({'error': 'Failed to read book file'}), 400

        else:
            return jsonify({'error': 'Provide book text or upload a .txt file'}), 400

        ref_audio_path = None
        temp_ref_path = None

        if url := request.form.get('ref_audio_url', '').strip():
            temp_ref_path = os.path.join(TEMP_DIRECTORY, f"ref_{uuid.uuid4().hex}.wav")
            success, msg = download_reference_audio(url, temp_ref_path)
            if not success:
                if os.path.exists(temp_ref_path):
                    os.remove(temp_ref_path)
                return jsonify({'error': msg}), 400
            ref_audio_path = temp_ref_path

        elif 'ref_audio_file' in request.files and request.files['ref_audio_file'].filename:
            file = request.files['ref_audio_file']
            valid, msg = is_valid_uploaded_file(file, MAX_REFERENCE_AUDIO_SIZE, '.wav')
            if not valid:
                return jsonify({'error': msg}), 400

            temp_ref_path = os.path.join(TEMP_DIRECTORY, f"ref_{uuid.uuid4().hex}.wav")
            try:
                file.save(temp_ref_path)
                ref_audio_path = temp_ref_path
            except Exception as exc:
                logger.error(f"Failed to save uploaded reference audio: {exc}")
                if os.path.exists(temp_ref_path):
                    os.remove(temp_ref_path)
                return jsonify({'error': 'Failed to save reference audio file'}), 400

        else:
            return jsonify({'error': 'Provide reference audio URL or upload a .wav file'}), 400

        session = GenerationSession()
        with sessions_lock:
            generation_sessions[session.id] = session

        logger.info(f"New generation session started: {session.id}")

        threading.Thread(
            target=generate_audiobook_in_background,
            args=(book_text, ref_audio_path, ref_text, language, session),
            daemon=True
        ).start()

        return jsonify({'status': 'started', 'session_id': session.id})

    except Exception as exc:
        logger.error(f"Error in /generate endpoint: {exc}", exc_info=True)
        return jsonify({'error': 'Unexpected server error'}), 500


@app.route('/progress/<session_id>')
def get_progress(session_id):
    with sessions_lock:
        session = generation_sessions.get(session_id)

    if not session:
        return jsonify({'error': 'Session not found or expired'}), 404

    return jsonify({
        'progress': session.progress_percentage,
        'audio_file': session.output_filename,
        'error': session.error_message,
        'generating': session.is_generating,
        'estimated_time_remaining': session.estimated_time_remaining,
        'processed_chunks': session.processed_chunks,
        'total_chunks': session.total_chunks
    })


@app.route('/outputs')
def list_output_files():
    try:
        files = [
            f for f in os.listdir(OUTPUT_DIRECTORY)
            if f.lower().endswith('.wav')
        ]
        files.sort(key=lambda x: os.path.getmtime(os.path.join(OUTPUT_DIRECTORY, x)), reverse=True)
        return jsonify({'files': files})
    except Exception as exc:
        logger.error(f"Failed to list output files: {exc}")
        return jsonify({'error': 'Failed to list generated files'}), 500


@app.route('/outputs/<path:filename>')
def serve_output_file(filename):
    safe_filename = secure_filename(filename)

    if not safe_filename or not safe_filename.lower().endswith('.wav'):
        return jsonify({'error': 'Invalid filename or file type'}), 400

    full_path = os.path.join(OUTPUT_DIRECTORY, safe_filename)

    if not os.path.abspath(full_path).startswith(os.path.abspath(OUTPUT_DIRECTORY)):
        logger.warning(f"Possible path traversal attempt: {filename}")
        return jsonify({'error': 'Invalid file path'}), 403

    if not os.path.isfile(full_path):
        return jsonify({'error': 'File not found'}), 404

    logger.info(f"Serving generated file: {safe_filename}")
    return send_from_directory(OUTPUT_DIRECTORY, safe_filename, as_attachment=True)


# ────────────────────────────────────────────────
# Error Handlers
# ────────────────────────────────────────────────

@app.errorhandler(429)
def rate_limit_exceeded(_):
    logger.warning(f"Rate limit hit from {request.remote_addr}")
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429


@app.errorhandler(413)
def payload_too_large(_):
    return jsonify({'error': 'Request too large. Maximum allowed is 100 MiB.'}), 413


@app.errorhandler(500)
def internal_server_error(exc):
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return jsonify({'error': 'An internal server error occurred'}), 500

def open_browser(host, port):
    time.sleep(0.5)
    url = f"http://{host}:{port}"
    webbrowser.open(url)

if __name__ == '__main__':
    if os.environ.get('SECRET_KEY') is None:
        logger.warning("SECRET_KEY not set in environment → using temporary random key")

    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    if debug:
        logger.warning("Running in DEBUG mode – not suitable for production")

    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))

    logger.info(f"Starting Booklet server on {host}:{port}")
    logger.info(f"Output directory: {OUTPUT_DIRECTORY}")
    logger.info(f"Files retained for: {FILE_RETENTION_SECONDS // 3600} hours")

    threading.Thread(
        target=open_browser,
        args=(host, port),
        daemon=True
    ).start()

    app.run(
        debug=debug,
        host=host,
        port=port
    )
