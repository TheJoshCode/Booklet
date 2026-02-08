import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel
from flask import Flask, request, jsonify, send_from_directory, session
from flask_limiter import Limiter
from flask_limiter.util import get_remote_address
from flask_wtf.csrf import CSRFProtect
from werkzeug.utils import secure_filename
import threading
import re
import numpy as np
import os
import sys
import time
from datetime import datetime
import uuid
import logging
from logging.handlers import RotatingFileHandler
from urllib.parse import urlparse
import socket
import ipaddress
import requests
import webbrowser
import fitz
import math
from threading import Lock
from contextlib import contextmanager
from functools import wraps
import pickle
import json
from pathlib import Path

# ────────────────────────────────────────────────
# UTF-8 Encoding Fix for Windows
# ────────────────────────────────────────────────
# Force UTF-8 encoding for stdout/stderr on Windows to prevent UnicodeEncodeError
if sys.platform == 'win32':
    try:
        sys.stdout.reconfigure(encoding='utf-8')
        sys.stderr.reconfigure(encoding='utf-8')
    except AttributeError:
        # Python < 3.7 fallback
        import codecs
        sys.stdout = codecs.getwriter('utf-8')(sys.stdout.buffer, 'strict')
        sys.stderr = codecs.getwriter('utf-8')(sys.stderr.buffer, 'strict')

# ────────────────────────────────────────────────
# Logging Stuff with UTF-8 Support
# ────────────────────────────────────────────────
class SafeFormatter(logging.Formatter):
    """Custom formatter that handles Unicode properly"""
    def format(self, record):
        try:
            return super().format(record)
        except UnicodeEncodeError:
            # Fallback to ASCII if Unicode fails
            record.msg = str(record.msg).encode('ascii', errors='replace').decode('ascii')
            return super().format(record)

# Create formatter
formatter = SafeFormatter(
    '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)

# File handler with UTF-8 encoding and rotation
file_handler = RotatingFileHandler(
    'booklet.log',
    maxBytes=10*1024*1024,  # 10 MB
    backupCount=5,
    encoding='utf-8'
)
file_handler.setFormatter(formatter)

# Stream handler with UTF-8 encoding
stream_handler = logging.StreamHandler(sys.stdout)
stream_handler.setFormatter(formatter)

# Configure root logger
logging.basicConfig(
    level=logging.INFO,
    handlers=[file_handler, stream_handler]
)

logger = logging.getLogger(__name__)

def safe_log(message, *args, **kwargs):
    """Safe logging function that handles Unicode characters"""
    try:
        logger.info(message, *args, **kwargs)
    except UnicodeEncodeError:
        # Sanitize message and retry
        safe_message = str(message).encode('ascii', errors='replace').decode('ascii')
        logger.info(safe_message, *args, **kwargs)

# ────────────────────────────────────────────────
# Flask App Conf Stuff
# ────────────────────────────────────────────────
app = Flask(__name__)

# Generate a secure random key if not provided
SECRET_KEY = os.environ.get('SECRET_KEY')
if not SECRET_KEY:
    SECRET_KEY = os.urandom(32).hex()
    logger.warning("SECRET_KEY not set in environment - using temporary random key")

app.config['SECRET_KEY'] = SECRET_KEY
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100 MiB max request size

# CSRF disabled for local development
app.config['WTF_CSRF_ENABLED'] = False
logger.warning("CSRF protection DISABLED - local development mode only")

csrf = CSRFProtect(app)

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
CACHE_DIRECTORY  = os.path.abspath('cache')  # For progress caching

try:
    os.makedirs(OUTPUT_DIRECTORY, exist_ok=True)
    os.makedirs(TEMP_DIRECTORY,   exist_ok=True)
    os.makedirs(CACHE_DIRECTORY,  exist_ok=True)
except Exception as e:
    logger.error(f"Failed to create directories: {e}")
    raise

# ────────────────────────────────────────────────
# Constants & Limits
# ────────────────────────────────────────────────
MAX_TEXT_CHARACTERS         = float('inf')  # unlimited text length
MAX_REFERENCE_TEXT_LENGTH   = 1000
MAX_REFERENCE_AUDIO_SIZE    = 10 * 1024 * 1024   # 10 MiB
MAX_BOOK_FILE_SIZE          = 10 * 1024 * 1024   # 10 MiB
FILE_RETENTION_SECONDS      = float('inf')  # keep files forever
MAX_URL_DOWNLOAD_SIZE       = 15 * 1024 * 1024   # 15 MiB for URL downloads
DOWNLOAD_TIMEOUT            = 30  # seconds

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
# Load the Model with Flash Attention 2
# ────────────────────────────────────────────────
tts_model = None
model_lock = Lock()

try:
    logger.info("Loading Qwen3-TTS model...")
    tts_model = Qwen3TTSModel.from_pretrained(
        "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
    )
    logger.info("Model loaded successfully")
except Exception as exc:
    logger.error(f"Failed to load model: {exc}", exc_info=True)
    tts_model = None

# ────────────────────────────────────────────────
# Dynamic VRAM-aware chunk sizing
# ────────────────────────────────────────────────
def get_max_chars_per_chunk():
    """Returns safe max characters per generate_voice_clone call based on VRAM"""
    if not torch.cuda.is_available():
        logger.warning("No CUDA detected - using conservative chunk size")
        return 700

    try:
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)
        device_name = torch.cuda.get_device_properties(0).name

        # Conservative but aggressive scaling (leaves ~1.5-2.5 GB headroom)
        if vram_gb <= 9:        # 6-8 GB cards (RTX 3060, laptop GPUs)
            return 650
        elif vram_gb <= 13:     # 10-12 GB (RTX 3080, 4070)
            return 1400
        elif vram_gb <= 17:     # 16 GB (RTX 3090 Ti, 4080)
            return 2300
        elif vram_gb <= 25:     # 24 GB (RTX 4090, A6000)
            return 4000
        else:                   # 32 GB+ (A100, H100, etc.)
            return 6000

    except Exception as e:
        logger.warning(f"VRAM detection failed: {e} - falling back to safe default")
        return 800


MAX_CHARS_PER_CHUNK = get_max_chars_per_chunk()

if torch.cuda.is_available():
    try:
        device_name = torch.cuda.get_device_properties(0).name
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        logger.info(f"VRAM-aware chunking enabled - max ~{MAX_CHARS_PER_CHUNK} characters per batch "
                    f"on {device_name} ({vram_gb:.1f} GB VRAM)")
    except Exception as e:
        logger.warning(f"Could not log VRAM info: {e}")
else:
    logger.info(f"CPU mode - max ~{MAX_CHARS_PER_CHUNK} characters per batch")

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
        
        # New fields for pause/resume
        self.is_paused = False
        self.pause_requested = False
        self.can_resume = False
        
        # Cache information
        self.book_text = None
        self.ref_audio = None
        self.ref_text = None
        self.language = None
        self.chunks = None
        self.audio_segments = []
        
        # Metadata
        self.last_updated = time.time()
        self.pause_time = None
        self.total_paused_duration = 0
    
    def to_dict(self):
        """Convert session to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'progress_percentage': self.progress_percentage,
            'output_filename': self.output_filename,
            'is_generating': self.is_generating,
            'error_message': self.error_message,
            'created_timestamp': self.created_timestamp,
            'start_time': self.start_time,
            'estimated_time_remaining': self.estimated_time_remaining,
            'total_chunks': self.total_chunks,
            'processed_chunks': self.processed_chunks,
            'is_paused': self.is_paused,
            'pause_requested': self.pause_requested,
            'can_resume': self.can_resume,
            'last_updated': self.last_updated,
            'pause_time': self.pause_time,
            'total_paused_duration': self.total_paused_duration,
        }
    
    def save_cache(self):
        """Save session progress to disk"""
        try:
            cache_file = os.path.join(CACHE_DIRECTORY, f"{self.id}.cache")
            
            # Save metadata as JSON
            metadata = self.to_dict()
            metadata['ref_text'] = self.ref_text
            metadata['language'] = self.language
            metadata['total_chunks'] = self.total_chunks
            metadata['processed_chunks'] = self.processed_chunks
            
            with open(cache_file + '.json', 'w', encoding='utf-8') as f:
                json.dump(metadata, f, indent=2)
            
            # Save binary data with pickle
            cache_data = {
                'chunks': self.chunks,
                'audio_segments': self.audio_segments,
                'book_text': self.book_text,
                'ref_audio': self.ref_audio,
            }
            
            with open(cache_file + '.pkl', 'wb') as f:
                pickle.dump(cache_data, f, protocol=pickle.HIGHEST_PROTOCOL)
            
            self.last_updated = time.time()
            safe_log(f"Session {self.id}: Progress cached successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to save cache for session {self.id}: {e}", exc_info=True)
            return False
    
    def load_cache(self):
        """Load session progress from disk"""
        try:
            cache_file = os.path.join(CACHE_DIRECTORY, f"{self.id}.cache")
            
            # Load metadata
            with open(cache_file + '.json', 'r', encoding='utf-8') as f:
                metadata = json.load(f)
            
            self.ref_text = metadata.get('ref_text')
            self.language = metadata.get('language')
            self.total_chunks = metadata.get('total_chunks', 0)
            self.processed_chunks = metadata.get('processed_chunks', 0)
            self.progress_percentage = metadata.get('progress_percentage', 0)
            self.created_timestamp = metadata.get('created_timestamp', time.time())
            self.start_time = metadata.get('start_time')
            self.total_paused_duration = metadata.get('total_paused_duration', 0)
            
            # Load binary data
            with open(cache_file + '.pkl', 'rb') as f:
                cache_data = pickle.load(f)
            
            self.chunks = cache_data.get('chunks')
            self.audio_segments = cache_data.get('audio_segments', [])
            self.book_text = cache_data.get('book_text')
            self.ref_audio = cache_data.get('ref_audio')
            
            self.can_resume = True
            safe_log(f"Session {self.id}: Cache loaded successfully ({self.processed_chunks}/{self.total_chunks} chunks)")
            return True
            
        except Exception as e:
            logger.error(f"Failed to load cache for session {self.id}: {e}", exc_info=True)
            return False
    
    def delete_cache(self):
        """Delete cache files for this session"""
        try:
            cache_file = os.path.join(CACHE_DIRECTORY, f"{self.id}.cache")
            
            for ext in ['.json', '.pkl']:
                path = cache_file + ext
                if os.path.exists(path):
                    os.remove(path)
                    
            safe_log(f"Session {self.id}: Cache deleted")
            
        except Exception as e:
            logger.error(f"Failed to delete cache for session {self.id}: {e}")
    
    @staticmethod
    def list_cached_sessions():
        """List all sessions with cached progress"""
        try:
            cached = []
            for filename in os.listdir(CACHE_DIRECTORY):
                if filename.endswith('.json'):
                    session_id = filename.replace('.cache.json', '')
                    cache_file = os.path.join(CACHE_DIRECTORY, f"{session_id}.cache.json")
                    
                    try:
                        with open(cache_file, 'r', encoding='utf-8') as f:
                            metadata = json.load(f)
                        cached.append(metadata)
                    except:
                        pass
                        
            return cached
        except Exception as e:
            logger.error(f"Failed to list cached sessions: {e}")
            return []


@contextmanager
def session_context(session_id):
    """Context manager for safe session access"""
    with sessions_lock:
        session = generation_sessions.get(session_id)
        yield session


def remove_expired_sessions():
    """Sessions never expire - disabled cleanup"""
    pass  # Disabled - sessions kept indefinitely


def remove_old_files():
    """File cleanup disabled - files kept indefinitely"""
    pass  # Disabled - keep all files


def periodic_cleanup_task():
    """Cleanup disabled - no automatic file/session removal"""
    while True:
        time.sleep(3600)  # Sleep forever, do nothing
        # Cleanup disabled to keep all files and sessions indefinitely


# Cleanup thread disabled - keeping all files and sessions
# threading.Thread(target=periodic_cleanup_task, daemon=True).start()
logger.info("Automatic cleanup disabled - files and sessions kept indefinitely")


# ────────────────────────────────────────────────
# Security Defenses
# ────────────────────────────────────────────────

def is_private_ip_address(ip_str: str) -> bool:
    """Check if IP is in private ranges"""
    try:
        ip = ipaddress.ip_address(ip_str)
        return any(ip in net for net in PRIVATE_IP_NETWORKS)
    except ValueError:
        return False


def is_safe_url(url: str) -> tuple[bool, str]:
    """
    Validate URL safety (no SSRF, safe scheme, etc.)
    Returns: (is_valid, error_message)
    """
    try:
        if not url or len(url) > 2048:
            return False, "Invalid URL length"

        parsed = urlparse(url)

        # Check scheme
        if parsed.scheme not in ALLOWED_URL_SCHEMES:
            return False, f"Unsupported URL scheme: {parsed.scheme}"

        # Check hostname
        hostname = parsed.hostname
        if not hostname:
            return False, "No hostname in URL"

        hostname_lower = hostname.lower()
        if hostname_lower in BLOCKED_HOSTNAMES:
            return False, "Blocked hostname"

        # Resolve and check IP
        try:
            ip = socket.gethostbyname(hostname)
            if is_private_ip_address(ip):
                return False, "Private IP addresses not allowed"
        except socket.gaierror:
            return False, "Could not resolve hostname"

        return True, ""

    except Exception as e:
        logger.error(f"Error validating URL: {e}")
        return False, "Invalid URL format"


def download_reference_audio(url: str, save_path: str) -> tuple[bool, str]:
    """
    Securely download reference audio from URL
    Returns: (success, error_message)
    """
    try:
        # Validate URL
        is_valid, error_msg = is_safe_url(url)
        if not is_valid:
            return False, error_msg

        # Download with safety limits
        response = requests.get(
            url,
            stream=True,
            timeout=DOWNLOAD_TIMEOUT,
            allow_redirects=True,
            headers={'User-Agent': 'Booklet-TTS/1.0'}
        )
        response.raise_for_status()

        # Check content type
        content_type = response.headers.get('Content-Type', '').lower()
        if 'audio' not in content_type and 'wav' not in content_type:
            logger.warning(f"Unexpected content type: {content_type}")

        # Stream download with size check
        downloaded_size = 0
        with open(save_path, 'wb') as f:
            for chunk in response.iter_content(chunk_size=8192):
                if chunk:
                    downloaded_size += len(chunk)
                    if downloaded_size > MAX_URL_DOWNLOAD_SIZE:
                        os.remove(save_path)
                        return False, f"File too large (max {MAX_URL_DOWNLOAD_SIZE // (1024*1024)} MB)"
                    f.write(chunk)

        # Validate it's a valid WAV file
        try:
            audio_data, sample_rate = sf.read(save_path)
            if len(audio_data) == 0:
                os.remove(save_path)
                return False, "Downloaded audio file is empty"
        except Exception as e:
            if os.path.exists(save_path):
                os.remove(save_path)
            return False, f"Invalid audio file: {str(e)}"

        logger.info(f"Successfully downloaded reference audio: {downloaded_size} bytes")
        return True, ""

    except requests.exceptions.Timeout:
        return False, "Download timeout"
    except requests.exceptions.RequestException as e:
        return False, f"Download failed: {str(e)}"
    except Exception as e:
        logger.error(f"Error downloading reference audio: {e}", exc_info=True)
        if os.path.exists(save_path):
            try:
                os.remove(save_path)
            except:
                pass
        return False, "Download failed"


def is_valid_uploaded_file(file, max_size: int, required_extension: str = None) -> tuple[bool, str]:
    """
    Validate uploaded file
    Returns: (is_valid, error_message)
    """
    try:
        if not file or not file.filename:
            return False, "No file provided"

        # Check filename
        filename = secure_filename(file.filename)
        if not filename:
            return False, "Invalid filename"

        # Check extension
        if required_extension:
            if not filename.lower().endswith(required_extension.lower()):
                return False, f"File must be {required_extension}"

        # Check size by seeking to end
        file.seek(0, os.SEEK_END)
        size = file.tell()
        file.seek(0)

        if size > max_size:
            return False, f"File too large (max {max_size // (1024*1024)} MB)"

        if size == 0:
            return False, "File is empty"

        return True, ""

    except Exception as e:
        logger.error(f"Error validating file: {e}")
        return False, "File validation failed"


def sanitize_text_input(text: str, max_length: int = None) -> tuple[bool, str]:
    """
    Sanitize and validate text input
    Returns: (is_valid, cleaned_text_or_error_message)
    """
    try:
        if not text:
            return False, "No text provided"

        # Remove null bytes
        text = text.replace('\x00', '')

        # Normalize whitespace
        text = re.sub(r'\r\n', '\n', text)
        text = re.sub(r'\r', '\n', text)

        # Remove excessive newlines
        text = re.sub(r'\n{4,}', '\n\n\n', text)

        # Strip
        text = text.strip()

        if not text:
            return False, "Text is empty after cleaning"

        # Check length
        if max_length and len(text) > max_length:
            return False, f"Text too long (max {max_length:,} characters)"

        return True, text

    except Exception as e:
        logger.error(f"Error sanitizing text: {e}")
        return False, "Text sanitization failed"


# ────────────────────────────────────────────────
# Text Chunking with Sentence Boundaries
# ────────────────────────────────────────────────

def split_into_sentences(text: str) -> list[str]:
    """
    Split text into sentences safely
    """
    try:
        # Handle common abbreviations
        text = re.sub(r'\bDr\.', 'Dr<DOT>', text)
        text = re.sub(r'\bMr\.', 'Mr<DOT>', text)
        text = re.sub(r'\bMrs\.', 'Mrs<DOT>', text)
        text = re.sub(r'\bMs\.', 'Ms<DOT>', text)
        text = re.sub(r'\bProf\.', 'Prof<DOT>', text)
        text = re.sub(r'\b([A-Z])\.', r'\1<DOT>', text)

        # Split on sentence boundaries
        pattern = r'(?<=[.!?])\s+'
        sentences = re.split(pattern, text)

        # Restore abbreviations
        sentences = [s.replace('<DOT>', '.') for s in sentences]

        # Filter empty
        sentences = [s.strip() for s in sentences if s.strip()]

        return sentences

    except Exception as e:
        logger.error(f"Error splitting sentences: {e}")
        # Fallback: split by periods
        return [s.strip() + '.' for s in text.split('.') if s.strip()]


def chunk_text_by_sentences(text: str, max_chars: int) -> list[str]:
    """
    Split text into chunks at sentence boundaries
    """
    try:
        sentences = split_into_sentences(text)
        chunks = []
        current_chunk = []
        current_length = 0

        for sentence in sentences:
            sentence_length = len(sentence)

            # If single sentence exceeds limit, split it
            if sentence_length > max_chars:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                    current_chunk = []
                    current_length = 0

                # Split long sentence by words
                words = sentence.split()
                temp_chunk = []
                temp_length = 0

                for word in words:
                    word_length = len(word) + 1  # +1 for space
                    if temp_length + word_length > max_chars:
                        if temp_chunk:
                            chunks.append(' '.join(temp_chunk))
                        temp_chunk = [word]
                        temp_length = len(word)
                    else:
                        temp_chunk.append(word)
                        temp_length += word_length

                if temp_chunk:
                    chunks.append(' '.join(temp_chunk))
                continue

            # Add sentence to current chunk
            if current_length + sentence_length + 1 <= max_chars:
                current_chunk.append(sentence)
                current_length += sentence_length + 1
            else:
                if current_chunk:
                    chunks.append(' '.join(current_chunk))
                current_chunk = [sentence]
                current_length = sentence_length

        # Add remaining chunk
        if current_chunk:
            chunks.append(' '.join(current_chunk))

        return chunks

    except Exception as e:
        logger.error(f"Error chunking text: {e}", exc_info=True)
        # Fallback: simple character-based chunking
        return [text[i:i+max_chars] for i in range(0, len(text), max_chars)]


# ────────────────────────────────────────────────
# TTS Generation with Safety
# ────────────────────────────────────────────────

def safe_generate_voice(text: str, ref_audio: str, ref_text: str, language: str):
    """
    Safely generate voice with error handling
    Returns: audio_array or None
    """
    try:
        if not tts_model:
            logger.error("TTS model not loaded")
            return None

        with model_lock:
            result = tts_model.generate_voice_clone(
                text=text,
                ref_audio=ref_audio,
                ref_text=ref_text,
                language=language,
                max_new_tokens=4096,
                do_sample=True,
                temperature=0.7,
                top_k=50,
                top_p=0.9
            )

        return result

    except torch.cuda.OutOfMemoryError:
        logger.error("CUDA out of memory during generation")
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return None
    except Exception as e:
        logger.error(f"Error in voice generation: {e}", exc_info=True)
        return None


def generate_audiobook_in_background(book_text: str, ref_audio: str, 
                                     ref_text: str, language: str, session: GenerationSession,
                                     resume: bool = False):
    """
    Background task for audiobook generation with pause/resume support
    """
    temp_ref_cleanup = ref_audio if ref_audio and ref_audio.startswith(TEMP_DIRECTORY) else None
    
    try:
        session.is_generating = True
        session.is_paused = False
        
        if not tts_model:
            session.error_message = "TTS model not loaded"
            session.is_generating = False
            return

        # Store generation parameters in session
        session.book_text = book_text
        session.ref_audio = ref_audio
        session.ref_text = ref_text
        session.language = language

        # Resume from cache or start fresh
        if resume and session.can_resume:
            safe_log(f"Session {session.id}: Resuming from chunk {session.processed_chunks}/{session.total_chunks}")
            chunks = session.chunks
            audio_segments = session.audio_segments
            chunk_count = session.processed_chunks
            
            # Adjust start time accounting for paused duration
            if session.start_time:
                session.start_time = time.time() - (session.processed_chunks / session.total_chunks) * (time.time() - session.start_time)
        else:
            # Start fresh
            if not session.start_time:
                session.start_time = time.time()
            
            safe_log(f"Session {session.id}: Starting text chunking...")
            chunks = chunk_text_by_sentences(book_text, MAX_CHARS_PER_CHUNK)
            session.chunks = chunks
            session.total_chunks = len(chunks)
            safe_log(f"Session {session.id}: Split into {len(chunks)} chunks")

            if not chunks:
                session.error_message = "Failed to chunk text"
                session.is_generating = False
                return

            audio_segments = []
            chunk_count = 0

        # Process remaining chunks
        for i in range(chunk_count, len(chunks)):
            # Check for pause request
            if session.pause_requested:
                session.is_paused = True
                session.is_generating = False
                session.pause_requested = False
                session.pause_time = time.time()
                session.can_resume = True
                
                # Save progress
                session.audio_segments = audio_segments
                session.processed_chunks = chunk_count
                session.save_cache()
                
                safe_log(f"Session {session.id}: Paused at chunk {chunk_count}/{len(chunks)}")
                return
            
            try:
                chunk = chunks[i]
                chunk_count += 1
                current_chars = len(chunk)
                num_sentences = len(split_into_sentences(chunk))

                # Safe logging with ASCII fallback
                try:
                    safe_log(f"Session {session.id}: Chunk {chunk_count} - {num_sentences} sentences (~{current_chars:,} chars)")
                except:
                    logger.info(f"Session {session.id}: Chunk {chunk_count} - {num_sentences} sentences (~{current_chars} chars)")

                # Generate audio
                audio = safe_generate_voice(chunk, ref_audio, ref_text, language)
                audio = np.asarray(audio)

                # If stereo, convert to mono
                if audio.ndim == 2:
                    audio = audio.mean(axis=1)

                # If shape (1, N) or (N, 1)
                audio = audio.squeeze()

                # Ensure float32 for consistency
                audio = audio.astype(np.float32)

                if audio is None:
                    raise Exception("Voice generation returned None")

                audio_segments.append(audio)
                session.processed_chunks = chunk_count
                session.audio_segments = audio_segments

                # Update progress
                session.progress_percentage = int((chunk_count / len(chunks)) * 100)

                # Estimate time remaining
                elapsed = time.time() - session.start_time
                chunks_remaining = len(chunks) - chunk_count
                if chunk_count > 0:
                    avg_time_per_chunk = elapsed / chunk_count
                    session.estimated_time_remaining = int(avg_time_per_chunk * chunks_remaining)

                # Save progress periodically (every 10 chunks)
                if chunk_count % 10 == 0:
                    session.save_cache()

                # Memory cleanup
                if torch.cuda.is_available() and i % 10 == 0:
                    torch.cuda.empty_cache()

            except Exception as chunk_error:
                logger.error(f"Session {session.id}: Error processing chunk {chunk_count}: {chunk_error}", exc_info=True)
                
                # Save progress before erroring out
                session.audio_segments = audio_segments
                session.processed_chunks = chunk_count
                session.save_cache()
                
                session.error_message = f"Failed at chunk {chunk_count}: {str(chunk_error)}"
                session.is_generating = False
                session.can_resume = True
                return

        # Concatenate audio segments
        safe_log(f"Session {session.id}: Concatenating {len(audio_segments)} audio segments...")
        
        try:
            audio_segments = [np.asarray(a).squeeze().astype(np.float32) for a in audio_segments]
            full_audio = np.concatenate(audio_segments)
        except Exception as concat_error:
            logger.error(f"Session {session.id}: Error concatenating audio: {concat_error}")
            session.error_message = "Failed to concatenate audio segments"
            session.is_generating = False
            return

        # Save output
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"audiobook_{timestamp}_{session.id[:8]}.wav"
        output_path = os.path.join(OUTPUT_DIRECTORY, output_filename)

        try:
            sf.write(output_path, full_audio, samplerate=24000)
            safe_log(f"Session {session.id}: Saved output to {output_filename}")
        except Exception as save_error:
            logger.error(f"Session {session.id}: Error saving audio: {save_error}")
            session.error_message = "Failed to save audio file"
            session.is_generating = False
            return

        # Success
        session.output_filename = output_filename
        session.progress_percentage = 100
        session.is_generating = False
        session.can_resume = False
        
        duration_minutes = (time.time() - session.start_time) / 60
        safe_log(f"Session {session.id}: Completed in {duration_minutes:.1f} minutes")
        
        # Delete cache after successful completion
        session.delete_cache()

    except Exception as exc:
        logger.error(f"Session {session.id}: Unexpected error in background generation: {exc}", exc_info=True)
        session.error_message = f"Unexpected error: {str(exc)}"
        session.is_generating = False
        
        # Save progress on error
        try:
            session.save_cache()
            session.can_resume = True
        except:
            pass

    finally:
        # Cleanup temporary reference audio
        if temp_ref_cleanup and os.path.exists(temp_ref_cleanup):
            try:
                os.remove(temp_ref_cleanup)
                logger.info(f"Cleaned up temporary reference audio: {temp_ref_cleanup}")
            except Exception as cleanup_error:
                logger.error(f"Failed to cleanup temp file {temp_ref_cleanup}: {cleanup_error}")

        # Memory cleanup
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


# ────────────────────────────────────────────────
# Flask Routes
# ────────────────────────────────────────────────

@app.route('/')
def index():
    """Serve the frontend"""
    try:
        return send_from_directory('.', 'index.html')
    except Exception as e:
        logger.error(f"Error serving index: {e}")
        return jsonify({'error': 'Failed to load interface'}), 500


@app.route('/health')
def health():
    """Health check endpoint"""
    try:
        status = {
            'status': 'healthy',
            'model_loaded': tts_model is not None,
            'cuda_available': torch.cuda.is_available(),
            'active_sessions': len(generation_sessions)
        }
        
        if torch.cuda.is_available():
            try:
                status['gpu'] = torch.cuda.get_device_name(0)
                status['vram_gb'] = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
            except:
                pass
                
        return jsonify(status)
    except Exception as e:
        logger.error(f"Health check error: {e}")
        return jsonify({'status': 'error', 'error': str(e)}), 500


@app.route('/generate', methods=['POST'])
@limiter.limit("10 per minute")
def generate_audiobook():
    """
    Main endpoint to generate audiobook with comprehensive validation
    """
    try:
        if not tts_model:
            return jsonify({'error': 'TTS model not available'}), 503

        # Validate and get language
        language = request.form.get('language', 'English').strip()
        if language not in SUPPORTED_LANGUAGES:
            return jsonify({'error': f'Unsupported language. Must be one of: {", ".join(SUPPORTED_LANGUAGES)}'}), 400

        # Validate and get reference text
        ref_text = request.form.get('ref_text', '').strip()
        if not ref_text:
            return jsonify({'error': 'Reference text is required'}), 400

        valid, cleaned_ref = sanitize_text_input(ref_text, MAX_REFERENCE_TEXT_LENGTH)
        if not valid:
            return jsonify({'error': f'Invalid reference text: {cleaned_ref}'}), 400
        ref_text = cleaned_ref

        # Get book text (from textarea or file)
        book_text = None
        source = None

        textarea_content = request.form.get('book_text', '').strip()
        
        if textarea_content:
            valid, cleaned = sanitize_text_input(textarea_content, MAX_TEXT_CHARACTERS)
            if not valid:
                return jsonify({'error': cleaned}), 400
            book_text = cleaned
            source = "textarea"

        else:
            # No text in textarea - must have valid file
            if 'novel_file' not in request.files or not request.files['novel_file'].filename:
                return jsonify({'error': 'Please provide book text in the box or upload a .txt / .pdf file'}), 400

            file = request.files['novel_file']
            valid, msg = is_valid_uploaded_file(file, MAX_BOOK_FILE_SIZE, None)
            if not valid:
                return jsonify({'error': msg}), 400

            filename_lower = file.filename.lower()
            
            try:
                if filename_lower.endswith('.pdf'):
                    stream = file.read()
                    
                    try:
                        doc = fitz.open(stream=stream, filetype="pdf")
                    except Exception as pdf_error:
                        logger.error(f"PDF open error: {pdf_error}")
                        return jsonify({'error': 'Invalid or corrupted PDF file'}), 400

                    if doc.needs_pass:
                        doc.close()
                        return jsonify({'error': 'Encrypted/protected PDF - cannot extract text'}), 400

                    pages_text = []
                    for page_num, page in enumerate(doc):
                        try:
                            text = page.get_text("text").strip()
                            if text:
                                pages_text.append(text)
                        except Exception as page_error:
                            logger.warning(f"Error extracting page {page_num}: {page_error}")
                            
                    doc.close()

                    if not pages_text:
                        return jsonify({'error': 'No readable text found in PDF'}), 400

                    book_text = "\n\n".join(pages_text)
                    source = "PDF"

                elif filename_lower.endswith('.txt'):
                    try:
                        content = file.read().decode('utf-8', errors='ignore').strip()
                    except Exception as decode_error:
                        logger.error(f"Text decode error: {decode_error}")
                        return jsonify({'error': 'Failed to decode text file'}), 400
                        
                    if not content:
                        return jsonify({'error': 'Uploaded .txt file is empty'}), 400
                    book_text = content
                    source = "TXT"

                else:
                    return jsonify({'error': 'Unsupported file type - only .txt and .pdf allowed'}), 400

            except fitz.FileDataError:
                return jsonify({'error': 'Invalid or corrupted PDF file'}), 400
            except Exception as exc:
                logger.error(f"Failed to process uploaded book file: {exc}", exc_info=True)
                return jsonify({'error': 'Failed to read/process book file'}), 400

        if not book_text:
            return jsonify({'error': 'No book content provided'}), 400

        safe_log(f"Book content source: {source} ({len(book_text):,} characters)")

        # Handle reference audio (URL or upload)
        ref_audio = None
        temp_ref_path = None

        url = request.form.get('ref_audio_url', '').strip()
        
        if url:
            temp_ref_path = os.path.join(TEMP_DIRECTORY, f"ref_{uuid.uuid4().hex}.wav")
            success, msg = download_reference_audio(url, temp_ref_path)
            if not success:
                if os.path.exists(temp_ref_path):
                    try:
                        os.remove(temp_ref_path)
                    except:
                        pass
                return jsonify({'error': msg}), 400
            ref_audio = temp_ref_path

        elif 'ref_audio_file' in request.files and request.files['ref_audio_file'].filename:
            file = request.files['ref_audio_file']
            valid, msg = is_valid_uploaded_file(file, MAX_REFERENCE_AUDIO_SIZE, '.wav')
            if not valid:
                return jsonify({'error': msg}), 400

            temp_ref_path = os.path.join(TEMP_DIRECTORY, f"ref_{uuid.uuid4().hex}.wav")
            try:
                file.save(temp_ref_path)
                
                # Validate saved file
                try:
                    audio_data, sample_rate = sf.read(temp_ref_path)
                    if len(audio_data) == 0:
                        raise ValueError("Audio file is empty")
                except Exception as audio_error:
                    os.remove(temp_ref_path)
                    return jsonify({'error': f'Invalid audio file: {str(audio_error)}'}), 400
                    
                ref_audio = temp_ref_path
            except Exception as exc:
                logger.error(f"Failed to save uploaded reference audio: {exc}")
                if os.path.exists(temp_ref_path):
                    try:
                        os.remove(temp_ref_path)
                    except:
                        pass
                return jsonify({'error': 'Failed to save reference audio file'}), 400

        else:
            return jsonify({'error': 'Provide reference audio URL or upload a .wav file'}), 400

        # Create session
        session = GenerationSession()
        with sessions_lock:
            generation_sessions[session.id] = session

        safe_log(f"New generation session started: {session.id}")

        # Start background generation
        threading.Thread(
            target=generate_audiobook_in_background,
            args=(book_text, ref_audio, ref_text, language, session),
            daemon=True
        ).start()

        return jsonify({'status': 'started', 'session_id': session.id})

    except Exception as exc:
        logger.error(f"Error in /generate endpoint: {exc}", exc_info=True)
        return jsonify({'error': 'Unexpected server error'}), 500


@app.route('/progress/<session_id>')
def get_progress(session_id):
    """Get progress for a session"""
    try:
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
            'total_chunks': session.total_chunks,
            'is_paused': session.is_paused,
            'can_resume': session.can_resume,
            'pause_requested': session.pause_requested,
        })
    except Exception as e:
        logger.error(f"Error getting progress: {e}")
        return jsonify({'error': 'Failed to get progress'}), 500


@app.route('/pause/<session_id>', methods=['POST'])
def pause_generation(session_id):
    """Pause an active generation"""
    try:
        with sessions_lock:
            session = generation_sessions.get(session_id)

        if not session:
            return jsonify({'error': 'Session not found'}), 404

        if not session.is_generating:
            return jsonify({'error': 'Session is not currently generating'}), 400

        if session.is_paused:
            return jsonify({'error': 'Session is already paused'}), 400

        session.pause_requested = True
        safe_log(f"Session {session.id}: Pause requested")

        return jsonify({
            'status': 'pause_requested',
            'message': 'Generation will pause after current chunk completes'
        })

    except Exception as e:
        logger.error(f"Error pausing session: {e}")
        return jsonify({'error': 'Failed to pause generation'}), 500


@app.route('/resume/<session_id>', methods=['POST'])
def resume_generation(session_id):
    """Resume a paused generation"""
    try:
        with sessions_lock:
            session = generation_sessions.get(session_id)

        if not session:
            # Try to load from cache
            session = GenerationSession()
            session.id = session_id
            
            if not session.load_cache():
                return jsonify({'error': 'Session not found and no cached progress available'}), 404
            
            generation_sessions[session.id] = session

        if session.is_generating:
            return jsonify({'error': 'Session is already generating'}), 400

        if not session.can_resume:
            return jsonify({'error': 'Session cannot be resumed (may be completed or have no cached progress)'}), 400

        # Resume generation
        safe_log(f"Session {session.id}: Resuming generation")
        
        # Calculate total paused time
        if session.pause_time:
            session.total_paused_duration += time.time() - session.pause_time
            session.pause_time = None

        threading.Thread(
            target=generate_audiobook_in_background,
            args=(
                session.book_text,
                session.ref_audio,
                session.ref_text,
                session.language,
                session
            ),
            kwargs={'resume': True},
            daemon=True
        ).start()

        return jsonify({
            'status': 'resumed',
            'session_id': session.id,
            'progress': session.progress_percentage,
            'processed_chunks': session.processed_chunks,
            'total_chunks': session.total_chunks
        })

    except Exception as e:
        logger.error(f"Error resuming session: {e}", exc_info=True)
        return jsonify({'error': 'Failed to resume generation'}), 500


@app.route('/sessions/cached')
def list_cached_sessions():
    """List all sessions with cached progress"""
    try:
        cached = GenerationSession.list_cached_sessions()
        return jsonify({'sessions': cached})
    except Exception as e:
        logger.error(f"Error listing cached sessions: {e}")
        return jsonify({'error': 'Failed to list cached sessions'}), 500


@app.route('/sessions/<session_id>/cache', methods=['DELETE'])
def delete_session_cache(session_id):
    """Delete cached progress for a session"""
    try:
        session = GenerationSession()
        session.id = session_id
        session.delete_cache()
        
        return jsonify({'status': 'deleted', 'message': 'Cache deleted successfully'})
    except Exception as e:
        logger.error(f"Error deleting cache: {e}")
        return jsonify({'error': 'Failed to delete cache'}), 500


@app.route('/outputs')
def list_output_files():
    """List generated audio files"""
    try:
        if not os.path.exists(OUTPUT_DIRECTORY):
            return jsonify({'files': []})
            
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
    """Serve a generated audio file"""
    try:
        safe_filename = secure_filename(filename)

        if not safe_filename or not safe_filename.lower().endswith('.wav'):
            return jsonify({'error': 'Invalid filename or file type'}), 400

        full_path = os.path.join(OUTPUT_DIRECTORY, safe_filename)

        # Path traversal protection
        if not os.path.abspath(full_path).startswith(os.path.abspath(OUTPUT_DIRECTORY)):
            logger.warning(f"Possible path traversal attempt: {filename}")
            return jsonify({'error': 'Invalid file path'}), 403

        if not os.path.isfile(full_path):
            return jsonify({'error': 'File not found'}), 404

        logger.info(f"Serving generated file: {safe_filename}")
        return send_from_directory(OUTPUT_DIRECTORY, safe_filename, as_attachment=True)
        
    except Exception as e:
        logger.error(f"Error serving file: {e}")
        return jsonify({'error': 'Failed to serve file'}), 500


# ────────────────────────────────────────────────
# Error Handlers
# ────────────────────────────────────────────────

@app.errorhandler(429)
def rate_limit_exceeded(e):
    logger.warning(f"Rate limit hit from {request.remote_addr}")
    return jsonify({'error': 'Rate limit exceeded. Please try again later.'}), 429


@app.errorhandler(413)
def payload_too_large(e):
    return jsonify({'error': 'Request too large. Maximum allowed is 100 MiB.'}), 413


@app.errorhandler(500)
def internal_server_error(exc):
    logger.error(f"Internal server error: {exc}", exc_info=True)
    return jsonify({'error': 'An internal server error occurred'}), 500


@app.errorhandler(Exception)
def handle_exception(e):
    logger.error(f"Unhandled exception: {e}", exc_info=True)
    return jsonify({'error': 'An unexpected error occurred'}), 500


def open_browser(host, port):
    """Open browser after short delay"""
    try:
        time.sleep(0.5)
        url = f"http://{host}:{port}"
        webbrowser.open(url)
    except Exception as e:
        logger.error(f"Failed to open browser: {e}")


if __name__ == '__main__':
    if os.environ.get('SECRET_KEY') is None:
        logger.warning("SECRET_KEY not set in environment - using temporary random key")

    debug = os.environ.get('FLASK_DEBUG', 'False').lower() == 'true'
    if debug:
        logger.warning("Running in DEBUG mode - not suitable for production")

    host = os.environ.get('FLASK_HOST', '127.0.0.1')
    port = int(os.environ.get('FLASK_PORT', '5000'))

    logger.info(f"Starting Booklet server on {host}:{port}")
    logger.info(f"Output directory: {OUTPUT_DIRECTORY}")
    logger.info(f"Files retained: indefinitely (no auto-cleanup)")
    logger.info(f"Dynamic max chunk size: ~{MAX_CHARS_PER_CHUNK} characters")

    threading.Thread(
        target=open_browser,
        args=(host, port),
        daemon=True
    ).start()

    app.run(
        debug=debug,
        host=host,
        port=port,
        threaded=True
    )