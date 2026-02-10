import sys
import os
# Force the current directory to be at the front of the path
sys.path.insert(0, os.path.abspath(os.path.dirname(__file__)))

from PySide6.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                               QLabel, QLineEdit, QPushButton, QComboBox, QFileDialog, 
                               QProgressBar, QMessageBox, QFrame, QTextEdit)
from PySide6.QtCore import QThread, Signal, Qt
from PySide6.QtGui import QFont, QPalette, QColor
from zipvoice.luxvoice import LuxTTS
import soundfile as sf
import numpy as np
from pydub import AudioSegment
import nltk
import pdfplumber
from docx import Document
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
import uuid
import gc
import psutil

# Ensure NLTK sentence tokenizer
try:
    nltk.download('punkt_tab', quiet=True)
except:
    nltk.download('punkt', quiet=True)

# Memory Management Configuration
MEMORY_THRESHOLD_MB = 500  # Unload model if available memory drops below this
CHUNKS_BEFORE_CLEANUP = 10  # Run garbage collection every N chunks
MODEL_RELOAD_INTERVAL = 50  # Reload model every N chunks to prevent memory leaks

def get_last_completed_chunk(chunks_dir):
    if not os.path.exists(chunks_dir):
        return 0

    files = [
        f for f in os.listdir(chunks_dir)
        if f.startswith("chunk_") and f.endswith(".mp3")
    ]

    if not files:
        return 0

    nums = [
        int(f.split("_")[1].split(".")[0])
        for f in files
        if f.split("_")[1].split(".")[0].isdigit()
    ]

    return max(nums) if nums else 0

def get_memory_info():
    """Get current memory usage information"""
    process = psutil.Process()
    mem_info = process.memory_info()
    virtual_mem = psutil.virtual_memory()
    
    return {
        'process_rss_mb': mem_info.rss / (1024 * 1024),
        'process_vms_mb': mem_info.vms / (1024 * 1024),
        'system_available_mb': virtual_mem.available / (1024 * 1024),
        'system_percent': virtual_mem.percent
    }


def clear_memory(verbose_callback=None):
    """Aggressive memory cleanup"""
    if verbose_callback:
        verbose_callback("🧹 Running memory cleanup...")
    
    # Python garbage collection
    collected = gc.collect()
    
    if verbose_callback:
        mem_info = get_memory_info()
        verbose_callback(f"   Collected {collected} objects | RAM: {mem_info['process_rss_mb']:.1f}MB | Available: {mem_info['system_available_mb']:.1f}MB")

def should_reload_model(mem_info):
    """Determine if model should be reloaded due to low memory"""
    return mem_info['system_available_mb'] < MEMORY_THRESHOLD_MB

def extract_text_from_file(file_path):
    ext = os.path.splitext(file_path)[1].lower()
    if ext == '.txt':
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    elif ext == '.pdf':
        text = ''
        with pdfplumber.open(file_path) as pdf:
            for page in pdf.pages:
                text += page.extract_text() or ''
        return text
    elif ext == '.docx':
        doc = Document(file_path)
        return '\n'.join([para.text for para in doc.paragraphs])
    elif ext == '.epub':
        book = epub.read_epub(file_path)
        text = ''
        for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
            soup = BeautifulSoup(item.get_content(), 'html.parser')
            text += soup.get_text() + '\n'
        return text
    else:
        raise ValueError(f"Unsupported file format: {ext}. Supported: .txt, .pdf, .docx, .epub")

def split_text_into_chunks(text, max_length=300):
    sentences = nltk.sent_tokenize(text)
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        # Skip empty sentences
        if not sentence.strip():
            continue
            
        if len(current_chunk) + len(sentence) > max_length and current_chunk:
            chunks.append(current_chunk.strip())
            current_chunk = sentence
        else:
            if current_chunk:
                current_chunk += " " + sentence
            else:
                current_chunk = sentence
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    # Filter out chunks that are too short (less than 10 characters)
    chunks = [chunk for chunk in chunks if len(chunk) >= 10]
    
    return chunks if chunks else [text.strip()]  # Return original text if no valid chunks

class GenerationThread(QThread):
    status_update = Signal(str)
    progress_update = Signal(int)
    finished = Signal(bool, str, str)  # success, message, output_path
    verbose_log = Signal(str)  # For detailed memory and process logging

    def __init__(self, text_file, voice_clip, voice_transcription, ref_duration, num_steps):
        super().__init__()
        self.text_file = text_file
        self.voice_clip = voice_clip
        self.voice_transcription = voice_transcription
        self.ref_duration = ref_duration
        self.num_steps = num_steps
        self.lux_tts = None
    
    def log_verbose(self, message):
        """Send verbose logging message"""
        self.verbose_log.emit(message)
    
    def log_memory_status(self, context=""):
        """Log current memory status"""
        mem_info = get_memory_info()
        self.log_verbose(f"💾 Memory Status {context}:")
        self.log_verbose(f"   Process: {mem_info['process_rss_mb']:.1f}MB | System: {mem_info['system_percent']:.1f}% used | Available: {mem_info['system_available_mb']:.1f}MB")
    
    def load_model(self):
            self.log_verbose("🔄 Loading LuxTTS model...")
            self.lux_tts = LuxTTS('YatharthS/LuxTTS', device='cuda')
            self.log_verbose("✅ Model loaded successfully")
    
    def unload_model(self):
        """Unload the TTS model and free memory"""
        if self.lux_tts is not None:
            self.log_verbose("🗑️ Unloading model to free memory...")
            del self.lux_tts
            self.lux_tts = None
            clear_memory(self.log_verbose)
            self.log_memory_status("(after model unload)")

    def run(self):
        try:
            # Initial memory status
            self.log_verbose("=" * 60)
            self.log_verbose("🚀 Starting audiobook generation")
            self.log_memory_status("(initial)")
            
            base_outputs_dir = "outputs"
            os.makedirs(base_outputs_dir, exist_ok=True)

            # Create run folder: outputs/output1, output2, ...
            existing_runs = [
                d for d in os.listdir(base_outputs_dir)
                if d.startswith("output") and os.path.isdir(os.path.join(base_outputs_dir, d))
            ]
            run_index = len(existing_runs) + 1

            run_dir = os.path.join(base_outputs_dir, f"output{run_index}")
            chunks_dir = os.path.join(run_dir, "chunks")

            os.makedirs(chunks_dir, exist_ok=True)

            final_output_file = os.path.join(run_dir, "final.mp3")

            self.log_verbose(f"📁 Run directory: {run_dir}")
            self.log_verbose(f"📁 Chunks directory: {chunks_dir}")
            
            # Extract text
            self.status_update.emit("Extracting text from book file...")
            self.log_verbose(f"📖 Reading book file: {os.path.basename(self.text_file)}")
            book_text = extract_text_from_file(self.text_file)
            
            if len(book_text.strip()) == 0:
                self.finished.emit(False, "The book file appears to be empty.", "")
                return
            
            self.log_verbose(f"   Extracted {len(book_text)} characters")
            
            # Load model
            self.status_update.emit("Loading LuxTTS model...")
            self.load_model()
            
            # Validate and encode voice
            self.status_update.emit(f"Encoding voice from {os.path.basename(self.voice_clip)}...")
            self.log_verbose(f"🎤 Processing voice clip: {os.path.basename(self.voice_clip)}")
            
            import soundfile as sf_check
            audio_data, sample_rate = sf_check.read(self.voice_clip)
            audio_duration = len(audio_data) / sample_rate
            
            self.log_verbose(f"   Voice duration: {audio_duration:.2f}s @ {sample_rate}Hz")
            
            if audio_duration < 2:
                self.finished.emit(False, f"Voice clip is too short ({audio_duration:.2f}s). Please use a clip that's at least 2 seconds long.", "")
                return
            
            # Adjust ref_duration if the audio clip is shorter
            actual_ref_duration = min(self.ref_duration, int(audio_duration))
            if actual_ref_duration < self.ref_duration:
                self.status_update.emit(f"Adjusting ref duration to {actual_ref_duration}s (voice clip length)...")
                self.log_verbose(f"⚠️ Adjusted ref_duration from {self.ref_duration}s to {actual_ref_duration}s")
            
            self.log_verbose("🔊 Encoding voice prompt...")
            encoded_prompt = self.lux_tts.encode_prompt(self.voice_transcription, self.voice_clip, rms=0.01, duration=actual_ref_duration)
            self.log_verbose("✅ Voice encoded successfully")
            
            # Split text into chunks
            self.log_verbose("✂️ Splitting text into chunks...")
            chunks = split_text_into_chunks(book_text)
            
            if not chunks:
                self.finished.emit(False, "Could not extract valid text chunks from the book.", "")
                return
                
            self.status_update.emit(f"Text split into {len(chunks)} chunks.")
            self.log_verbose(f"   Created {len(chunks)} chunks")
            avg_chunk_len = sum(len(c) for c in chunks) / len(chunks)
            self.log_verbose(f"   Average chunk length: {avg_chunk_len:.0f} characters")
            self.progress_update.emit(0)
            
            # Clear memory before starting generation
            clear_memory(self.log_verbose)
            
            # Generate audio
            audio_segments = []
            total_chunks = len(chunks)
            
            self.log_verbose("=" * 60)
            self.log_verbose("🎙️ Starting audio generation...")
            
            for i, chunk in enumerate(chunks):
                if len(chunk.strip()) < 5:
                    self.log_verbose(f"⏭️ Skipping chunk {i+1}/{total_chunks} (too short)")
                    continue
                
                chunk_num = i + 1
                self.status_update.emit(f"Generating audio for chunk {chunk_num}/{total_chunks}...")
                self.log_verbose(f"\n📝 Chunk {chunk_num}/{total_chunks} ({len(chunk)} chars)")
                
                # Periodic memory cleanup
                if chunk_num % CHUNKS_BEFORE_CLEANUP == 0:
                    self.log_verbose(f"🧹 Periodic cleanup at chunk {chunk_num}...")
                    clear_memory(self.log_verbose)
                    self.log_memory_status(f"(after cleanup at chunk {chunk_num})")
                
                # Check if we need to reload model due to memory pressure
                mem_info = get_memory_info()
                if should_reload_model(mem_info):
                    self.log_verbose(f"⚠️ Low memory detected! Available: {mem_info['system_available_mb']:.1f}MB")
                    self.log_verbose("🔄 Reloading model to free memory...")
                    self.unload_model()
                    clear_memory(self.log_verbose)
                    self.load_model()
                    # Re-encode prompt after model reload
                    encoded_prompt = self.lux_tts.encode_prompt(self.voice_transcription, self.voice_clip, rms=0.01, duration=actual_ref_duration)
                
                # Periodic model reload to prevent memory leaks
                elif chunk_num % MODEL_RELOAD_INTERVAL == 0 and chunk_num > 0:
                    self.log_verbose(f"🔄 Periodic model reload at chunk {chunk_num}...")
                    self.unload_model()
                    clear_memory(self.log_verbose)
                    self.load_model()
                    encoded_prompt = self.lux_tts.encode_prompt(self.voice_transcription, self.voice_clip, rms=0.01, duration=actual_ref_duration)
                
                try:
                    # Generate speech
                    self.log_verbose("   Generating speech...")
                    wav = self.lux_tts.generate_speech(chunk, encoded_prompt, num_steps=self.num_steps)
                    wav_np = wav.cpu().numpy().squeeze()
                    
                    # Ensure wav_np has sufficient length
                    if len(wav_np) < 100:
                        self.log_verbose(f"   ⚠️ Generated audio too short ({len(wav_np)} samples), skipping")
                        continue
                    
                    self.log_verbose(f"   ✅ Generated {len(wav_np)} samples ({len(wav_np)/48000:.2f}s)")
                    
                    segment = AudioSegment(
                        wav_np.tobytes(),
                        frame_rate=48000,
                        sample_width=4,
                        channels=1
                    )

                    # Save individual chunk
                    chunk_filename = f"chunk_{chunk_num:04d}.mp3"
                    chunk_path = os.path.join(chunks_dir, chunk_filename)
                    segment.export(chunk_path, format="mp3")

                    self.log_verbose(f"   💾 Saved chunk → {chunk_filename}")

                    audio_segments.append(segment)

                    # Clear the wav data immediately
                    del wav, wav_np
                    
                except Exception as chunk_error:
                    self.log_verbose(f"   ❌ Error: {str(chunk_error)}")
                    self.status_update.emit(f"Warning: Skipping chunk {chunk_num} due to error")
                    continue
                
                # Update progress
                progress = int((chunk_num) / total_chunks * 100)
                self.progress_update.emit(progress)
            
            if not audio_segments:
                self.finished.emit(False, "No audio segments were generated successfully. Please check your voice clip and text file.", "")
                return
            
            self.log_verbose("=" * 60)
            self.log_verbose(f"🎵 Concatenating {len(audio_segments)} audio segments...")
            self.status_update.emit("Concatenating audio chunks...")
            full_audio = sum(audio_segments)
            
            # Clear segments from memory
            self.log_verbose("🧹 Clearing audio segments from memory...")
            audio_segments.clear()
            clear_memory(self.log_verbose)
            
            # Export
            self.log_verbose(f"💾 Exporting to {os.path.basename(output_file)}...")
            self.status_update.emit(f"Exporting to {os.path.basename(output_file)}...")
            full_audio.export(final_output_file, format="mp3")
            
            file_size_mb = os.path.getsize(final_output_file) / (1024 * 1024)
            self.log_verbose(f"✅ Export complete! File size: {file_size_mb:.2f}MB")
            
            # Final cleanup
            self.log_verbose("🧹 Final cleanup...")
            self.unload_model()
            del full_audio
            clear_memory(self.log_verbose)
            
            self.log_memory_status("(final)")
            self.log_verbose("=" * 60)
            self.log_verbose("🎉 Audiobook generation complete!")
            
            self.finished.emit(
                True,
                f"Audiobook generated successfully!\n\nSaved to: {final_output_file}\nFile size: {file_size_mb:.2f}MB",
                final_output_file
            )
            
        except Exception as e:
            self.log_verbose(f"❌ Fatal error: {str(e)}")
            import traceback
            self.log_verbose(traceback.format_exc())
            self.finished.emit(False, f"An error occurred: {str(e)}", "")


class FileInputRow(QWidget):
    """Custom widget for file input with label, entry, and browse button"""
    def __init__(self, label_text, placeholder_text, file_filter, is_save=False):
        super().__init__()
        self.file_filter = file_filter
        self.is_save = is_save
        
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 12)
        layout.setSpacing(6)
        
        # Label
        label = QLabel(label_text)
        label.setStyleSheet("font-weight: 600; font-size: 13px; color: #e0e0e0;")
        layout.addWidget(label)
        
        # Entry and button in horizontal layout
        h_layout = QHBoxLayout()
        h_layout.setSpacing(8)
        
        self.entry = QLineEdit()
        self.entry.setPlaceholderText(placeholder_text)
        self.entry.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d30;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 10px 12px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #0078d4;
                background-color: #323235;
            }
            QLineEdit:hover {
                border: 1px solid #505055;
            }
        """)
        
        browse_btn = QPushButton("Browse")
        browse_btn.setStyleSheet("""
            QPushButton {
                background-color: #3f3f46;
                border: none;
                border-radius: 6px;
                padding: 10px 20px;
                color: #e0e0e0;
                font-size: 13px;
                font-weight: 500;
            }
            QPushButton:hover {
                background-color: #505055;
            }
            QPushButton:pressed {
                background-color: #2d2d30;
            }
        """)
        browse_btn.setFixedWidth(100)
        browse_btn.clicked.connect(self.browse)
        
        h_layout.addWidget(self.entry)
        h_layout.addWidget(browse_btn)
        layout.addLayout(h_layout)
    
    def browse(self):
        if self.is_save:
            file = QFileDialog.getSaveFileName(self, "Save File", "", self.file_filter)[0]
        else:
            file = QFileDialog.getOpenFileName(self, "Select File", "", self.file_filter)[0]
        if file:
            self.entry.setText(file)
    
    def text(self):
        return self.entry.text()

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Booklet - Audiobook Generator")
        self.setGeometry(100, 100, 650, 900)
        
        # Main widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(30, 30, 30, 30)
        main_layout.setSpacing(20)
        
        # Header
        header = QLabel("Booklet")
        header.setStyleSheet("""
            font-size: 28px;
            font-weight: 700;
            color: #ffffff;
            margin-bottom: 10px;
        """)
        header.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(header)
        
        subtitle = QLabel("Transform your books into high-quality audiobooks")
        subtitle.setStyleSheet("""
            font-size: 14px;
            color: #a0a0a0;
            margin-bottom: 20px;
        """)
        subtitle.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(subtitle)
        
        # Separator
        separator = QFrame()
        separator.setFrameShape(QFrame.HLine)
        separator.setStyleSheet("background-color: #3f3f46; margin: 10px 0;")
        separator.setFixedHeight(1)
        main_layout.addWidget(separator)
        
        # File inputs
        self.text_input = FileInputRow(
            "Book File",
            "Select your book file (TXT, PDF, DOCX, or EPUB)",
            "Book Files (*.txt *.pdf *.docx *.epub);;All Files (*)"
        )
        main_layout.addWidget(self.text_input)
        
        self.voice_input = FileInputRow(
            "Voice Reference",
            "Select a 5-second voice sample (WAV or MP3)",
            "Audio Files (*.wav *.mp3)"
        )
        main_layout.addWidget(self.voice_input)
        
        # Voice Transcription input
        transcription_label = QLabel("Voice Transcription")
        transcription_label.setStyleSheet("font-weight: 600; font-size: 13px; color: #e0e0e0;")
        main_layout.addWidget(transcription_label)
        
        self.transcription_input = QLineEdit()
        self.transcription_input.setPlaceholderText("Enter the text that is spoken in the voice reference clip")
        self.transcription_input.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d30;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 10px 12px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #0078d4;
                background-color: #323235;
            }
            QLineEdit:hover {
                border: 1px solid #505055;
            }
        """)
        main_layout.addWidget(self.transcription_input)
        
        # Settings section
        settings_label = QLabel("⚙️ Settings")
        settings_label.setStyleSheet("font-weight: 600; font-size: 15px; color: #e0e0e0; margin-top: 10px;")
        main_layout.addWidget(settings_label)
        
        # Settings grid
        settings_layout = QHBoxLayout()
        settings_layout.setSpacing(15)
        
        # Ref Duration
        ref_widget = QWidget()
        ref_layout = QVBoxLayout(ref_widget)
        ref_layout.setContentsMargins(0, 0, 0, 0)
        ref_layout.setSpacing(6)
        
        ref_label = QLabel("Ref Duration (s)")
        ref_label.setStyleSheet("font-size: 13px; color: #e0e0e0;")
        ref_layout.addWidget(ref_label)
        
        self.ref_entry = QLineEdit("5")
        self.ref_entry.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d30;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 10px 12px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #0078d4;
                background-color: #323235;
            }
            QLineEdit:hover {
                border: 1px solid #505055;
            }
        """)
        ref_layout.addWidget(self.ref_entry)
        settings_layout.addWidget(ref_widget)
        
        # Num Steps
        steps_widget = QWidget()
        steps_layout = QVBoxLayout(steps_widget)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(6)
        
        steps_label = QLabel("Num Steps")
        steps_label.setStyleSheet("font-size: 13px; color: #e0e0e0;")
        steps_layout.addWidget(steps_label)
        
        self.steps_entry = QLineEdit("4")
        self.steps_entry.setStyleSheet("""
            QLineEdit {
                background-color: #2d2d30;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 10px 12px;
                color: #e0e0e0;
                font-size: 13px;
            }
            QLineEdit:focus {
                border: 1px solid #0078d4;
                background-color: #323235;
            }
            QLineEdit:hover {
                border: 1px solid #505055;
            }
        """)
        steps_layout.addWidget(self.steps_entry)
        settings_layout.addWidget(steps_widget)
        
        main_layout.addLayout(settings_layout)
        
        # Generate Button
        self.generate_btn = QPushButton("Generate")
        self.generate_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 8px;
                padding: 14px;
                color: #ffffff;
                font-size: 15px;
                font-weight: 600;
                margin-top: 20px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
            QPushButton:pressed {
                background-color: #006cbe;
            }
            QPushButton:disabled {
                background-color: #3f3f46;
                color: #808080;
            }
        """)
        self.generate_btn.clicked.connect(self.start_generation)
        main_layout.addWidget(self.generate_btn)
        
        # Status Label
        self.status_label = QLabel("Ready to generate")
        self.status_label.setStyleSheet("""
            font-size: 13px;
            color: #a0a0a0;
            padding: 8px;
            margin-top: 10px;
        """)
        self.status_label.setAlignment(Qt.AlignCenter)
        main_layout.addWidget(self.status_label)
        
        # Progress Bar
        self.progress_bar = QProgressBar()
        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                background-color: #2d2d30;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                text-align: center;
                color: #e0e0e0;
                font-size: 12px;
                height: 28px;
            }
            QProgressBar::chunk {
                background-color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #0078d4, stop:1 #1084d8);
                border-radius: 5px;
            }
        """)
        main_layout.addWidget(self.progress_bar)
        
        # Verbose Log Section
        log_header = QLabel("📊 Verbose Logging")
        log_header.setStyleSheet("font-weight: 600; font-size: 13px; color: #e0e0e0; margin-top: 15px;")
        main_layout.addWidget(log_header)
        
        self.verbose_log = QTextEdit()
        self.verbose_log.setReadOnly(True)
        self.verbose_log.setMaximumHeight(200)
        self.verbose_log.setStyleSheet("""
            QTextEdit {
                background-color: #1a1a1c;
                border: 1px solid #3f3f46;
                border-radius: 6px;
                padding: 8px;
                color: #d4d4d4;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                line-height: 1.4;
            }
        """)
        main_layout.addWidget(self.verbose_log)
        
        # Add stretch to push everything up
        main_layout.addStretch()
        
        # Set window style
        self.setStyleSheet("""
            QMainWindow {
                background-color: #1e1e1e;
            }
            QWidget {
                background-color: #1e1e1e;
                color: #e0e0e0;
            }
        """)
    
    def start_generation(self):
        text_file = self.text_input.text()
        voice_clip = self.voice_input.text()
        voice_transcription = self.transcription_input.text()
        
        try:
            ref_duration = int(self.ref_entry.text())
            num_steps = int(self.steps_entry.text())
        except ValueError:
            self.show_message("Input Error", "Reference duration and steps must be integers.", error=True)
            return
        
        if not all([text_file, voice_clip, voice_transcription]):
            self.show_message("Missing Input", "Please select book file, voice clip, and enter voice transcription.", error=True)
            return
        
        # Clear verbose log
        self.verbose_log.clear()
        
        self.generate_btn.setEnabled(False)
        self.progress_bar.setValue(0)
        
        self.thread = GenerationThread(text_file, voice_clip, voice_transcription, ref_duration, num_steps)
        self.thread.status_update.connect(self.update_status)
        self.thread.progress_update.connect(self.update_progress)
        self.thread.finished.connect(self.generation_finished)
        self.thread.verbose_log.connect(self.update_verbose_log)
        self.thread.start()
    
    def update_status(self, text):
        self.status_label.setText(text)
    
    def update_progress(self, value):
        self.progress_bar.setValue(value)
    
    def update_verbose_log(self, text):
        """Append verbose log message and auto-scroll"""
        self.verbose_log.append(text)
        # Auto-scroll to bottom
        scrollbar = self.verbose_log.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())
    
    def generation_finished(self, success, message, output_path):
        self.generate_btn.setEnabled(True)
        if success and output_path:
            # Open the outputs folder after successful generation
            import subprocess
            import platform
            outputs_dir = os.path.dirname(output_path)
            
            try:
                if platform.system() == "Windows":
                    os.startfile(outputs_dir)
                elif platform.system() == "Darwin":  # macOS
                    subprocess.run(["open", outputs_dir])
                else:  # Linux
                    subprocess.run(["xdg-open", outputs_dir])
            except:
                pass  # Silently fail if can't open folder
                
        self.show_message("Success" if success else "Error", message, error=not success)
    
    def show_message(self, title, message, error=False):
        msg = QMessageBox(self)
        msg.setWindowTitle(title)
        msg.setText(message)
        msg.setIcon(QMessageBox.Critical if error else QMessageBox.Information)
        
        # Style the message box
        msg.setStyleSheet("""
            QMessageBox {
                background-color: #2d2d30;
            }
            QMessageBox QLabel {
                color: #e0e0e0;
                font-size: 13px;
            }
            QPushButton {
                background-color: #0078d4;
                border: none;
                border-radius: 4px;
                padding: 8px 16px;
                color: #ffffff;
                font-size: 13px;
                min-width: 70px;
            }
            QPushButton:hover {
                background-color: #1084d8;
            }
        """)
        msg.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Set application-wide font
    font = QFont("Segoe UI", 10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
