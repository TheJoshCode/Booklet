#!/usr/bin/env python3
"""
Pocket TTS CLI Wrapper
Processes text files sentence-by-sentence with parallel generation support.
Outputs MP3 format with async conversion and ETA tracking.
"""

import argparse
import re
import sys
import os
import unicodedata
import asyncio
import threading
import queue
import time
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Tuple, Optional, Dict
import warnings
import tempfile
import shutil
from dataclasses import dataclass
from collections import deque

try:
    from pocket_tts import TTSModel, export_model_state
    import scipy.io.wavfile
    import torch
    import numpy as np
except ImportError as e:
    print(f"Error: Missing required package - {e}")
    print("Install with: pip install pocket_tts scipy torch numpy")
    sys.exit(1)


@dataclass
class ProgressStats:
    """Track progress and estimate time remaining."""
    total: int
    completed: int = 0
    start_time: float = 0.0
    times: deque = None
    
    def __post_init__(self):
        if self.times is None:
            self.times = deque(maxlen=20)  # Keep last 20 samples for moving average
        if self.start_time == 0.0:
            self.start_time = time.time()
    
    def update(self, item_time: float):
        """Update with latest item completion time."""
        self.completed += 1
        self.times.append(item_time)
    
    def eta_seconds(self) -> float:
        """Calculate estimated seconds remaining."""
        if self.completed == 0 or not self.times:
            return 0.0
        
        avg_time = sum(self.times) / len(self.times)
        remaining = self.total - self.completed
        return avg_time * remaining
    
    def format_eta(self) -> str:
        """Format ETA as human readable string."""
        eta = self.eta_seconds()
        if eta == 0:
            return "calculating..."
        
        if eta < 60:
            return f"{int(eta)}s"
        elif eta < 3600:
            mins = int(eta // 60)
            secs = int(eta % 60)
            return f"{mins}m {secs}s"
        else:
            hrs = int(eta // 3600)
            mins = int((eta % 3600) // 60)
            return f"{hrs}h {mins}m"
    
    def percent(self) -> float:
        """Calculate completion percentage."""
        return (self.completed / self.total) * 100 if self.total > 0 else 0
    
    def elapsed(self) -> str:
        """Format elapsed time."""
        elapsed = time.time() - self.start_time
        if elapsed < 60:
            return f"{int(elapsed)}s"
        elif elapsed < 3600:
            return f"{int(elapsed // 60)}m {int(elapsed % 60)}s"
        else:
            return f"{int(elapsed // 3600)}h {int((elapsed % 3600) // 60)}m"


def normalize_text(text: str) -> str:
    """Normalize text: remove formatting, convert to plain text, handle unicode."""
    text = re.sub(r'\*\*', '', text)
    text = re.sub(r'__', '', text)
    text = re.sub(r'\*', '', text)
    text = re.sub(r'_', '', text)
    text = re.sub(r'`', '', text)
    text = re.sub(r'~', '', text)
    text = re.sub(r'#+\s*', '', text)
    text = re.sub(r'>\s*', '', text)
    text = re.sub(r'<[^>]+>', '', text)
    
    text = unicodedata.normalize('NFKD', text)
    text = ''.join(c for c in text if unicodedata.category(c) != 'Mn')
    
    text = text.replace('"', '"').replace('"', '"')
    text = text.replace("'", "'").replace("'", "'")
    text = text.replace('—', '--').replace('–', '-')
    text = text.replace('…', '...')
    text = text.replace('\u200b', '')
    text = text.replace('\xa0', ' ')
    
    text = ''.join(c for c in text if unicodedata.category(c)[0] != 'C' or c in '\n\r\t')
    text = re.sub(r'[\t ]+', ' ', text)
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()
    
    return text


def parse_sentences(text: str) -> List[str]:
    """Parse text into sentences."""
    text = normalize_text(text)
    text = re.sub(r'(Mr|Mrs|Ms|Dr|Prof|Sr|Jr|vs|Vol|vol|etc)\.', r'\1<PERIOD>', text)
    text = re.sub(r'([A-Z])\.', r'\1<PERIOD>', text)
    sentences = re.split(r'(?<=[.!?])\s+(?=[A-Z"\'])', text)
    sentences = [s.replace('<PERIOD>', '.').strip() for s in sentences if s.strip()]
    return sentences


def get_cache_path(voice_wav: str) -> Path:
    """Get the path for cached voice state."""
    voice_path = Path(voice_wav)
    cache_dir = Path.home() / ".cache" / "pocket_tts_cli"
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache_name = voice_path.stem + ".safetensors"
    return cache_dir / cache_name


def load_or_create_voice_state(model: TTSModel, voice_wav: str, force_refresh: bool = False):
    """Load voice state from cache or create and cache it."""
    cache_path = get_cache_path(voice_wav)
    
    if not force_refresh and cache_path.exists():
        print(f"Loading cached voice state from {cache_path}")
        return model.get_state_for_audio_prompt(str(cache_path))
    
    print(f"Creating voice state from {voice_wav}...")
    voice_state = model.get_state_for_audio_prompt(voice_wav)
    print(f"Caching voice state to {cache_path}")
    export_model_state(voice_state, str(cache_path))
    return voice_state


def wav_to_mp3(wav_path: str, mp3_path: str, bitrate: str = "192k") -> bool:
    """Convert WAV to MP3 using ffmpeg."""
    try:
        import subprocess
        cmd = [
            "ffmpeg", "-y", "-i", wav_path,
            "-codec:a", "libmp3lame",
            "-q:a", "2",
            "-loglevel", "error",
            mp3_path
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        return True
    except Exception as e:
        print(f"MP3 conversion failed for {wav_path}: {e}")
        return False


class AsyncMP3Converter:
    """Handles async MP3 conversion in background threads with progress tracking."""
    
    def __init__(self, max_workers: int = 2, total_items: int = 0):
        self.queue = queue.Queue()
        self.results = {}
        self.max_workers = max_workers
        self.workers = []
        self._stop_event = threading.Event()
        self.progress = ProgressStats(total=total_items)
        self._lock = threading.Lock()
        self._submitted = 0  # actual number of jobs submitted (may be < total_items if TTS fails)
        
    def start(self):
        """Start converter worker threads."""
        for _ in range(self.max_workers):
            t = threading.Thread(target=self._worker_loop, daemon=True)
            t.start()
            self.workers.append(t)
    
    def stop(self):
        """Signal workers to stop and wait for completion."""
        self._stop_event.set()
        for _ in range(self.max_workers):
            self.queue.put(None)
        for w in self.workers:
            w.join(timeout=5)
    
    def _worker_loop(self):
        """Worker thread loop."""
        while not self._stop_event.is_set():
            try:
                item = self.queue.get(timeout=1)
                if item is None:
                    break
                
                idx, wav_path, mp3_path = item
                start = time.time()
                success = wav_to_mp3(wav_path, mp3_path)
                duration = time.time() - start
                
                with self._lock:
                    self.progress.update(duration)
                    self.results[idx] = (mp3_path, success)
                
                if success:
                    try:
                        os.remove(wav_path)
                    except OSError:
                        pass
                self.queue.task_done()
            except queue.Empty:
                continue
    
    def submit(self, idx: int, wav_path: str, mp3_path: str):
        """Submit a conversion task."""
        with self._lock:
            self._submitted += 1
            # Keep progress.total in sync with what's actually been submitted so
            # the wait loop doesn't spin forever when some TTS chunks failed.
            self.progress.total = self._submitted
        self.queue.put((idx, wav_path, mp3_path))
    
    def wait_for_all(self):
        """Wait for all queued conversions to complete."""
        self.queue.join()
    
    def get_progress_str(self) -> str:
        """Get formatted progress string."""
        with self._lock:
            return f"[{self.progress.completed}/{self.progress.total}] {self.progress.percent():.1f}% | ETA: {self.progress.format_eta()}"


def generate_single(
    model: TTSModel,
    voice_state,
    sentence: str,
    output_path: Path,
    index: int,
    total: int,
    progress: ProgressStats
) -> Tuple[int, str, bool, str]:
    """Generate audio for a single sentence."""
    start_time = time.time()
    try:
        audio = model.generate_audio(voice_state, sentence)
        audio_np = audio.numpy() if hasattr(audio, 'numpy') else audio
        scipy.io.wavfile.write(str(output_path), model.sample_rate, audio_np)
        
        duration = time.time() - start_time
        progress.update(duration)
        
        # Print progress with ETA
        eta_str = progress.format_eta()
        pct = progress.percent()
        print(f"[{index+1}/{total}] {pct:.1f}% | ETA: {eta_str} | {sentence[:50]}{'...' if len(sentence) > 50 else ''}")
        
        return (index, str(output_path), True, sentence)
    except Exception as e:
        duration = time.time() - start_time
        progress.update(duration)  # Still count failed items for ETA
        print(f"[{index+1}/{total}] ERROR: {e}")
        return (index, str(e), False, sentence)


def combine_mp3s(file_list: List[str], output_path: Path):
    """Combine multiple MP3 files into one using ffmpeg."""
    try:
        import subprocess
        
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            for mp3_file in file_list:
                f.write(f"file '{os.path.abspath(mp3_file)}'\n")
            list_file = f.name
        
        cmd = [
            "ffmpeg", "-y", "-f", "concat", "-safe", "0",
            "-i", list_file,
            "-codec:a", "libmp3lame",
            "-q:a", "2",
            "-loglevel", "error",
            str(output_path)
        ]
        subprocess.run(cmd, check=True, capture_output=True)
        os.unlink(list_file)
        return True
    except Exception as e:
        print(f"Error combining MP3s: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description="Pocket TTS CLI - Text-to-Speech with MP3 output and ETA",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --voice_wav speaker.wav --text_file book.txt
  %(prog)s --voice_wav speaker.wav --text_file book.txt --workers 4 --convert_workers 3
  %(prog)s --voice_wav speaker.wav --text_file book.txt --combine --output final.mp3
        """
    )
    
    parser.add_argument("--voice_wav", required=True, help="Path to voice WAV file")
    parser.add_argument("--text_file", required=True, help="Path to text file")
    parser.add_argument("--output_dir", default="./tts_output", help="Output directory")
    parser.add_argument("--output", default="combined.mp3", help="Final output filename")
    parser.add_argument("--workers", type=int, default=1, help="TTS generation workers")
    parser.add_argument("--convert_workers", type=int, default=2, help="MP3 conversion workers")
    parser.add_argument("--combine", action="store_true", help="Combine into single MP3")
    parser.add_argument("--refresh-cache", action="store_true", help="Refresh voice cache")
    parser.add_argument("--keep-chunks", action="store_true", help="Keep individual MP3s")
    parser.add_argument("--preview", action="store_true", help="Preview normalized text")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.voice_wav):
        print(f"Error: Voice file not found: {args.voice_wav}")
        sys.exit(1)
    
    if not os.path.exists(args.text_file):
        print(f"Error: Text file not found: {args.text_file}")
        sys.exit(1)
    
    if not shutil.which("ffmpeg"):
        print("Error: ffmpeg not found. Install ffmpeg to use MP3 output.")
        sys.exit(1)
    
    # Read and parse text
    print(f"Reading {args.text_file}...")
    with open(args.text_file, 'r', encoding='utf-8') as f:
        text = f.read()
    
    sentences = parse_sentences(text)
    print(f"Parsed {len(sentences)} sentences")
    
    if not sentences:
        print("No sentences found!")
        sys.exit(1)
    
    if args.preview:
        print("\n=== Normalized Text Preview ===\n")
        for i, s in enumerate(sentences[:10]):
            print(f"{i+1}. {s}")
        if len(sentences) > 10:
            print(f"\n... and {len(sentences) - 10} more")
        print("\n==============================")
        sys.exit(0)
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    temp_dir = tempfile.mkdtemp(prefix="tts_wav_")
    
    # Initialize progress trackers
    gen_progress = ProgressStats(total=len(sentences))
    converter = AsyncMP3Converter(max_workers=args.convert_workers, total_items=len(sentences))
    converter.start()
    
    # Load model
    print("\nLoading TTS model...")
    model_start = time.time()
    model = TTSModel.load_model()
    voice_state = load_or_create_voice_state(model, args.voice_wav, args.refresh_cache)
    print(f"Model loaded in {time.time() - model_start:.1f}s\n")
    
    # Prepare tasks
    tasks = []
    for i, sentence in enumerate(sentences):
        wav_path = Path(temp_dir) / f"chunk_{i:04d}.wav"
        mp3_path = output_dir / f"chunk_{i:04d}.mp3"
        tasks.append((i, sentence, wav_path, mp3_path))
    
    # Generate audio
    print(f"Generating {len(sentences)} audio chunks with {args.workers} worker(s)...")
    print(f"MP3 conversion running on {args.convert_workers} background thread(s)...\n")
    
    generated_mp3s: List[Optional[str]] = [None] * len(sentences)
    
    import signal
    
    def _sigint_handler(sig, frame):
        print("\nInterrupted — cleaning up...")
        converter.stop()
        try:
            shutil.rmtree(temp_dir)
        except OSError:
            pass
        sys.exit(1)
    
    signal.signal(signal.SIGINT, _sigint_handler)
    
    if args.workers == 1:
        for i, sentence, wav_path, mp3_path in tasks:
            result = generate_single(model, voice_state, sentence, wav_path, i, len(tasks), gen_progress)
            if result[2]:  # success
                converter.submit(i, str(wav_path), str(mp3_path))
                generated_mp3s[i] = str(mp3_path)
    else:
        from threading import Lock
        model_lock = Lock()
        
        def thread_safe_generate(i, sentence, wav_path, mp3_path):
            with model_lock:
                result = generate_single(model, voice_state, sentence, wav_path, i, len(tasks), gen_progress)
                if result[2]:
                    converter.submit(i, str(wav_path), str(mp3_path))
                    return (i, str(mp3_path), True)
                return (i, result[1], False)
        
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {executor.submit(thread_safe_generate, i, s, w, m): i 
                      for i, s, w, m in tasks}
            
            for future in as_completed(futures):
                idx, path, success = future.result()
                if success:
                    generated_mp3s[idx] = path
    
    print(f"\nGeneration complete: {gen_progress.completed}/{gen_progress.total} in {gen_progress.elapsed()}")
    
    # Wait for MP3 conversions with progress updates
    print("\nWaiting for MP3 conversions...")
    last_completed = 0
    while converter._submitted > 0 and converter.progress.completed < converter.progress.total:
        if converter.progress.completed != last_completed:
            print(f"  Conversion: {converter.get_progress_str()}")
            last_completed = converter.progress.completed
        time.sleep(0.5)
    
    converter.wait_for_all()
    converter.stop()
    print(f"  Conversion complete: {converter.progress.completed}/{converter.progress.total} in {converter.progress.elapsed()}")
    
    # Cleanup temp dir
    try:
        shutil.rmtree(temp_dir)
    except OSError:
        pass
    
    # Check results
    successful = [p for p in generated_mp3s if p is not None]
    failed_count = len(sentences) - len(successful)
    
    if failed_count > 0:
        print(f"\nWarning: {failed_count} sentence(s) failed")
    
    # Combine if requested
    if args.combine and successful:
        successful.sort(key=lambda x: int(Path(x).stem.split('_')[1]))
        final_path = output_dir / args.output
        
        print(f"\nCombining {len(successful)} MP3 files into {args.output}...")
        combine_start = time.time()
        
        if combine_mp3s(successful, final_path):
            combine_time = time.time() - combine_start
            print(f"Combined in {combine_time:.1f}s -> {final_path}")
            
            if not args.keep_chunks:
                print("Cleaning up chunk files...")
                for f in successful:
                    try:
                        os.remove(f)
                    except OSError:
                        pass
        else:
            print("Failed to combine MP3s, keeping individual files")
    
    total_time = gen_progress.elapsed()
    print(f"\n✓ Done! Total time: {total_time}")


if __name__ == "__main__":
    main()