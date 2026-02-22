#!/usr/bin/env python3
"""
Worker instance for PocketTTS ONNX.
Loads model once, processes text chunks via ZMQ, saves MP3s to disk.
REQUIRES: pydub for MP3 encoding
"""

import zmq
import json
import sys
import os
import numpy as np
import logging
from pathlib import Path

# MP3 encoding is REQUIRED
try:
    from pydub import AudioSegment
except ImportError:
    print("ERROR: pydub is required for MP3 encoding!", file=sys.stderr)
    print("Install with: pip install pydub", file=sys.stderr)
    print("Also ensure ffmpeg is installed on your system", file=sys.stderr)
    sys.exit(1)

# Add pocket_tts_onnx to path if needed
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from gimma_pocket_onnx import PocketTTSOnnx

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [Worker %(worker_id)s] %(message)s',
    datefmt='%H:%M:%S'
)


class TTSWorker:
    def __init__(self, worker_id, socket_addr, model_precision="int8", temperature=0.7, lsd_steps=10):
        self.worker_id = worker_id
        self.socket_addr = socket_addr
        self.context = zmq.Context()
        self.socket = self.context.socket(zmq.REP)

        # CRITICAL: Use bind, not connect (we are the server)
        self.socket.bind(socket_addr)

        self.logger = logging.getLogger(f"Worker-{worker_id}")
        self.logger = logging.LoggerAdapter(self.logger, {'worker_id': worker_id})

        self.logger.info(f"Loading model (precision={model_precision})...")
        self.tts = PocketTTSOnnx(
            precision=model_precision,
            temperature=temperature,
            lsd_steps=lsd_steps
        )
        self.logger.info("Model loaded successfully")

        # NOTE: Voice embedding caching is handled entirely inside PocketTTSOnnx._voice_cache.
        # A second cache layer here was redundant: the onnx class already caches by path,
        # and the "populate from tts.last_voice_embedding" code referenced an attribute that
        # never existed on PocketTTSOnnx, so the outer cache never populated after the first
        # call anyway.  Single-layer caching in the onnx class is correct and sufficient.

    def save_mp3(self, audio_np: np.ndarray, output_path: str, sample_rate: int = 24000) -> str:
        """Save numpy audio array as MP3 file. ALWAYS outputs MP3."""
        # Ensure output path ends with .mp3
        if not output_path.lower().endswith('.mp3'):
            output_path = output_path + '.mp3'

        # Convert numpy array to AudioSegment
        # Normalize to 16-bit range for pydub
        audio_int16 = (audio_np * 32767).astype(np.int16)

        # Create AudioSegment from raw data
        audio_segment = AudioSegment(
            audio_int16.tobytes(),
            frame_rate=sample_rate,
            sample_width=2,  # 16-bit = 2 bytes
            channels=1       # Mono
        )

        # Export as MP3 with high quality
        audio_segment.export(output_path, format="mp3", bitrate="192k")

        self.logger.info(f"Saved MP3: {output_path}")
        return output_path

    def ensure_output_dir(self, output_path: str):
        """Create parent directory for output_path if it doesn't exist.
        
        OPTIMIZATION: Previously os.makedirs was called inside save_mp3 on
        every single chunk.  Now the orchestrator calls this once when the
        run directory is known, and individual chunk saves skip the syscall.
        """
        parent = os.path.dirname(output_path)
        if parent:
            os.makedirs(parent, exist_ok=True)

    def process_chunk(self, text: str, voice_wav_path: str, output_path: str) -> dict:
        """Process a single text chunk and save to MP3."""
        try:
            # PocketTTSOnnx._get_voice_embeddings caches embeddings by path internally.
            # Pass the path directly; the onnx class handles encoding + caching.
            audio = self.tts.generate(
                text=text,
                voice=voice_wav_path
            )

            # Save as MP3 (directory already guaranteed to exist)
            saved_path = self.save_mp3(audio, output_path, sample_rate=24000)

            return {
                'status': 'success',
                'output_path': saved_path,
                'sample_rate': 24000,
                'duration_seconds': len(audio) / 24000
            }

        except Exception as e:
            # Use json.dumps for safe serialization rather than manual escape
            return {
                'status': 'error',
                'error': str(e)
            }

    def run(self):
        """Main processing loop."""
        self.logger.info(f"Worker ready at {self.socket_addr}")

        # Track last seen output directory to avoid redundant makedirs calls.
        # For the common case where all chunks go to the same run dir, this
        # reduces the per-chunk overhead to a single string comparison.
        _last_output_dir: str = ""

        while True:
            try:
                # Receive request
                message = self.socket.recv_json()

                if message.get('command') == 'shutdown':
                    self.logger.info("Shutting down...")
                    self.socket.send_json({'status': 'shutdown_ok'})
                    break

                if message.get('command') == 'ping':
                    self.socket.send_json({'status': 'pong'})
                    continue

                if message.get('command') == 'process':
                    text = message['text']
                    voice_path = message['voice_path']
                    output_path = message['output_path']

                    # OPTIMIZATION: ensure output directory exists exactly once per
                    # unique directory rather than on every chunk save.
                    output_dir = os.path.dirname(output_path)
                    if output_dir and output_dir != _last_output_dir:
                        os.makedirs(output_dir, exist_ok=True)
                        _last_output_dir = output_dir

                    self.logger.info(
                        f"Processing chunk {message.get('chunk_index', '?')}: {text[:50]}..."
                    )

                    result = self.process_chunk(text, voice_path, output_path)
                    self.socket.send_json(result)

                else:
                    self.socket.send_json({
                        'status': 'error',
                        'error': f"Unknown command: {message.get('command')}"
                    })

            except zmq.ZMQError as e:
                self.logger.error(f"ZMQ Error: {e}")
                break
            except Exception as e:
                self.logger.error(f"Unexpected error: {e}")
                try:
                    self.socket.send_json({'status': 'error', 'error': str(e)})
                except Exception:
                    pass

        self.socket.close()
        self.context.term()
        self.logger.info("Worker terminated")


def main():
    import argparse
    parser = argparse.ArgumentParser(description='TTS Worker Instance')
    parser.add_argument('--worker-id', type=int, required=True)
    parser.add_argument('--socket', type=str, required=True,
                        help='ZMQ socket address (e.g., ipc:///tmp/tts_worker_0.sock)')
    parser.add_argument('--precision', type=str, default='int8', choices=['int8', 'fp32'])
    parser.add_argument('--temperature', type=float, default=0.7)
    parser.add_argument('--lsd-steps', type=int, default=10)

    args = parser.parse_args()

    worker = TTSWorker(
        worker_id=args.worker_id,
        socket_addr=args.socket,
        model_precision=args.precision,
        temperature=args.temperature,
        lsd_steps=args.lsd_steps
    )

    worker.run()


if __name__ == '__main__':
    main()