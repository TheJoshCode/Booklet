#!/usr/bin/env python3
"""
Simple example script for Pocket TTS ONNX inference.

Usage:
    python generate.py "Hello, this is a test." samples/reference.wav output.wav
    python generate.py "Hello world" samples/expresso_02_ex03-ex01_calm_005.wav output.wav
"""

import argparse
import time
from gimma_pocket_onnx import PocketTTSOnnx


def main():
    parser = argparse.ArgumentParser(description="Generate speech with Pocket TTS ONNX")
    parser.add_argument("text", help="Text to synthesize")
    parser.add_argument("voice", help="Path to voice reference audio file")
    parser.add_argument("output", help="Output audio file path")
    parser.add_argument("--precision", choices=["int8", "fp32"], default="int8",
                        help="Model precision (default: int8)")
    args = parser.parse_args()

    print(f"Loading models (precision={args.precision})...")
    t0 = time.time()
    tts = PocketTTSOnnx(precision=args.precision)
    print(f"  Loaded in {time.time() - t0:.2f}s")

    print(f"Generating speech...")
    print(f"  Text: {args.text}")
    print(f"  Voice: {args.voice}")

    t0 = time.time()
    audio = tts.generate(args.text, voice=args.voice)
    gen_time = time.time() - t0

    duration = len(audio) / tts.SAMPLE_RATE
    rtfx = duration / gen_time

    print(f"  Generated {duration:.2f}s audio in {gen_time:.2f}s (RTFx: {rtfx:.2f}x)")

    tts.save_audio(audio, args.output)
    print(f"  Saved to: {args.output}")


if __name__ == "__main__":
    main()
