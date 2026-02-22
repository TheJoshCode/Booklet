---
license: cc-by-4.0
language:
- en
library_name: pocket-tts-onnx
base_model:
- kyutai/pocket-tts
pipeline_tag: text-to-speech
tags:
- tts
- voice-cloning
- onnx
- onnxruntime
---

# Pocket TTS ONNX

ONNX export of [Pocket TTS](https://huggingface.co/kyutai/pocket-tts) for lightweight text-to-speech with zero-shot voice cloning.

## Model Description

Pocket TTS is a compact text-to-speech model from Kyutai that supports voice cloning from short audio samples. This ONNX export provides:

- **Zero-shot voice cloning** from any audio reference
- **INT8 quantized models** for fast CPU inference
- **Streaming support** with adaptive chunking for real-time playback
- **Temperature control** for generation diversity
- **Dual model architecture** for flexible flow matching

## Architecture

This export uses a **dual model split** for the Flow LM:

```
flow_lm_main   - Transformer/conditioner (produces conditioning vectors)
flow_lm_flow   - Flow network only (Euler integration for latent sampling)
```

This architecture enables:
- **Temperature control**: Adjust generation diversity via noise scaling
- **Variable LSD steps**: Trade off speed vs quality
- **External flow loop**: Full control over the sampling process

## Performance

Benchmarked on a 16-core CPU, `lsd_steps=1`, measuring flow loop + mimi decoder (excluding voice encoder).

| Precision | Flow Main | Flow Net | Decoder | Total Size | RTFx (CPU) |
|-----------|-----------|----------|---------|------------|------------|
| INT8 | 76 MB | 10 MB | 23 MB | ~200 MB | **~4.0x** |
| FP32 | 303 MB | 39 MB | 42 MB | ~475 MB | **~2.8x** |
| PyTorch FP32 (reference) | — | — | — | — | ~4.0x |

RTFx = Real-time factor (>1.0 means faster than real-time). ONNX INT8 matches PyTorch FP32 throughput.

### Optimized Inference

Thread count is automatically tuned to avoid over-subscription on the small sequential matmuls in the autoregressive loop (`intra_op_num_threads=min(cpu_count, 4)`, `inter_op_num_threads=1`). This alone provides a **~2x speedup** over default ORT settings on multi-core machines.

### Decoding Strategy

- **Offline (`generate`)**: Uses **threaded parallel decoding** — the mimi decoder runs in a background thread, decoding 12-frame chunks while the flow loop generates the next frames. This overlaps generation and decoding for maximum throughput.
- **Streaming (`stream`)**: Uses **adaptive chunking** (starts at 2 frames). This ensures instant start (low TTFB) while scaling up chunk sizes for throughput.

## Usage

```python
from pocket_tts_onnx import PocketTTSOnnx

# Load model (INT8 by default)
tts = PocketTTSOnnx()

# Generate speech with voice cloning
audio = tts.generate(
    text="Hello, this is a test of voice cloning.",
    voice="reference_sample.wav"
)

# Save output
tts.save_audio(audio, "output.wav")
```

### Temperature Control

Adjust generation diversity with the `temperature` parameter:

```python
# More deterministic (lower temperature)
tts = PocketTTSOnnx(temperature=0.3)

# Default balance
tts = PocketTTSOnnx(temperature=0.7)

# More diverse/expressive (higher temperature)
tts = PocketTTSOnnx(temperature=1.0)
```

### LSD Steps

Trade off speed vs quality with `lsd_steps`:

```python
# Default (10 steps)
tts = PocketTTSOnnx(lsd_steps=10)

# Faster (fewer steps, lower quality)
tts = PocketTTSOnnx(lsd_steps=1)
```

### Streaming Mode

For real-time applications with low time-to-first-audio:

```python
for chunk in tts.stream("Hello world!", voice="reference_sample.wav"):
    play_audio(chunk)  # Process each chunk as it arrives
```

### Command Line

```bash
python generate.py "Hello, this is a test." reference_sample.wav output.wav
```

## Files

```
pocket-tts-onnx/
├── onnx/
│   ├── flow_lm_main.onnx          # 303 MB - Flow LM transformer (FP32)
│   ├── flow_lm_main_int8.onnx     #  76 MB - Flow LM transformer (INT8)
│   ├── flow_lm_flow.onnx          #  39 MB - Flow network (FP32)
│   ├── flow_lm_flow_int8.onnx     #  10 MB - Flow network (INT8)
│   ├── mimi_decoder.onnx          #  42 MB - Audio decoder (FP32)
│   ├── mimi_decoder_int8.onnx     #  23 MB - Audio decoder (INT8)
│   ├── mimi_encoder.onnx          #  73 MB - Voice encoder
│   └── text_conditioner.onnx      #  16 MB - Text embeddings
├── reference_sample.wav           # Example voice reference
├── tokenizer.model                # SentencePiece tokenizer
├── pocket_tts_onnx.py             # Inference wrapper
├── generate.py                    # CLI script
├── requirements.txt               # Python dependencies
└── README.md
```

## Requirements

```
onnxruntime>=1.16.0
numpy
soundfile
sentencepiece
scipy  # Only needed if resampling from non-24kHz audio
```

Install with:
```bash
pip install -r requirements.txt
```

## License

- **Models**: CC BY 4.0 (inherited from [kyutai/pocket-tts](https://huggingface.co/kyutai/pocket-tts))
- **Code**: Apache 2.0

## Prohibited Use

Use of our model must comply with all applicable laws and regulations and must not result in, involve, or facilitate any illegal, harmful, deceptive, fraudulent, or unauthorized activity. Prohibited uses include, without limitation, voice impersonation or cloning without explicit and lawful consent; misinformation, disinformation, or deception (including fake news, fraudulent calls, or presenting generated content as genuine recordings of real people or events); and the generation of unlawful, harmful, libelous, abusive, harassing, discriminatory, hateful, or privacy-invasive content. We disclaim all liability for any non-compliant use.

## Acknowledgments

- [Kyutai](https://kyutai.org/) for the original Pocket TTS model
