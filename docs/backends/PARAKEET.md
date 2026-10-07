# Parakeet

Parakeet TDT V3 via [onnx-asr](https://github.com/istupakov/onnx-asr).

## Setup

Run `hyprwhspr setup` and select **Parakeet → Parakeet v3**. The model (~1 GB) is downloaded during setup.

## Configuration

```jsonc
{
    "transcription_backend": "onnx-asr",
    "onnx_asr_model": "nemo-parakeet-tdt-0.6b-v3",  // default
    "onnx_asr_quantization": "int8",                  // int8 (default) | null (FP32)
    "onnx_asr_use_vad": true,                         // Silero VAD for longer recordings (default: true)
    "onnx_asr_vad_min_duration": 30                   // seconds before VAD is used (default: 30)
}
```

[Orukeet](https://huggingface.co/oruk/orukeet): an optional Parakeet v3
fine-tune for 25 European languages. Pick **Parakeet → Orukeet** in setup, or:

```sh
hyprwhspr setup auto --backend onnx-asr --model orukeet
```

Or set `"onnx_asr_model": "orukeet"`. Always int8; your
`onnx_asr_quantization` stays put for Parakeet.

About 672 MB, checked against pinned checksums, run locally. On a checksum
error, `hyprwhspr model download` repairs the file.

[Benchmarks and limitations](../benchmarks/orukeet-linux-20260918.md).
