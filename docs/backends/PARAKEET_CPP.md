# Parakeet.cpp

A second engine for Parakeet TDT v3: [Parakeet.cpp](https://github.com/mudler/parakeet.cpp) v0.5.0, native, CPU or Vulkan. ONNX remains the default.

Run `hyprwhspr setup`; choose **Parakeet → Parakeet.cpp**. Or: `hyprwhspr setup auto --backend parakeet-cpp`. Switching engines keeps ONNX settings and models.

```jsonc
{
    "transcription_backend": "parakeet-cpp",
    "parakeet_cpp_device": "auto"   // auto | cpu | vulkan
}
```

`auto` keeps an installed runtime. Otherwise: Vulkan when found, CPU when not. If Vulkan fails to load during setup, `auto` falls back to CPU; explicit `vulkan` reports why.

One model, pinned: `tdt-0.6b-v3-q8_0` (~941 MB). Language is detected; prompts, translation and forced language do nothing here.

Linux x64 and ARM64. Downloads are checksummed at setup; dictation never downloads.

Runtime stored in: `~/.local/share/hyprwhspr/runtime/parakeet-cpp/`

Model stored in: `~/.local/share/hyprwhspr/parakeet-cpp/models/`
