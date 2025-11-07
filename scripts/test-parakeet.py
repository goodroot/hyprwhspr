"""
Test script for Parakeet ONNX backend in hyprwhspr

hyprwhspr is a native speech-to-text application for Arch Linux/Omarchy
with Hyprland desktop environment. This script tests the Parakeet TDT v3
model using the onnx-asr library.

Usage:
    python3 test-parakeet.py

Note: This script requires the Parakeet model to be downloaded first.
Model: nemo-parakeet-tdt-0.6b-v3
"""

import onnx_asr

model = onnx_asr.load_model("nemo-parakeet-tdt-0.6b-v3")
print(model.recognize("2086-149220-0033.wav"))
