# PLAN
- I'm interested in creating a pull request for this project on GitHub that's designed for Omarchy as a voice to text app. The purpose of the pull request is add parakeetv3 compatability to the project. 
- ParakeetV3 - https://huggingface.co/istupakov/parakeet-tdt-0.6b-v3-onnx

## Helpful Information About hyprwhspr
- https://github.com/goodroot/hyprwhspr
- Existing Architecture - hyprwhspr is designed as a system package:
  - /usr/lib/hyprwhspr/ - Main installation directory
  - /usr/lib/hyprwhspr/lib/ - Python application
  - ~/.local/share/pywhispercpp/models/ - Whisper models (user space)
  - ~/.config/hyprwhspr/ - User configuration
  - ~/.config/systemd/user/ - Systemd service

## Parakeet Information
- Input Type(s): 16kHz Audio Input Format(s): .wav and .flac audio formats Input Parameters: 1D (audio signal) Other Properties Related to Input: Monochannel audio
- Output Type(s): Text Output Format: String Output Parameters: 1D (text) Other Properties Related to Output: Punctuations and Capitalizations included.

