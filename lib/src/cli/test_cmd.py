"""
Microphone / backend connectivity test command for hyprwhspr
"""

from pathlib import Path

try:
    from ..config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from ..backend_installer import PYWHISPERCPP_MODELS_DIR
except ImportError:
    from backend_installer import PYWHISPERCPP_MODELS_DIR

try:
    from ..backend_utils import normalize_backend
except ImportError:
    from backend_utils import normalize_backend

try:
    from ..credential_manager import get_credential
except ImportError:
    from credential_manager import get_credential

try:
    from ..output_control import log_info, log_success, log_warning, log_error
except ImportError:
    from output_control import log_info, log_success, log_warning, log_error

from ._shared import HYPRWHSPR_ROOT


# ==================== Test Command ====================

def test_command(live: bool = False, mic_only: bool = False):
    """Test microphone and backend connectivity end-to-end"""
    import time
    import wave
    from io import BytesIO

    print("\n" + "="*60)
    print("hyprwhspr Diagnostic Test")
    print("="*60)

    all_passed = True

    # ===== MICROPHONE TEST =====
    print("\n[Microphone]")

    from audio_capture import AudioCapture

    # Ensure audio is defined on all code paths (e.g., no devices found)
    audio = None

    try:
        # Check for available devices
        devices = AudioCapture.get_available_input_devices()
        if not devices:
            log_error("No input devices found")
            all_passed = False
        else:
            log_success(f"Found {len(devices)} input device(s)")

            # Get configured device from config
            config = ConfigManager()
            device_id = config.get_setting('audio_device_id', None)

            # Initialize audio capture
            audio = AudioCapture(device_id=device_id, config_manager=config)

            if audio.is_available():
                device_info = audio.get_current_device_info()
                if device_info:
                    log_success(f"Using: {device_info['name']}")
                else:
                    log_success("Audio device available")
            else:
                log_error("Failed to initialize audio capture")
                all_passed = False

    except Exception as e:
        log_error(f"Microphone test failed: {e}")
        all_passed = False
        audio = None

    # If mic-only, stop here
    if mic_only:
        print("\n" + "-"*60)
        if all_passed:
            log_success("Microphone test passed")
        else:
            log_error("Microphone test failed")
        return all_passed

    # ===== BACKEND TEST =====
    print("\n[Backend]")

    config = ConfigManager()
    backend = config.get_setting('transcription_backend', 'pywhispercpp')
    backend = normalize_backend(backend)

    log_info(f"Configured backend: {backend}")

    backend_ready = False

    if backend == 'rest-api':
        # Test REST API connectivity
        endpoint_url = config.get_setting('rest_endpoint_url')
        if not endpoint_url:
            log_error("REST endpoint URL not configured")
            all_passed = False
        else:
            log_success(f"Endpoint: {endpoint_url}")

            # Check credentials
            provider_id = config.get_setting('rest_api_provider')
            if provider_id:
                api_key = get_credential(provider_id)
                if api_key:
                    log_success(f"Credentials configured (provider: {provider_id})")
                    backend_ready = True
                else:
                    log_error(f"API key not found for provider: {provider_id}")
                    all_passed = False
            else:
                # Check for legacy api key
                api_key = config.get_setting('rest_api_key')
                if api_key:
                    log_success("Credentials configured (legacy)")
                    backend_ready = True
                else:
                    log_warning("No API credentials configured")
                    # May still work if endpoint doesn't require auth
                    backend_ready = True

    elif backend == 'realtime-ws':
        # Test WebSocket configuration
        provider_id = config.get_setting('websocket_provider')
        model_id = config.get_setting('websocket_model')

        if not provider_id:
            log_error("WebSocket provider not configured")
            all_passed = False
        elif not model_id:
            log_error("WebSocket model not configured")
            all_passed = False
        else:
            api_key = get_credential(provider_id)
            if api_key:
                log_success(f"Provider: {provider_id}, Model: {model_id}")
                log_success("Credentials configured")
                backend_ready = True
            else:
                log_error(f"API key not found for provider: {provider_id}")
                all_passed = False

    elif backend == 'onnx-asr':
        # Test ONNX-ASR model availability
        try:
            import onnx_asr
            model_name = config.get_setting('onnx_asr_model', 'nemo-parakeet-tdt-0.6b-v3')
            log_success(f"onnx-asr available, model: {model_name}")
            backend_ready = True
        except ImportError:
            log_error("onnx-asr not installed")
            all_passed = False

    elif backend == 'faster-whisper':
        # Test faster-whisper availability
        try:
            import faster_whisper  # noqa: F401
            model_name = config.get_setting('faster_whisper_model', 'base')
            log_success(f"faster-whisper available, model: {model_name}")
            backend_ready = True
        except ImportError:
            log_error("faster-whisper not installed. Run: hyprwhspr setup")
            all_passed = False

    elif backend == 'cohere-transcribe':
        # Test Cohere Transcribe availability
        try:
            from transformers import AutoModelForSpeechSeq2Seq  # noqa: F401
            log_success("Cohere Transcribe (transformers) available")
            hf_cache = Path.home() / '.cache' / 'huggingface' / 'hub' / 'models--CohereLabs--cohere-transcribe-03-2026'
            if hf_cache.exists():
                log_success("Model weights cached in ~/.cache/huggingface/hub/")
            else:
                log_warning("Model weights not yet downloaded — will fetch on first use (~4 GB)")
            backend_ready = True
        except ImportError:
            log_error("transformers not installed. Run: hyprwhspr setup and select cohere-transcribe")
            all_passed = False

    elif backend in ('pywhispercpp', 'nvidia', 'cpu', 'vulkan'):
        # Test pywhispercpp model availability (covers all local whisper variants)
        try:
            try:
                from pywhispercpp.model import Model
            except ImportError:
                from pywhispercpp import Model

            model_name = config.get_setting('model', 'base')
            model_file = PYWHISPERCPP_MODELS_DIR / f"ggml-{model_name}.bin"

            # Try English-only variant if base not found
            if not model_file.exists() and not model_name.endswith('.en'):
                model_file = PYWHISPERCPP_MODELS_DIR / f"ggml-{model_name}.en.bin"

            if model_file.exists():
                log_success(f"pywhispercpp available, model: {model_name}")
                backend_ready = True
            else:
                log_error(f"Model file not found: {model_file}")
                log_info(f"Download with: hyprwhspr model download {model_name}")
                all_passed = False
        except ImportError:
            log_error("pywhispercpp not installed")
            all_passed = False
    else:
        log_warning(f"Unknown backend: {backend}")
        all_passed = False

    # ===== TRANSCRIPTION TEST =====
    print("\n[Transcription]")

    if not backend_ready:
        log_warning("Skipping transcription test (backend not ready)")
    else:
        # Get audio data - either from test.wav or live recording
        audio_data = None
        audio_source = None

        if live:
            # Record live audio
            if audio and audio.is_available():
                print("  Recording for 3 seconds... speak now!")
                try:
                    audio.start_recording()
                    time.sleep(3.0)
                    audio_data = audio.stop_recording()

                    if audio_data is not None and len(audio_data) > 0:
                        # Calculate audio level
                        import numpy as np
                        rms = np.sqrt(np.mean(audio_data**2))
                        db = 20 * np.log10(max(rms, 1e-10))
                        log_success(f"Recorded {len(audio_data)/16000:.1f}s audio (level: {db:.0f}dB)")
                        audio_source = "live recording"

                        # Warn if audio is very quiet (likely silence)
                        if db < -40:
                            log_warning("Audio level very low - check microphone")
                    else:
                        log_error("No audio data captured")
                        all_passed = False
                except Exception as e:
                    log_error(f"Recording failed: {e}")
                    all_passed = False
            else:
                log_error("Cannot record - audio capture not available")
                all_passed = False
        else:
            # Use test.wav
            test_wav_path = Path(HYPRWHSPR_ROOT) / 'share' / 'assets' / 'test.wav'

            if not test_wav_path.exists():
                log_error(f"Test audio file not found: {test_wav_path}")
                log_info("Use --live to record audio instead")
                all_passed = False
            else:
                try:
                    import numpy as np
                    with wave.open(str(test_wav_path), 'rb') as wf:
                        # Read audio data
                        frames = wf.readframes(wf.getnframes())
                        sample_rate = wf.getframerate()

                        # Convert to float32 numpy array
                        sample_width = wf.getsampwidth()
                        if sample_width == 2:  # 16-bit
                            audio_data = np.frombuffer(frames, dtype=np.int16).astype(np.float32) / 32768.0
                        elif sample_width == 4:  # 32-bit
                            audio_data = np.frombuffer(frames, dtype=np.int32).astype(np.float32) / 2147483648.0
                        else:
                            audio_data = np.frombuffer(frames, dtype=np.float32)

                        # Resample to 16kHz if needed
                        if sample_rate != 16000:
                            from scipy import signal
                            audio_data = signal.resample(audio_data, int(len(audio_data) * 16000 / sample_rate))

                        duration = len(audio_data) / 16000
                        log_success(f"Loaded test.wav ({duration:.1f}s)")
                        audio_source = "test.wav"

                except Exception as e:
                    log_error(f"Failed to load test.wav: {e}")
                    all_passed = False

        # Transcribe if we have audio
        if audio_data is not None and len(audio_data) > 0:
            from whisper_manager import WhisperManager

            try:
                log_info("Initializing backend...")
                whisper = WhisperManager(config_manager=config)

                if not whisper.initialize():
                    log_error("Failed to initialize transcription backend")
                    all_passed = False
                else:
                    duration = len(audio_data) / 16000
                    if duration > 5:
                        log_info(f"Transcribing {duration:.0f}s of audio (this may take a moment)...")
                    else:
                        log_info("Transcribing...")

                    # For realtime-ws, we need to handle differently
                    if backend == 'realtime-ws':
                        # Realtime requires streaming - not ideal for test
                        # Just verify connection worked during initialize()
                        log_success("WebSocket connected successfully")
                        log_info("(Realtime transcription requires streaming audio)")
                        whisper.cleanup()
                    else:
                        result = whisper.transcribe_audio(audio_data)

                        if result:
                            # Clean up the result for display
                            result_clean = result.strip()
                            if result_clean:
                                log_success("Transcription successful")
                                print(f"  -> \"{result_clean}\"")
                            else:
                                log_warning("Transcription returned empty result")
                                log_info("This may be normal if audio was silence")
                        else:
                            log_error("Transcription returned no result")
                            all_passed = False

                        # Cleanup
                        if hasattr(whisper, 'cleanup'):
                            whisper.cleanup()

            except Exception as e:
                log_error(f"Transcription test failed: {e}")
                import traceback
                traceback.print_exc()
                all_passed = False

    # ===== SUMMARY =====
    print("\n" + "-"*60)
    if all_passed:
        log_success("All tests passed!")
    else:
        log_error("Some tests failed")

    return all_passed
