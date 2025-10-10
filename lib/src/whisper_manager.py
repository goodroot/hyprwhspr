"""
Whisper manager for hyprwhspr
PyWhisperCPP-only backend (in-process, model kept hot)
"""

import subprocess
import tempfile
import os
import wave
import numpy as np
import shutil
from typing import Optional

try:
    from .config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager


class WhisperManager:
    """Manages whisper transcription with dual backend support"""
    
    def __init__(self, config_manager: Optional[ConfigManager] = None):
        if config_manager is None:
            self.config = ConfigManager()
        else:
            self.config = config_manager
            
        # Whisper configuration
        self.current_model = self.config.get_setting('model', 'base')
        # Backend-specific attributes (pywhispercpp only)
        self._pywhisper_model = None
        self.temp_dir = None
        
        # State
        self.ready = False
        
    def initialize(self) -> bool:
        """Initialize the whisper manager and check dependencies"""
        try:
            self.temp_dir = self.config.get_temp_directory()
            
            # Detect GPU backend for logging
            gpu_backend = self._detect_gpu_backend()

            try:
                from pywhispercpp.model import Model

                print(f"[pywhispercpp] Initializing model: {self.current_model}")
                print(f"[pywhispercpp] Detected GPU backend: {gpu_backend}")

                self._pywhisper_model = Model(
                    model=self.current_model,
                    n_threads=self.config.get_setting('threads', 4),
                    redirect_whispercpp_logs_to=None
                )

                print(f"[pywhispercpp] Model loaded successfully", flush=True)
                print(f"[BACKEND] Using pywhispercpp (in-process) with {gpu_backend} acceleration", flush=True)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                self.ready = True
                return True

            except ImportError as e:
                print(f"[pywhispercpp] Import failed: {e}")
                return False
            except Exception as e:
                print(f"[pywhispercpp] Initialization failed: {e}")
                return False
            
        except Exception as e:
            print(f"ERROR: Failed to initialize Whisper manager: {e}")
            return False
    
    def _detect_gpu_backend(self) -> str:
        """Detect available GPU backend for logging purposes"""
        # Check NVIDIA CUDA
        if shutil.which('nvidia-smi'):
            return "CUDA (NVIDIA)"
        # Check AMD ROCm
        if shutil.which('rocm-smi') or os.path.exists('/opt/rocm'):
            return "ROCm (AMD)"
        # Check Vulkan
        if shutil.which('vulkaninfo'):
            return "Vulkan"
        return "CPU"
    
    def is_ready(self) -> bool:
        """Check if whisper is ready for transcription"""
        return self.ready
    
    def transcribe_audio(self, audio_data: np.ndarray, sample_rate: int = 16000) -> str:
        """
        Transcribe audio data using whisper
        
        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data
            
        Returns:
            Transcribed text string
        """
        if not self.ready:
            raise RuntimeError("Whisper manager not initialized")
        
        print("[TRANSCRIBE] Using pywhispercpp backend", flush=True)
        import sys
        sys.stdout.flush()
        sys.stderr.flush()
        
        # Check if we have valid audio data
        if audio_data is None:
            print("No audio data provided to transcribe")
            return ""
        
        if len(audio_data) == 0:
            print("Empty audio data provided to transcribe")
            return ""
        
        # Check if audio is too short (less than 0.1 seconds)
        min_samples = int(sample_rate * 0.1)  # 0.1 seconds minimum
        if len(audio_data) < min_samples:
            print(f"Audio too short: {len(audio_data)} samples (minimum {min_samples})")
            return ""
        
        try:
            segments = self._pywhisper_model.transcribe(audio_data)
            return ' '.join(seg.text for seg in segments).strip()
        except Exception as e:
            print(f"ERROR: pywhispercpp transcription failed: {e}")
            return ""

    def set_threads(self, num_threads: int) -> bool:
        """Update the number of threads used by the backend.

        Attempts to update the active model; if unsupported, reloads the model with
        the same parameters but a new thread count.
        """
        try:
            # Persist to config
            self.config.set_setting('threads', int(num_threads))

            # Try dynamic update if the backend supports it
            try:
                if hasattr(self._pywhisper_model, 'set_n_threads'):
                    self._pywhisper_model.set_n_threads(int(num_threads))
                    print(f"[pywhispercpp] Threads updated to {int(num_threads)}")
                    return True
            except Exception:
                pass

            # Fallback: reload model with new thread count
            from pywhispercpp.model import Model
            self._pywhisper_model = Model(
                model=self.current_model,
                n_threads=int(num_threads),
                redirect_whispercpp_logs_to=None
            )
            print(f"[pywhispercpp] Model reloaded with threads={int(num_threads)}")
            return True
        except Exception as e:
            print(f"ERROR: Failed to set threads: {e}")
            return False
    
    def _save_audio_as_wav(self, audio_data: np.ndarray, filepath: str, sample_rate: int):
        """Save numpy audio data as a WAV file"""
        # Convert float32 to int16 for WAV format
        if audio_data.dtype == np.float32:
            # Scale from [-1, 1] to [-32768, 32767]
            audio_int16 = (audio_data * 32767).astype(np.int16)
        else:
            audio_int16 = audio_data.astype(np.int16)
        
        with wave.open(filepath, 'wb') as wav_file:
            wav_file.setnchannels(1)  # Mono
            wav_file.setsampwidth(2)  # 16-bit
            wav_file.setframerate(sample_rate)
            wav_file.writeframes(audio_int16.tobytes())
    
    def _run_whisper_cli(self, audio_file_path: str) -> str:
        """Run whisper.cpp CLI on the given audio file"""
        try:
            # Get whisper prompt from config or use default
            whisper_prompt = self.config.get_setting(
                'whisper_prompt', 
                'Transcribe with proper capitalization, including sentence beginnings, proper nouns, titles, and standard English capitalization rules.'
            )
            
            # Construct whisper.cpp command
            cmd = [
                str(self.whisper_binary),
                '-m', str(self.model_path),
                '-f', audio_file_path,
                '--output-txt',
                '--language', 'en',
                '--threads', str(self.config.get_setting('threads', 4)),
                '--prompt', whisper_prompt
            ]
            
            # Run the command
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=30  # 30 second timeout
            )
            
            if result.returncode == 0:
                # Try to read the output txt file
                txt_file = audio_file_path + '.txt'
                if os.path.exists(txt_file):
                    with open(txt_file, 'r', encoding='utf-8') as f:
                        transcription = f.read().strip()
                    # Clean up the txt file
                    os.unlink(txt_file)
                    return transcription
                else:
                    # Fall back to stdout if no txt file
                    return result.stdout.strip()
            else:
                print(f"Whisper command failed with return code {result.returncode}")
                print(f"stderr: {result.stderr}")
                return ""
                
        except subprocess.TimeoutExpired:
            print("Whisper transcription timed out")
            return ""
        except Exception as e:
            print(f"Error running whisper: {e}")
            return ""
    
    def set_model(self, model_name: str) -> bool:
        """
        Change the whisper model
        
        Args:
            model_name: Name of the model (e.g., 'base', 'small')
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self._pywhisper_model:
                # Reinitialize pywhispercpp model
                try:
                    from pywhispercpp.model import Model
                    self._pywhisper_model = Model(
                        model=model_name,
                        n_threads=self.config.get_setting('threads', 4),
                        redirect_whispercpp_logs_to=None
                    )
                    print(f"[pywhispercpp] Switched to model: {model_name}")
                except Exception as e:
                    print(f"ERROR: Failed to load model {model_name} with pywhispercpp: {e}")
                    return False
            # No CLI path; pywhispercpp only
            
            # Update current model
            self.current_model = model_name
            
            # Update config
            self.config.set_setting('model', model_name)
            
            return True
            
        except Exception as e:
            print(f"ERROR: Failed to set model {model_name}: {e}")
            return False
    
    def get_current_model(self) -> str:
        """Get the current model name"""
        return self.current_model
    
    def get_available_models(self) -> list:
        """Get list of available whisper models"""
        from pathlib import Path
        models_dir = Path.home() / '.local' / 'share' / 'pywhispercpp' / 'models'
        available_models = []
        
        # Look for the supported model files
        supported_models = ['tiny', 'base', 'small', 'medium', 'large']
        
        for model in supported_models:
            # Check for both English-only and multilingual versions
            model_files = [
                models_dir / f"ggml-{model}.en.bin",  # English-only
                models_dir / f"ggml-{model}.bin"      # Multilingual
            ]
            
            for model_file in model_files:
                if model_file.exists():
                    # Add model name with suffix if it's English-only
                    if model_file.name.endswith('.en.bin'):
                        model_name = f"{model}.en"
                    else:
                        model_name = model
                    
                    if model_name not in available_models:
                        available_models.append(model_name)
                    break  # Don't add both versions of same model
        
        return sorted(available_models)
    
    def get_backend_info(self) -> str:
        """Get information about the current backend"""
        return f"pywhispercpp (in-process, model: {self.current_model})"