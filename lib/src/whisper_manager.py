"""
Whisper manager for hyprwhspr
Now supports two backends:
- pywhispercpp (in-process, model kept hot, default)
- whisper_cpp CLI (subprocess fallback)
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
        self.fallback_cli = self.config.get_setting('fallback_cli', False)
        
        # Backend-specific attributes
        self._pywhisper_model = None  # pywhispercpp model (in-process)
        self.whisper_binary = None    # CLI binary path (fallback)
        self.model_path = None        # CLI model path (fallback)
        self.temp_dir = None
        
        # State
        self.ready = False
        
    def initialize(self) -> bool:
        """Initialize the whisper manager and check dependencies"""
        try:
            self.temp_dir = self.config.get_temp_directory()
            
            # Detect GPU backend for logging
            gpu_backend = self._detect_gpu_backend()
            
            if not self.fallback_cli:
                # Try pywhispercpp backend (default)
                try:
                    from pywhispercpp.model import Model
                    
                    print(f"[pywhispercpp] Initializing model: {self.current_model}")
                    print(f"[pywhispercpp] Detected GPU backend: {gpu_backend}")
                    
                    self._pywhisper_model = Model(
                        model=self.current_model,
                        n_threads=self.config.get_setting('threads', 4),
                        redirect_whispercpp_logs_to=None  # suppress logs
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
                    print("[pywhispercpp] Falling back to CLI subprocess mode")
                    self.fallback_cli = True
                except Exception as e:
                    print(f"[pywhispercpp] Initialization failed: {e}")
                    print("[pywhispercpp] Falling back to CLI subprocess mode")
                    self.fallback_cli = True
            
            if self.fallback_cli:
                # Fallback to CLI subprocess (existing path)
                self.whisper_binary = self.config.get_whisper_binary_path()
                if not self.whisper_binary.exists():
                    print(f"ERROR: Whisper binary not found at: {self.whisper_binary}")
                    print("  Please build whisper.cpp first by running the build scripts")
                    return False
                
                self.model_path = self.config.get_whisper_model_path(self.current_model)
                if not self.model_path.exists():
                    print(f"ERROR: Whisper model not found at: {self.model_path}")
                    print(f"  Please download the {self.current_model} model first")
                    return False
                
                print(f"[CLI] Using model: {self.current_model} at {self.model_path}", flush=True)
                print(f"[CLI] Detected GPU backend: {gpu_backend}", flush=True)
                print(f"[BACKEND] Using CLI subprocess with {gpu_backend} acceleration", flush=True)
                import sys
                sys.stdout.flush()
                sys.stderr.flush()
                self.ready = True
                return True
            
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
        
        # Log which backend is being used for transcription
        if not self.fallback_cli:
            print("[TRANSCRIBE] Using pywhispercpp backend", flush=True)
        else:
            print("[TRANSCRIBE] Using CLI subprocess backend", flush=True)
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
        
        if self._pywhisper_model:
            # Use pywhispercpp (in-process, direct numpy array)
            try:
                segments = self._pywhisper_model.transcribe(audio_data)
                return ' '.join(seg.text for seg in segments).strip()
            except Exception as e:
                print(f"ERROR: pywhispercpp transcription failed: {e}")
                return ""
        else:
            # Fallback to CLI subprocess (existing path)
            return self._transcribe_via_cli(audio_data, sample_rate)
    
    def _transcribe_via_cli(self, audio_data: np.ndarray, sample_rate: int) -> str:
        """Transcribe using CLI subprocess (fallback method)"""
        # Create temporary WAV file
        with tempfile.NamedTemporaryFile(suffix='.wav', delete=False, dir=self.temp_dir) as temp_file:
            temp_wav_path = temp_file.name
            
        try:
            # Save audio data as WAV file
            self._save_audio_as_wav(audio_data, temp_wav_path, sample_rate)
            
            # Run whisper.cpp transcription
            transcription = self._run_whisper_cli(temp_wav_path)
            
            return transcription.strip() if transcription else ""
            
        finally:
            # Clean up temporary file
            try:
                os.unlink(temp_wav_path)
            except:
                pass  # Ignore cleanup errors
    
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
            else:
                # Existing CLI path validation
                new_model_path = self.config.get_whisper_model_path(model_name)
                
                if not new_model_path.exists():
                    print(f"ERROR: Model {model_name} not found at {new_model_path}")
                    return False
                
                self.model_path = new_model_path
                print(f"[CLI] Switched to model: {model_name}")
            
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
        models_dir = self.config.get_whisper_model_path('').parent
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
        if self._pywhisper_model:
            return f"pywhispercpp (in-process, model: {self.current_model})"
        else:
            return f"CLI subprocess (model: {self.current_model})"