"""
Realtime WebSocket transcription backend.

Streams audio to a provider WebSocket (OpenAI, Gemini Live, ElevenLabs)
during capture via the streaming callback; transcribe() then commits the
buffered audio and waits for the final transcript.
"""

from typing import Callable, Optional

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')

try:
    from ..backend_utils import normalize_backend
    from ..credential_manager import get_credential
    from ..provider_registry import get_provider
except ImportError:
    from backend_utils import normalize_backend
    from credential_manager import get_credential
    from provider_registry import get_provider

from .base import TranscriptionBackend


class RealtimeWsBackend(TranscriptionBackend):
    """Streaming WebSocket backend; reconnects by full re-initialization on resume."""

    name = 'realtime-ws'
    is_local = False
    reinit_on_resume = True

    def __init__(self, manager):
        super().__init__(manager)
        # Realtime WebSocket client
        self._realtime_client = None
        self._realtime_streaming_callback = None
        # Connection parameters used for reconnect-on-demand.
        # (Stored in-memory only; do not log API keys.)
        self._realtime_connect_params = None

    @property
    def _realtime_partial_callback(self):
        # Owned by the manager so it survives backend re-creation on resume
        return self._manager._realtime_partial_callback

    def initialize(self) -> bool:
        """Configure the Realtime WebSocket backend and connect the client"""
        # Validate WebSocket configuration
        provider_id = self.config.get_setting('websocket_provider')
        model_id = self.config.get_setting('websocket_model')

        if not provider_id:
            print('ERROR: Realtime WebSocket backend selected but websocket_provider not configured')
            return False

        if not model_id:
            print('ERROR: Realtime WebSocket backend selected but websocket_model not configured')
            return False

        # Get API key from credential manager
        api_key = get_credential(provider_id)
        if not api_key:
            print(f'ERROR: Provider {provider_id} configured but API key not found in credential store')
            return False

        # Select appropriate client based on provider
        if provider_id == 'google':
            # Use Gemini Live API client
            try:
                from ..gemini_realtime_client import GeminiRealtimeClient
            except ImportError:
                from gemini_realtime_client import GeminiRealtimeClient

            realtime_mode = self.config.get_setting('realtime_mode', 'transcribe')
            self._realtime_client = GeminiRealtimeClient(mode=realtime_mode)

            # Get WebSocket URL
            websocket_url = self.config.get_setting('websocket_url')
            if not websocket_url:
                provider = get_provider(provider_id)
                if provider and 'websocket_endpoint' in provider:
                    websocket_url = provider['websocket_endpoint']
                else:
                    websocket_url = 'wss://generativelanguage.googleapis.com/ws/google.ai.generativelanguage.v1beta.GenerativeService.BidiGenerateContent'

            # Build instructions
            instructions_parts = []
            whisper_prompt = self.config.get_setting('whisper_prompt', None)
            if whisper_prompt:
                instructions_parts.append(whisper_prompt)

            language = self.config.get_setting('language', None)
            if language:
                instructions_parts.append(f"Transcribe in {language} language.")

            instructions = ' '.join(instructions_parts) if instructions_parts else None

            # Set language
            self._realtime_client.language = language

            # Set buffer max seconds
            buffer_max = self.config.get_setting('realtime_buffer_max_seconds', 5)
            self._realtime_client.set_max_buffer_seconds(buffer_max)

            # Connect (API key goes in URL query param, handled by client)
            self._realtime_connect_params = {
                'websocket_url': websocket_url,
                'api_key': api_key,
                'model_id': model_id,
                'instructions': instructions,
            }
            if not self._realtime_client.connect(websocket_url, api_key, model_id, instructions):
                print('ERROR: Failed to connect to Gemini Live API')
                try:
                    self._realtime_client.close()
                except Exception:
                    pass
                self._realtime_client = None
                return False

            def _send_direct(audio_chunk: np.ndarray):
                """Send audio directly to Gemini; client resamples if needed."""
                try:
                    self._realtime_client.append_audio(audio_chunk)
                except Exception as e:
                    print(f'[GEMINI] Streaming error: {e}', flush=True)

            _send_direct.set_input_sample_rate = self._realtime_client.set_input_sample_rate
            self._realtime_streaming_callback = _send_direct

        elif provider_id == 'elevenlabs':
            # Use ElevenLabs-specific client (Scribe v2 Realtime)
            try:
                from ..elevenlabs_realtime_client import ElevenLabsRealtimeClient
            except ImportError:
                from elevenlabs_realtime_client import ElevenLabsRealtimeClient

            self._realtime_client = ElevenLabsRealtimeClient()

            # Get WebSocket URL
            websocket_url = self.config.get_setting('websocket_url')
            if not websocket_url:
                provider = get_provider(provider_id)
                if provider and 'websocket_endpoint' in provider:
                    websocket_url = provider['websocket_endpoint']
                else:
                    websocket_url = 'wss://api.elevenlabs.io/v1/speech-to-text/realtime'

            # Set language (used at connection time via query params)
            language = self.config.get_setting('language', None)
            self._realtime_client.language = language

            # Set buffer max seconds
            buffer_max = self.config.get_setting('realtime_buffer_max_seconds', 5)
            self._realtime_client.set_max_buffer_seconds(buffer_max)

            # Connect (ElevenLabs doesn't use instructions)
            self._realtime_connect_params = {
                'websocket_url': websocket_url,
                'api_key': api_key,
                'model_id': model_id,
                'instructions': None,
            }
            if not self._realtime_client.connect(websocket_url, api_key, model_id, None):
                print('ERROR: Failed to connect to ElevenLabs Realtime WebSocket')
                try:
                    self._realtime_client.close()
                except Exception:
                    pass
                self._realtime_client = None
                return False

            def _send_direct(audio_chunk: np.ndarray):
                """Send audio directly to ElevenLabs; client resamples if needed."""
                try:
                    self._realtime_client.append_audio(audio_chunk)
                except Exception as e:
                    print(f'[ELEVENLABS] Streaming error: {e}', flush=True)

            _send_direct.set_input_sample_rate = self._realtime_client.set_input_sample_rate
            self._realtime_streaming_callback = _send_direct

        else:
            # Use OpenAI-compatible client (default)
            try:
                from ..realtime_client import RealtimeClient
            except ImportError:
                from realtime_client import RealtimeClient

            # Initialize RealtimeClient with mode
            realtime_mode = self.config.get_setting('realtime_mode', 'transcribe')
            if provider_id == 'openai' and model_id == 'gpt-realtime-whisper' and realtime_mode != 'transcribe':
                print('ERROR: gpt-realtime-whisper is supported only with realtime_mode="transcribe"', flush=True)
                return False
            self._realtime_client = RealtimeClient(mode=realtime_mode)

            # Get WebSocket URL
            websocket_url = self.config.get_setting('websocket_url')
            if not websocket_url:
                # For custom providers, websocket_url must be explicitly set
                if provider_id == 'custom':
                    print('ERROR: Custom realtime backend requires websocket_url to be configured')
                    return False

                # For known providers, derive from provider registry
                try:
                    websocket_url = self._get_websocket_url(provider_id, model_id, realtime_mode)
                except Exception as e:
                    print(f'ERROR: Failed to derive WebSocket URL: {e}')
                    return False

            # Build instructions from whisper_prompt and language
            instructions_parts = []
            whisper_prompt = self.config.get_setting('whisper_prompt', None)
            if whisper_prompt:
                instructions_parts.append(whisper_prompt)

            language = self.config.get_setting('language', None)
            if language:
                instructions_parts.append(f"Transcribe in {language} language.")

            instructions = ' '.join(instructions_parts) if instructions_parts else None

            # Set language in realtime client (for session.update)
            self._realtime_client.language = language

            delay = self.config.get_setting('realtime_transcription_delay', 'low')
            self._realtime_client.set_transcription_delay(delay)
            if self._is_realtime_whisper_preview_enabled(provider_id, model_id, realtime_mode):
                self._realtime_client.set_partial_transcript_callback(self._realtime_partial_callback)
            else:
                self._realtime_client.set_partial_transcript_callback(None)
                self._clear_realtime_partial_preview()

            # Set buffer max seconds
            buffer_max = self.config.get_setting('realtime_buffer_max_seconds', 5)
            self._realtime_client.set_max_buffer_seconds(buffer_max)

            # Connect
            self._realtime_connect_params = {
                'websocket_url': websocket_url,
                'api_key': api_key,
                'model_id': model_id,
                'instructions': instructions,
            }
            if not self._realtime_client.connect(websocket_url, api_key, model_id, instructions):
                print('ERROR: Failed to connect to Realtime WebSocket')
                # Clean up failed client
                try:
                    self._realtime_client.close()
                except Exception:
                    pass
                self._realtime_client = None
                return False

            def _send_direct(audio_chunk: np.ndarray):
                """Send audio to realtime client; client handles resampling/queueing."""
                try:
                    self._realtime_client.append_audio(audio_chunk)
                except Exception as e:
                    print(f'[REALTIME] Streaming error: {e}', flush=True)

            _send_direct.set_input_sample_rate = self._realtime_client.set_input_sample_rate
            self._realtime_streaming_callback = _send_direct

        print(f'[BACKEND] Using Realtime WebSocket: {websocket_url}')
        print(f'[REALTIME] Model: {model_id}, Provider: {provider_id}')

        # Explicitly set to None to avoid confusion with top-level model setting
        self.current_model = None
        self.ready = True
        return True

    def _get_websocket_url(self, provider_id: str, model_id: str, mode: str = 'transcribe') -> str:
        """
        Get WebSocket URL for a provider and model.
        
        Args:
            provider_id: Provider identifier (e.g., 'openai')
            model_id: Model identifier (e.g., 'gpt-realtime-whisper')
            mode: 'transcribe' or 'converse'
        
        Returns:
            WebSocket URL with appropriate query parameters
        """
        provider = get_provider(provider_id)
        if not provider:
            raise ValueError(f"Unknown provider: {provider_id}")
        
        # Check if provider has explicit websocket_endpoint
        if 'websocket_endpoint' in provider:
            base_url = provider['websocket_endpoint']
        else:
            # Derive from HTTP endpoint
            endpoint = provider.get('endpoint', '')
            if not endpoint:
                raise ValueError(f"Provider {provider_id} has no endpoint or websocket_endpoint")
            
            # Transform: https:// -> wss://, replace /audio/transcriptions -> /realtime
            base_url = endpoint.replace('https://', 'wss://').replace('http://', 'ws://')
            if '/audio/transcriptions' in base_url:
                base_url = base_url.replace('/audio/transcriptions', '/realtime')
            elif '/transcriptions' in base_url:
                base_url = base_url.replace('/transcriptions', '/realtime')
        
        # Build query parameters based on mode
        if mode == 'transcribe':
            # Transcription mode uses intent=transcription
            return f"{base_url}?intent=transcription"
        else:
            # Converse mode uses model parameter
            return f"{base_url}?model={model_id}"

    def transcribe(self, _audio_data: np.ndarray, _sample_rate: int = 16000, language_override: Optional[str] = None) -> str:
        """
        Transcribe audio using Realtime WebSocket backend.
        
        Note: For realtime-ws backend, audio should be streamed during capture
        via the streaming callback. This method handles the commit and wait.
        
        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data (should be 16000)
            language_override: Optional language code to override config language
        
        Returns:
            Transcribed text string
        """
        if not self._realtime_client:
            print('[REALTIME] Client not initialized')
            return ""
        
        if not self._realtime_client.connected:
            print('[REALTIME] Client not connected')
            return ""
        
        try:
            # Update language if override provided.
            # Some clients (e.g. Gemini) bake language into the setup message at
            # connect time and cannot update it after audio has been streamed —
            # doing so would trigger a reconnect and silently drop the audio.
            if language_override is not None:
                if getattr(self._realtime_client, 'supports_mid_session_language_update', True):
                    self._realtime_client.update_language(language_override)
                else:
                    print(
                        f'[REALTIME] Provider does not support mid-session language override '
                        f'(requested: {language_override}); change will take effect on next session',
                        flush=True,
                    )
            
            # Get timeout from config
            timeout = self.config.get_setting('realtime_timeout', 30)
            
            # Commit and get text (audio was already streamed via callback)
            transcription = self._realtime_client.commit_and_get_text(timeout=timeout)
            
            return transcription.strip()
            
        except Exception as e:
            print(f'[REALTIME] Transcription failed: {e}')
            return ""

    def get_streaming_callback(self) -> Optional[Callable]:
        """
        Get the streaming callback for realtime-ws backend.
        
        Returns:
            Callback function if realtime-ws backend is active, None otherwise
        """
        backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
        backend = normalize_backend(backend)
        
        if backend == 'realtime-ws' and self._realtime_client:
            # If the server closed the socket while idle, reconnect on-demand here
            # (before we start capturing audio) to avoid dropping the first chunks.
            if not self._realtime_client.connected:
                if not self._reconnect_realtime_client():
                    return None

            # Clear server buffer before starting new recording
            self._realtime_client.clear_audio_buffer()
            return self._realtime_streaming_callback
        return None

    def apply_partial_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Apply the partial-preview callback to a connected client (storage lives on the manager)."""
        if self._realtime_client and hasattr(self._realtime_client, 'set_partial_transcript_callback'):
            provider_id = self.config.get_setting('websocket_provider')
            model_id = self.config.get_setting('websocket_model')
            realtime_mode = self.config.get_setting('realtime_mode', 'transcribe')
            if self._is_realtime_whisper_preview_enabled(provider_id, model_id, realtime_mode):
                self._realtime_client.set_partial_transcript_callback(callback)
            else:
                self._realtime_client.set_partial_transcript_callback(None)
                self._clear_realtime_partial_preview()

    def _is_realtime_whisper_preview_enabled(self, provider_id: str, model_id: str, realtime_mode: str) -> bool:
        return (
            self.config.get_setting('mic_osd_enabled', True)
            and provider_id == 'openai'
            and model_id == 'gpt-realtime-whisper'
            and realtime_mode == 'transcribe'
            and self._realtime_partial_callback is not None
        )

    def _clear_realtime_partial_preview(self) -> None:
        if not self._realtime_partial_callback:
            return
        try:
            self._realtime_partial_callback("")
        except Exception as e:
            print(f'[REALTIME] Failed to clear partial transcript preview: {e}', flush=True)

    def _reconnect_realtime_client(self) -> bool:
        """Reconnect realtime client using stored connect params."""
        if not self._realtime_client:
            return False

        params = self._realtime_connect_params or {}
        websocket_url = params.get('websocket_url')
        api_key = params.get('api_key')
        model_id = params.get('model_id')
        instructions = params.get('instructions')

        if not (websocket_url and api_key and model_id):
            print('[REALTIME] Missing connection parameters; cannot reconnect', flush=True)
            return False

        try:
            # Best-effort: close any stale connection first
            try:
                self._realtime_client.close()
            except Exception:
                pass

            if not self._realtime_client.connect(websocket_url, api_key, model_id, instructions):
                print('[REALTIME] Reconnect failed', flush=True)
                return False

            print('[REALTIME] Reconnected on-demand', flush=True)
            return True
        except Exception as e:
            print(f'[REALTIME] Reconnect failed: {e}', flush=True)
            return False

    def close(self) -> None:
        """Cleanup Realtime WebSocket client"""
        if self._realtime_client:
            try:
                self._realtime_client.close()
                self._realtime_client = None
                self._realtime_streaming_callback = None
            except Exception as e:
                print(f"[WARN] Failed to cleanup realtime client: {e}")

    def update_language(self, language: Optional[str]) -> None:
        """Apply a language override to a connected client (no-op otherwise)."""
        if self._realtime_client:
            self._realtime_client.update_language(language)

    def reinitialize(self) -> bool:
        """Re-establish the connection after suspend/resume (full re-init)."""
        return self._manager.initialize()

    def cleanup(self) -> None:
        self.close()

    @property
    def is_loaded(self) -> bool:
        return self._realtime_client is not None
