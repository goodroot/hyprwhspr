"""
Realtime WebSocket transcription backend.

Streams audio to a provider WebSocket (OpenAI, Gemini Live, ElevenLabs)
during capture via the streaming callback; transcribe() then commits the
buffered audio and waits for the final transcript.
"""

import time
from typing import Callable, Optional

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')

try:
    from ..backend_utils import normalize_backend
    from ..credential_manager import get_credential
    from ..openai_realtime_models import (
        is_continuous,
        is_transcription_only,
        uses_language_context,
    )
    from ..provider_registry import get_provider
except ImportError:
    from backend_utils import normalize_backend
    from credential_manager import get_credential
    from openai_realtime_models import (
        is_continuous,
        is_transcription_only,
        uses_language_context,
    )
    from provider_registry import get_provider

from .base import TranscriptionBackend


class RealtimeWsBackend(TranscriptionBackend):
    """Streaming WebSocket backend; reconnects by full re-initialization on resume."""

    name = 'realtime-ws'
    is_local = False
    reinit_on_resume = True

    # Don't rebuild a torn-down client on every keypress while an endpoint is down.
    REBUILD_COOLDOWN_SECS = 5.0

    def __init__(self, manager):
        super().__init__(manager)
        # Realtime WebSocket client
        self._realtime_client = None
        self._realtime_streaming_callback = None
        # Connection parameters used for reconnect-on-demand.
        # (Stored in-memory only; do not log API keys.)
        self._realtime_connect_params = None
        # Why the last recovery attempt failed, for an honest user-facing message:
        # 'connecting' | 'cooldown' | 'failed' | None
        self._last_connect_failure = None
        self._last_rebuild_attempt = None

    @property
    def _realtime_partial_callback(self):
        # Owned by the manager so it survives backend re-creation on resume
        return self._manager._realtime_partial_callback

    def _update_client_language(
        self,
        language: Optional[str],
        model_id: Optional[str] = None,
    ) -> None:
        """Keep new-model transcription language hints and prompts in sync."""
        if not self._realtime_client:
            return

        active_model = model_id or getattr(self._realtime_client, 'model', None)
        if uses_language_context(active_model):
            self._realtime_client.update_transcription_config(
                language,
                self.resolve_whisper_prompt(language)[0],
            )
        else:
            self._realtime_client.update_language(language)

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
            language = self.config.get_setting('language', None)
            instructions_parts = []
            whisper_prompt, _ = self.resolve_whisper_prompt(language)
            if whisper_prompt:
                instructions_parts.append(whisper_prompt)

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
            if (
                provider_id == 'openai'
                and is_transcription_only(model_id)
                and realtime_mode != 'transcribe'
            ):
                print(
                    f'ERROR: {model_id} is supported only with realtime_mode="transcribe"',
                    flush=True,
                )
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
            language = self.config.get_setting('language', None)
            instructions_parts = []
            whisper_prompt, _ = self.resolve_whisper_prompt(language)
            if whisper_prompt:
                instructions_parts.append(whisper_prompt)

            if language:
                instructions_parts.append(f"Transcribe in {language} language.")

            instructions = ' '.join(instructions_parts) if instructions_parts else None

            # Set language and any model-specific transcription context.
            self._update_client_language(language, model_id=model_id)

            delay = self.config.get_setting('realtime_transcription_delay', 'low')
            self._realtime_client.set_transcription_delay(delay)
            if hasattr(self._realtime_client, 'set_conversation_history'):
                history = self.config.get_setting('realtime_conversation_history', 'session')
                self._realtime_client.set_conversation_history(history)
            if self._is_partial_preview_enabled(provider_id, model_id, realtime_mode):
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
                    self.update_language(language_override)
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
        
        if backend != 'realtime-ws':
            return None

        # Recover here — before we start capturing audio — so the first chunks
        # aren't dropped, whether the socket went idle or the client was torn
        # down entirely by an earlier failure.
        if not self._ensure_client():
            return None

        # Clear server buffer before starting new recording
        self._realtime_client.clear_audio_buffer()
        self._clear_realtime_partial_preview()
        return self._realtime_streaming_callback

    def apply_partial_callback(self, callback: Optional[Callable[[str], None]]) -> None:
        """Apply the partial-preview callback to the active realtime provider."""
        if not self._realtime_client:
            return

        provider_id = self.config.get_setting('websocket_provider')
        model_id = self.config.get_setting('websocket_model')
        realtime_mode = self.config.get_setting('realtime_mode', 'transcribe')
        enabled = self._is_partial_preview_enabled(
            provider_id,
            model_id,
            realtime_mode,
        )

        if hasattr(self._realtime_client, 'set_partial_transcript_callback'):
            self._realtime_client.set_partial_transcript_callback(
                callback if enabled else None
            )
        if not enabled:
            self._clear_realtime_partial_preview()

    def _is_partial_preview_enabled(
        self,
        provider_id: str,
        model_id: str,
        realtime_mode: str,
    ) -> bool:
        if (
            not self.config.get_setting('mic_osd_enabled', True)
            or realtime_mode != 'transcribe'
            or self._realtime_partial_callback is None
        ):
            return False

        # Pill: any provider with partial-transcript support qualifies.
        if self.config.get_setting('mic_osd_style', 'waveform') == 'pill':
            return bool(
                self._realtime_client is not None
                and hasattr(self._realtime_client, 'set_partial_transcript_callback')
                and self.config.get_setting('mic_osd_pill_transcript_enabled', False)
            )

        # Waveform: only continuously streaming OpenAI models emit live deltas.
        if provider_id == 'openai':
            return is_continuous(model_id)

        return False

    def _clear_realtime_partial_preview(self) -> None:
        if not self._realtime_partial_callback:
            return
        try:
            self._realtime_partial_callback("")
        except Exception as e:
            print(f'[REALTIME] Failed to clear partial transcript preview: {e}', flush=True)

    @property
    def last_connect_failure(self) -> Optional[str]:
        """Why the last recovery attempt failed: 'connecting', 'cooldown', 'failed', or None."""
        return self._last_connect_failure

    def _ensure_client(self) -> bool:
        """Make the realtime client usable for a new recording.

        Single recovery entry point for the three states a client can be in:
        connected, disconnected (idle close), or gone entirely — the last one
        happens whenever close_realtime_connection() runs on a recording failure
        or suspend, and nothing outside the resume path rebuilds it.
        """
        if self._realtime_client:
            if self._realtime_client.connected:
                self._last_connect_failure = None
                return True
            return self._reconnect_realtime_client()

        now = time.monotonic()
        last = self._last_rebuild_attempt
        if last is not None and (now - last) < self.REBUILD_COOLDOWN_SECS:
            print('[REALTIME] Rebuild failed recently; waiting before retry', flush=True)
            self._last_connect_failure = 'cooldown'
            return False

        # initialize() can block on the connect timeout, so don't retry it on
        # every keypress while the endpoint is down.
        self._last_rebuild_attempt = now
        print('[REALTIME] Rebuilding client after teardown', flush=True)
        try:
            rebuilt = self.initialize()
        except Exception as e:
            print(f'[REALTIME] Rebuild failed: {e}', flush=True)
            self._last_connect_failure = 'failed'
            return False

        if not rebuilt or not self._realtime_client:
            self._last_connect_failure = 'failed'
            return False

        # Only failures should hold the cooldown, or a teardown shortly after a
        # good rebuild would be stalled by the previous success.
        self._last_rebuild_attempt = None
        self._last_connect_failure = None
        return True

    def _reconnect_realtime_client(self) -> bool:
        """Reconnect realtime client using stored connect params."""
        if not self._realtime_client:
            return False

        # A handshake may already be in flight (startup init or auto-reconnect).
        # Never destroy it — wait briefly for it to land instead.
        if getattr(self._realtime_client, 'connecting', False):
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if self._realtime_client.connected:
                    break
                if not getattr(self._realtime_client, 'connecting', False):
                    break
                time.sleep(0.1)
            if self._realtime_client.connected:
                print('[REALTIME] In-flight connection landed; proceeding', flush=True)
                self._last_connect_failure = None
                return True
            if getattr(self._realtime_client, 'connecting', False):
                print('[REALTIME] Still connecting; try again in a moment', flush=True)
                self._last_connect_failure = 'connecting'
                return False
            # Attempt finished without connecting; fall through to reconnect.

        params = self._realtime_connect_params or {}
        websocket_url = params.get('websocket_url')
        api_key = params.get('api_key')
        model_id = params.get('model_id')
        instructions = params.get('instructions')

        if not (websocket_url and api_key and model_id):
            print('[REALTIME] Missing connection parameters; cannot reconnect', flush=True)
            self._last_connect_failure = 'failed'
            return False

        try:
            # Best-effort: drop stale socket/thread state first. Use reset() where
            # available — close() latches the client shut and would make every
            # reconnect from here fail instantly (issue #229). ElevenLabs has no
            # reset(); its close() is already a transient teardown.
            try:
                teardown = getattr(self._realtime_client, 'reset', None)
                if teardown is None:
                    teardown = self._realtime_client.close
                teardown()
            except Exception:
                pass

            if not self._realtime_client.connect(websocket_url, api_key, model_id, instructions):
                print('[REALTIME] Reconnect failed', flush=True)
                self._last_connect_failure = 'failed'
                return False

            print('[REALTIME] Reconnected on-demand', flush=True)
            self._last_connect_failure = None
            return True
        except Exception as e:
            print(f'[REALTIME] Reconnect failed: {e}', flush=True)
            self._last_connect_failure = 'failed'
            return False

    def discard_audio(self) -> None:
        """Drop buffered audio client- and server-side; keep the connection alive."""
        if self._realtime_client:
            try:
                self._realtime_client.clear_audio_buffer()
                self._clear_realtime_partial_preview()
            except Exception as e:
                print(f'[REALTIME] Failed to discard audio: {e}', flush=True)

    def close(self) -> None:
        """Cleanup Realtime WebSocket client"""
        if self._realtime_client:
            try:
                self._realtime_client.close()
                self._realtime_client = None
                self._realtime_streaming_callback = None
                self._clear_realtime_partial_preview()
            except Exception as e:
                print(f"[WARN] Failed to cleanup realtime client: {e}")

    def update_language(self, language: Optional[str]) -> None:
        """Apply a language override to a connected client (no-op otherwise)."""
        self._update_client_language(language)

    def reinitialize(self) -> bool:
        """Re-establish the connection after suspend/resume (full re-init)."""
        return self._manager.initialize()

    def cleanup(self) -> None:
        self.close()

    @property
    def is_loaded(self) -> bool:
        return self._realtime_client is not None
