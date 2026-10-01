"""
Realtime WebSocket transcription backend.

Streams audio to a provider WebSocket during capture via the streaming
callback; transcribe() then commits the buffered audio and waits for the
final transcript. The wire protocol comes from realtime_protocols.
"""

import time
from typing import Callable, Optional

try:
    from ..service_log import log
except ImportError:
    from service_log import log

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')

try:
    from ..backend_utils import is_valid_websocket_url, normalize_backend
    from ..credential_manager import get_credential
    from ..openai_realtime_models import uses_language_context
    from ..provider_registry import get_provider, get_realtime_capabilities
    from ..realtime_protocols import live_text_mode, live_typing_conflict, load_client_class, resolve_protocol
except ImportError:
    from backend_utils import is_valid_websocket_url, normalize_backend
    from credential_manager import get_credential
    from openai_realtime_models import uses_language_context
    from provider_registry import get_provider, get_realtime_capabilities
    from realtime_protocols import live_text_mode, live_typing_conflict, load_client_class, resolve_protocol

from .base import TranscriptionBackend


class RealtimeWsBackend(TranscriptionBackend):
    """Streaming WebSocket backend; reconnects by full re-initialization on resume."""

    name = 'realtime-ws'
    is_local = False
    reinit_on_resume = True
    streams_audio = True

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
        provider_id = self.config.get_setting('websocket_provider')
        model_id = self.config.get_setting('websocket_model')

        if not provider_id:
            log('ERROR: Realtime WebSocket backend selected but websocket_provider not configured')
            return False

        if not model_id:
            log('ERROR: Realtime WebSocket backend selected but websocket_model not configured')
            return False

        try:
            protocol = resolve_protocol(provider_id, self.config.get_setting('websocket_protocol'))
        except ValueError as e:
            log(f'ERROR: {e}')
            return False

        # Custom endpoints may be self-hosted and keyless; known providers need a key
        api_key = get_credential(provider_id)
        if not api_key and provider_id != 'custom':
            log(f'ERROR: Provider {provider_id} configured but API key not found in credential store')
            return False

        realtime_mode = self.config.get_setting('realtime_mode', 'transcribe')
        if realtime_mode not in protocol.modes:
            log(f'[REALTIME] {protocol.id} supports realtime_mode {"/".join(protocol.modes)}; '
                f'using transcribe instead of {realtime_mode!r}')
            realtime_mode = 'transcribe'
        if (
            realtime_mode != 'transcribe'
            and get_realtime_capabilities(provider_id, model_id).get('transcription_only')
        ):
            log(f'ERROR: {model_id} is supported only with realtime_mode="transcribe"')
            return False

        if (
            protocol.per_recording_session
            and self.config.get_setting('recording_mode', 'toggle') in ('continuous', 'long_form')
        ):
            log(f'ERROR: {protocol.id} streams one recording at a time; use toggle, push_to_talk or auto')
            return False

        client = load_client_class(protocol)(mode=realtime_mode)
        client.configure(self._client_setting_reader(provider_id))

        websocket_url = self.config.get_setting('websocket_url')
        if websocket_url and not is_valid_websocket_url(websocket_url):
            log(f'ERROR: websocket_url must be a ws:// or wss:// URL with a host: {websocket_url!r}')
            return False
        if not websocket_url:
            if provider_id == 'custom':
                log('ERROR: Custom realtime backend requires websocket_url to be configured')
                return False
            if protocol.derives_url:
                # Derived transcribe URLs carry ?intent=transcription, which a realtime session contradicts
                if (
                    realtime_mode == 'transcribe'
                    and getattr(client, 'transcription_session_type', None) == 'realtime'
                ):
                    log('ERROR: realtime_transcription_session_type "realtime" requires websocket_url')
                    return False
                try:
                    websocket_url = self._get_websocket_url(provider_id, model_id, realtime_mode)
                except Exception as e:
                    log(f'ERROR: Failed to derive WebSocket URL: {e}')
                    return False
            else:
                websocket_url = (get_provider(provider_id) or {}).get('websocket_endpoint')
                if not websocket_url:
                    log(f'ERROR: Provider {provider_id} has no websocket_endpoint')
                    return False

        language = self.config.get_setting('language', None)
        instructions = self._build_instructions(language) if protocol.uses_instructions else None

        self._realtime_client = client
        try:
            self._update_client_language(language, model_id=model_id)
        except ValueError as e:
            log(f'ERROR: {e}')
            self._realtime_client = None
            return False
        client.set_max_buffer_seconds(self.config.get_setting('realtime_buffer_max_seconds', 5))
        self.apply_partial_callback(self._realtime_partial_callback)
        self.apply_live_listener(getattr(self._manager, '_realtime_live_listener', None))
        if self.config.get_setting('realtime_live_typing', False):
            conflict = live_typing_conflict(self.config.get_setting)
            if conflict:
                log(f'[REALTIME] Live typing off: {conflict}; text pastes at stop')

        self._realtime_connect_params = {
            'websocket_url': websocket_url,
            'api_key': api_key,
            'model_id': model_id,
            'instructions': instructions,
        }
        if not client.connect(websocket_url, api_key, model_id, instructions):
            log(f'ERROR: Failed to connect to realtime WebSocket ({protocol.id})')
            try:
                client.close()
            except Exception:
                pass
            self._realtime_client = None
            return False

        def _send_direct(audio_chunk: np.ndarray):
            """Queue audio on the client; it resamples and sends off-thread."""
            try:
                client.append_audio(audio_chunk)
            except Exception as e:
                log(f'{client.LOG_TAG} Streaming error: {e}')

        _send_direct.set_input_sample_rate = client.set_input_sample_rate
        self._realtime_streaming_callback = _send_direct

        log(f'[BACKEND] Using Realtime WebSocket: {websocket_url}')
        log(f'[REALTIME] Model: {model_id}, Provider: {provider_id}, Protocol: {protocol.id}')

        # Explicitly set to None to avoid confusion with top-level model setting
        self.current_model = None
        self.ready = True
        return True

    # Server-shape options that only make sense for a custom endpoint
    CUSTOM_ONLY_SETTINGS = ('websocket_sample_rate', 'websocket_session_format')

    def _client_setting_reader(self, provider_id: str):
        """get_setting for the client; built-in providers never see custom-only options."""
        def get_setting(key, default=None):
            if provider_id != 'custom' and key in self.CUSTOM_ONLY_SETTINGS:
                return default
            return self.config.get_setting(key, default)
        return get_setting

    def _build_instructions(self, language: Optional[str]) -> Optional[str]:
        """Session instructions from the whisper prompt and language."""
        parts = []
        whisper_prompt, _ = self.resolve_whisper_prompt(language)
        if whisper_prompt:
            parts.append(whisper_prompt)
        if language:
            parts.append(f"Transcribe in {language} language.")
        return ' '.join(parts) if parts else None

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
            log('[REALTIME] Client not initialized')
            return ""
        
        if not (self._realtime_client.connected or self._realtime_client.commit_after_disconnect):
            log('[REALTIME] Client not connected')
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
                    log(f'[REALTIME] Provider does not support mid-session language override '
                        f'(requested: {language_override}); change will take effect on next session')
            
            # Get timeout from config
            timeout = self.config.get_setting('realtime_timeout', 30)
            
            # Commit and get text (audio was already streamed via callback)
            transcription = self._realtime_client.commit_and_get_text(timeout=timeout)
            
            return transcription.strip()
            
        except Exception as e:
            log(f'[REALTIME] Transcription failed: {e}')
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
        try:
            self._realtime_client.clear_audio_buffer()
        except RuntimeError as e:
            # A one-stream-per-recording socket can close between the readiness
            # check and here. Capture hasn't started, so one fresh attempt is safe.
            log(f'[REALTIME] {e}; reconnecting before capture')
            if not self._reconnect_realtime_client():
                return None
            try:
                self._realtime_client.clear_audio_buffer()
            except RuntimeError:
                return None
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

    def apply_live_listener(self, listener) -> None:
        """Route structured live text (committed, tail) to `listener`."""
        if self._realtime_client:
            self._realtime_client.set_live_text_listener(listener)

    @property
    def live_typing_supported(self) -> bool:
        """Config allows typing live text early (append-only model, whole-dictation consumers off)."""
        return live_typing_conflict(self.config.get_setting) is None

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

        # Waveform: only models whose text streams mid-utterance.
        return live_text_mode(
            provider_id, model_id, self.config.get_setting('websocket_live_text'),
            self.config.get_setting('websocket_protocol'),
        ) != 'none'

    def _clear_realtime_partial_preview(self) -> None:
        if not self._realtime_partial_callback:
            return
        try:
            self._realtime_partial_callback("")
        except Exception as e:
            log(f'[REALTIME] Failed to clear partial transcript preview: {e}')

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
            if self._realtime_client.busy:
                log('[REALTIME] Previous recording is still awaiting its transcript')
                self._last_connect_failure = 'processing'
                return False
            if self._realtime_client.connected and not self._realtime_client.needs_fresh_session:
                self._last_connect_failure = None
                return True
            return self._reconnect_realtime_client()

        now = time.monotonic()
        last = self._last_rebuild_attempt
        if last is not None and (now - last) < self.REBUILD_COOLDOWN_SECS:
            log('[REALTIME] Rebuild failed recently; waiting before retry')
            self._last_connect_failure = 'cooldown'
            return False

        # initialize() can block on the connect timeout, so don't retry it on
        # every keypress while the endpoint is down.
        self._last_rebuild_attempt = now
        log('[REALTIME] Rebuilding client after teardown')
        try:
            rebuilt = self.initialize()
        except Exception as e:
            log(f'[REALTIME] Rebuild failed: {e}')
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
                log('[REALTIME] In-flight connection landed; proceeding')
                self._last_connect_failure = None
                return True
            if getattr(self._realtime_client, 'connecting', False):
                log('[REALTIME] Still connecting; try again in a moment')
                self._last_connect_failure = 'connecting'
                return False
            # Attempt finished without connecting; fall through to reconnect.

        params = self._realtime_connect_params or {}
        websocket_url = params.get('websocket_url')
        api_key = params.get('api_key')
        model_id = params.get('model_id')
        instructions = params.get('instructions')

        # api_key may be None: custom endpoints can be keyless
        if not (websocket_url and model_id):
            log('[REALTIME] Missing connection parameters; cannot reconnect')
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
                log('[REALTIME] Reconnect failed')
                self._last_connect_failure = 'failed'
                return False

            log('[REALTIME] Reconnected on-demand')
            self._last_connect_failure = None
            return True
        except Exception as e:
            log(f'[REALTIME] Reconnect failed: {e}')
            self._last_connect_failure = 'failed'
            return False

    def discard_audio(self) -> None:
        """Drop buffered audio client- and server-side; keep the connection alive."""
        if self._realtime_client:
            try:
                self._realtime_client.discard_audio()
                self._clear_realtime_partial_preview()
            except Exception as e:
                log(f'[REALTIME] Failed to discard audio: {e}')

    def close(self) -> None:
        """Cleanup Realtime WebSocket client"""
        if self._realtime_client:
            try:
                self._realtime_client.close()
                self._realtime_client = None
                self._realtime_streaming_callback = None
                self._clear_realtime_partial_preview()
            except Exception as e:
                log(f"[WARN] Failed to cleanup realtime client: {e}")

    def update_language(self, language: Optional[str]) -> None:
        """Apply a language override to a connected client (no-op otherwise)."""
        try:
            self._update_client_language(language)
        except ValueError as e:
            log(f'[REALTIME] Language override ignored: {e}')

    def reinitialize(self) -> bool:
        """Re-establish the connection after suspend/resume (full re-init)."""
        return self._manager.initialize()

    def cleanup(self) -> None:
        self.close()

    @property
    def is_busy(self) -> bool:
        """A one-stream-per-recording client is still finishing the last recording."""
        return bool(self._realtime_client and self._realtime_client.busy)

    @property
    def is_loaded(self) -> bool:
        return self._realtime_client is not None
