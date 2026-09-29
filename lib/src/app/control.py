"""Recording-control commands, file transcription and model lifecycle."""

import threading
import time

from audio_file import AudioFileError, decode_audio_file
from paths import MODEL_UNLOADED_FILE
from service_log import log
from backend_utils import normalize_backend
from text_injector import preprocess_text

# Announce readiness when the load took long enough to be noticed, or when
# someone was told to wait for it.
READY_NOTIFY_AFTER_S = 5


class ControlMixin:
    """Recording-control commands, file transcription and model lifecycle."""

    def _handle_file_transcribe(self, path, language=None, clean=False):
        """Transcribe a file through the daemon's already-loaded batch backend."""
        backend = normalize_backend(
            self.config.get_setting('transcription_backend', 'pywhispercpp')
        )
        if backend == 'realtime-ws':
            return False, (
                'The realtime-ws backend supports live capture only; configure '
                'a local or REST backend to transcribe files'
            )

        with self._recording_lock:
            if self.is_recording:
                return False, 'Cannot transcribe a file while recording'
            if self.is_processing:
                return False, 'The daemon is already processing audio'
            if self._file_transcription_active:
                # One handler thread per connection, so concurrent requests
                # would otherwise overlap and clear each other's claim
                return False, 'Another file transcription is already running'
            if self._longform_active:
                return False, 'A long-form session is active; submit or cancel it first'
            if self._model_operation_active:
                return False, 'The transcription backend is loading; retry shortly'
            if self._recording_finalizing.is_set():
                return False, 'The daemon is finalizing a recording'
            if self._model_initializing:
                return False, 'The transcription backend is still initializing; retry shortly'
            if MODEL_UNLOADED_FILE.exists() or not self.whisper_manager.is_ready():
                return False, 'The transcription backend is not loaded; run hyprwhspr model reload'
            self._file_transcription_active = True

        try:
            audio_data, sample_rate = decode_audio_file(path)
            text = self.whisper_manager.transcribe_audio(
                audio_data,
                sample_rate=sample_rate,
                language_override=language,
            )
            text = text.strip() if text else ''
            if not text:
                return False, 'Transcription produced no text'
            if clean:
                text = preprocess_text(text, self.config)
                if not text.strip():
                    return False, 'Transcription produced no text after cleanup'
            return True, text
        except AudioFileError as exc:
            return False, str(exc)
        except Exception as exc:
            log(f"[TRANSCRIBE] File transcription failed: {exc}")
            return False, f'File transcription failed: {exc}'
        finally:
            with self._recording_lock:
                self._file_transcription_active = False

    def _handle_control_command(self, action, language=None):
        """Apply recording policy for a command received by the control server."""
        recording_mode = self.config.get_setting("recording_mode", "toggle")
        if action == "start":
            lang_info = f" (language: {language})" if language else ""
            if recording_mode == "long_form":
                self._longform.request_start(language_override=language)
            elif not self.is_recording:
                log(f"[CONTROL] Recording start requested (immediate){lang_info}")
                self._start_recording(language_override=language)
                if recording_mode == "continuous":
                    self._continuous_start_silence_monitor()
                elif recording_mode in ("toggle", "auto"):
                    self._autostop_start_silence_monitor()
            else:
                if recording_mode == "push_to_talk" and self._ptt_locked:
                    log("[CONTROL] Locked push-to-talk session ended by start request")
                    self._stop_recording()
                else:
                    if recording_mode == "push_to_talk":
                        self._ptt_mark_press()
                    log("[CONTROL] Recording already in progress, ignoring start request")
        elif action in ("stop", "release"):
            # "release" is a key-up from an external binding and may latch a
            # long push-to-talk hold; an explicit "stop" always stops. Outside
            # push-to-talk the two are identical.
            if recording_mode == "long_form":
                self._longform.request_pause()
            elif self.is_recording:
                if (action == "release" and recording_mode == "push_to_talk"
                        and self._ptt_release_latches()):
                    return
                log(f"[CONTROL] Recording {action} requested (immediate)")
                if recording_mode == "continuous":
                    self._continuous_stop_and_wait()
                self._stop_recording()
            else:
                log(f"[CONTROL] Not currently recording, ignoring {action} request")
        elif action == "cancel":
            if recording_mode == "long_form":
                self._longform.request_cancel()
            elif self.is_recording:
                log("[CONTROL] Recording cancel requested (immediate)")
                if recording_mode == "continuous":
                    self._continuous_cancelled = True
                    self._continuous_stop_silence_monitor()
                self._cancel_recording()
            else:
                log("[CONTROL] Not currently recording, ignoring cancel request")
        elif action == "submit":
            if recording_mode == "long_form":
                log("[CONTROL] Long-form submit requested (immediate)")
                self._longform.submit_shortcut()
            else:
                log("[CONTROL] Submit command only valid in long_form mode")
        elif action == "model_unload":
            self._handle_model_operation("unload")
        elif action == "model_reload":
            self._handle_model_operation("reload")
        else:
            log(f"[CONTROL] Unknown recording control action: {action}")

    def _handle_model_operation(self, operation):
        """Load or unload the model without holding the recording lock.

        A reload can take minutes, so the lock is only used to claim the
        exclusive flag; recording and file requests reject on that flag while
        the slow work runs instead of blocking on the lock.
        """
        with self._recording_lock:
            busy = (self.is_recording or self.is_processing
                    or self._recording_finalizing.is_set()
                    or self._file_transcription_active
                    or self._model_operation_active
                    or self._longform_active)
            if not busy:
                self._model_operation_active = True
        if busy:
            log(f"[CONTROL] Cannot {operation} model while the backend is in use")
            self._notify_user(
                "hyprwhspr", "Finish the current recording or transcription first", urgency="normal"
            )
            return

        log(f"[CONTROL] Model {operation} requested")
        try:
            if operation == "unload":
                succeeded = self.whisper_manager.unload_model()
            else:
                succeeded = self.whisper_manager.reload_model()
        finally:
            with self._recording_lock:
                self._model_operation_active = False

        if operation == "unload":
            if succeeded:
                try:
                    MODEL_UNLOADED_FILE.touch()
                except Exception:
                    pass
                self._notify_user("hyprwhspr", "Model unloaded — GPU resources freed", urgency="low")
            else:
                self._notify_user("hyprwhspr", "Unload not applicable for this backend", urgency="normal")
        else:
            if succeeded:
                try:
                    MODEL_UNLOADED_FILE.unlink(missing_ok=True)
                except Exception:
                    pass
                self._notify_user("hyprwhspr", "Model reloaded — ready to record", urgency="low")
            else:
                self._notify_user("hyprwhspr", "Model reload failed — check logs", urgency="critical")

    def _start_backend_init_background(self):
        """Initialize the transcription backend in a background thread.

        Used for slow backends at startup and to retry after a failed init.
        Guarded so concurrent callers can't spawn duplicate init threads.
        """
        with self._backend_init_lock:
            if self._model_initializing:
                return
            self._model_initializing = True
            self._backend_init_failed = False

        def _bg_init():
            started = time.monotonic()
            ok = self.whisper_manager.initialize()
            self._backend_init_failed = not ok
            self._model_initializing = False
            if ok:
                log("[READY] Model ready — recording now available")
                if self._notify_when_ready or time.monotonic() - started >= READY_NOTIFY_AFTER_S:
                    self._notify_when_ready = False
                    self._notify_user("hyprwhspr", "Ready", urgency="low")
            else:
                log("[ERROR] Failed to initialize backend in background")

        threading.Thread(target=_bg_init, daemon=True, name="BackendInit").start()
