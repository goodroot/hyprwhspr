"""The recording lifecycle: start, stop, cancel, processing and injection."""

import json
import threading
import time

from paths import DEBUG_RECORDINGS_DIR
from text_injector import InjectionOutcome
from processing_trace import build_processing_trace
from hallucination import is_hallucination
from service_log import log
from backend_utils import normalize_backend

try:
    import numpy as np
except ImportError:
    np = None  # Will be checked when needed


class RecordingMixin:
    """The recording lifecycle: start, stop, cancel, processing and injection."""

    def _set_longform_processing(self, processing):
        with self._recording_lock:
            self.is_processing = processing

    def _claim_longform_recording(self):
        """Reserve the backend for a long-form session; False if busy or not loaded."""
        with self._recording_lock:
            busy = self._file_transcription_active or self._model_operation_active
            loading = self._model_initializing or self._backend_init_failed
            if not (busy or loading):
                self._longform_active = True
                return True
        if loading:
            # As for a refused normal start: announce Ready later, retry a failed init
            self._notify_when_ready = True
            if self._backend_init_failed:
                self._start_backend_init_background()
        return False

    def _release_longform_recording(self):
        with self._recording_lock:
            self._longform_active = False

    def _suppress_recording_playback(self, session):
        """Serialize external audio operations without holding recording state."""
        with self._playback_lock:
            with self._recording_lock:
                active = (self.is_recording and self._recording_session is session
                          and not self._playback_shutdown)
            if not active:
                return False
            if self.config.get_setting('audio_ducking', False):
                self._playback_session = session
                try:
                    self.playback_suppressor.suppress(
                        mode=self.config.get_setting('audio_ducking_mode', 'duck'),
                        reduction_percent=self.config.get_setting('audio_ducking_percent', 50))
                finally:
                    with self._recording_lock:
                        active = (self.is_recording and self._recording_session is session
                                  and not self._playback_shutdown)
                    if not active:
                        self.playback_suppressor.restore()
                        self._playback_session = None
            return active

    def _restore_recording_playback(self, session=None, shutdown=False):
        with self._playback_lock:
            if shutdown:
                self._playback_shutdown = True
            owned = session is not None and self._playback_session is session
            if shutdown or owned:
                self.playback_suppressor.restore()
                self._playback_session = None

    def _wait_for_start_settled(self):
        """Let an in-flight start finish before a stop touches capture.

        Otherwise the stop can run before capture opens (orphaning it), or race
        the aborted start's teardown and lose the recorded audio.
        """
        if self._start_owner != threading.get_ident():
            if not self._start_settled.wait(timeout=5.0):
                log("[WARN] Recording start still settling; stopping capture anyway")

    def _start_recording(self, language_override=None):
        """Start voice recording
        
        Args:
            language_override: Optional language code to use for this recording session
                              (overrides the default language from config)
        """
        # Gate checks happen under the same lock that sets the flag, so a
        # concurrent toggle never sees is_recording=True for a start that
        # gets blocked (and stops a recording that never began)
        with self._recording_lock:
            if self.is_recording:
                return
            if self._recording_starting:
                # A stopped start is still releasing capture; refuse until it settles.
                blocked = 'starting'
            elif self._model_initializing:
                blocked = 'initializing'
            elif self._backend_init_failed:
                blocked = 'init-failed'
            elif self._file_transcription_active:
                blocked = 'file-transcription'
            elif self._model_operation_active:
                blocked = 'model-operation'
            elif getattr(self.whisper_manager, '_model_manually_unloaded', False):
                blocked = 'unloaded'
            elif self.whisper_manager.realtime_client_missing():
                blocked = 'realtime-reconnect'
            else:
                blocked = None
                # Set flag immediately to prevent duplicate starts
                self.is_recording = True
                self._recording_starting = True
                self._start_settled.clear()
                self._start_owner = threading.get_ident()
                session = object()
                self._recording_session = session
                # Store language override for this recording session
                self._current_language_override = language_override
                # Push-to-talk hold is measured from this accepted press
                self._ptt_press_time = time.monotonic()
                self._ptt_locked = False

        # A capture client self-triggered this start over the FIFO and is now
        # blocking on completion. Release it here or it waits forever and keeps
        # the capture slot occupied (no recording will ever finish for it).
        if blocked is not None:
            self._release_blocked_capture()

        # Model is still loading in background
        if blocked in ('initializing', 'init-failed', 'realtime-reconnect'):
            self._notify_when_ready = True

        if blocked == 'starting':
            self._notify_user("hyprwhspr", "Still stopping — try again", urgency="normal")
            log("[CONTROL] Recording blocked: previous start still settling")
            return

        if blocked == 'initializing':
            self._notify_user("hyprwhspr", "Model still loading, please wait…", urgency="normal")
            log("[CONTROL] Recording blocked: model is still initializing")
            return

        # Backend init failed (e.g. deps missing at boot): retry it instead
        # of recording audio that can never be transcribed
        if blocked == 'init-failed':
            self._notify_user("hyprwhspr", "Backend failed to load — retrying, try again shortly", urgency="normal")
            log("[CONTROL] Recording blocked: backend init failed - retrying")
            self._start_backend_init_background()
            return

        if blocked == 'file-transcription':
            self._notify_user("hyprwhspr", "File transcription in progress", urgency="normal")
            log("[CONTROL] Recording blocked: file transcription in progress")
            return

        if blocked == 'model-operation':
            self._notify_user("hyprwhspr", "Model load in progress — try again shortly", urgency="normal")
            log("[CONTROL] Recording blocked: model load/unload in progress")
            return

        # Realtime client was torn down; rebuild in background instead of blocking
        if blocked == 'realtime-reconnect':
            self._notify_user("hyprwhspr", "Reconnecting — try again in a moment", urgency="normal")
            log("[CONTROL] Recording blocked: realtime client not connected - reconnecting")
            self._start_backend_init_background()
            return

        # Model was deliberately unloaded to free GPU resources
        if blocked == 'unloaded':
            self._notify_user(
                "hyprwhspr",
                "Model unloaded — run: hyprwhspr model reload",
                urgency="normal",
            )
            log("[CONTROL] Recording blocked: model is unloaded. Run: hyprwhspr model reload")
            return

        try:
            self._clear_mic_osd_preview_text()

            # Clear zero-volume signal file when starting a new recording
            # This allows waybar to recover immediately on successful start
            self._clear_zero_volume_signal()
            
            # Write recording status to file for tray script
            self._write_recording_status(True)

            # Update language in realtime client if override is provided
            if language_override is not None:
                self.whisper_manager.update_realtime_language(language_override)
            
            # Check if using realtime-ws backend and get streaming callback
            streaming_callback = self.whisper_manager.get_realtime_streaming_callback()
            backend = normalize_backend(self.config.get_setting('transcription_backend', 'pywhispercpp'))
            if backend == 'realtime-ws' and streaming_callback is None:
                # Fail fast: realtime-ws requires an active streaming callback (and a connected client)
                with self._recording_lock:
                    self.is_recording = False
                self._write_recording_status(False)
                self._release_blocked_capture()
                self._hide_mic_osd()
                self._stop_audio_level_monitoring()
                self._notify_zero_volume(
                    self._realtime_unavailable_message(),
                    log_level="ERROR",
                )
                # Restore audio if it was ducked or paused
                self._restore_recording_playback(session)
                return
            
            # Helper function to verify stream is working and play sound
            def verify_and_play_sound():
                """Wait for callbacks and play sound if stream works"""
                import time
                start_time = time.monotonic()
                while time.monotonic() - start_time < 1.5:  # Wait up to 1.5s
                    with self._recording_lock:
                        if not self.is_recording or self._playback_shutdown:
                            return None
                    # Read frames_since_start with lock held to avoid data race
                    with self.audio_capture.lock:
                        frames_count = self.audio_capture.frames_since_start
                    if frames_count > 0:
                        # At least one callback received - stream is working
                        self.audio_manager.play_start_sound()
                        return True
                    time.sleep(0.05)
                # No callbacks received - stream likely broken (will be handled by caller)
                return False
            
            # Helper function to verify stream continues working after initial check
            def verify_stream_stable():
                """Verify stream continues receiving callbacks after initial verification"""
                import time
                initial_frames = 0
                with self.audio_capture.lock:
                    initial_frames = self.audio_capture.frames_since_start
                
                # Wait a bit more to ensure stream is stable
                time.sleep(0.2)
                
                with self.audio_capture.lock:
                    current_frames = self.audio_capture.frames_since_start
                    # Stream should have received more callbacks if it's stable
                    return current_frames > initial_frames
            
            def abandon_start():
                # Stop/cancel wait for this start to settle and then own capture
                # teardown (and the audio). Shutdown may already have stopped
                # capture before it opened here, so release it ourselves.
                if self._playback_shutdown:
                    self.audio_capture.stop_recording()

            # Start audio capture (with streaming callback for realtime-ws)
            try:
                if not self.audio_capture.start_recording(streaming_callback=streaming_callback):
                    raise RuntimeError("start_recording() returned False")
                
                # Verify stream is working before playing sound
                verified = verify_and_play_sound()
                if verified is None:
                    abandon_start()
                    return
                if not verified:
                    # Stream broken - stop recording (thread will clean up stream)
                    self.audio_capture.stop_recording()

                    # Reset state
                    with self._recording_lock:
                        self.is_recording = False
                    self._write_recording_status(False)
                    self._release_blocked_capture()
                    
                    # Hide mic-osd visualization
                    self._hide_mic_osd()

                    message = self._mic_failure_message(
                        "Microphone not responding - please unplug and replug USB microphone, then try recording again")
                    self._notify_zero_volume(message, log_level="ERROR")

                    # Restore audio if it was ducked or paused
                    self._restore_recording_playback(session)
                    return  # Don't attempt recovery during user-initiated recording

                if not self._suppress_recording_playback(session):
                    abandon_start()
                    return

                # Stream is verified working - show mic-osd visualization
                log("Recording started")
                self._show_mic_osd()
                
                # Additional stability check - verify stream continues working
                stable = verify_stream_stable()
                with self._recording_lock:
                    stopped = not self.is_recording or self._playback_shutdown
                if stopped:
                    # Stop freezes the frame count; that is not an unstable stream.
                    abandon_start()
                    return
                if not stable:
                    # Stream stopped working shortly after starting
                    self.audio_capture.stop_recording()
                    with self._recording_lock:
                        self.is_recording = False
                    self._write_recording_status(False)
                    self._release_blocked_capture()
                    
                    # Hide mic-osd visualization
                    self._hide_mic_osd()

                    fallback = "Microphone stream unstable - please wait a moment and try recording again"
                    message = self._mic_failure_message(fallback)
                    self._notify_zero_volume(message, log_level="WARN" if message == fallback else "ERROR")

                    # Restore audio if it was ducked or paused
                    self._restore_recording_playback(session)
                    return
                
                with self._recording_lock:
                    if not self.is_recording or self._playback_shutdown:
                        return

                # Recording is confirmed working - abort any in-progress recovery and clear background retries
                try:
                    self.audio_capture.abort_recovery()
                except Exception:
                    pass
                if self._background_recovery_needed.is_set():
                    log("[HEALTH] Recording succeeded - canceling background recovery")
                    self._background_recovery_needed.clear()
                
                # Stream is working and stable - start monitoring
                self._start_audio_level_monitoring()
                    
            except (RuntimeError, Exception) as e:
                log(f"[ERROR] Failed to start recording: {e}")

                # Clean up resources
                self._hide_mic_osd()
                self._stop_audio_level_monitoring()

                self.whisper_manager.close_realtime_connection("recording start failure")

                # Stop recording (will clean up if thread started)
                try:
                    self.audio_capture.stop_recording()
                except Exception:
                    pass  # Ignore if already stopped

                # Reset state - fail fast, don't attempt recovery
                with self._recording_lock:
                    self.is_recording = False
                self._write_recording_status(False)
                self._release_blocked_capture()
                self._notify_zero_volume(
                    self._mic_failure_message(
                        "Microphone disconnected or not responding - please unplug and replug USB microphone, then try recording again"),
                    log_level="ERROR")

                # Restore audio if it was ducked or paused
                self._restore_recording_playback(session)
                return

        except Exception as e:
            log(f"[ERROR] Failed to start recording: {e}")

            # Clean up resources
            self._hide_mic_osd()
            self._stop_audio_level_monitoring()

            self.whisper_manager.close_realtime_connection("recording start failure")

            with self._recording_lock:
                self.is_recording = False
            self._write_recording_status(False)
            # No recording will complete, so any capture client waiting on this
            # start has to be released here too or it holds the slot forever
            self._release_blocked_capture()

            # Restore audio if it was ducked or paused
            self._restore_recording_playback(session)

        finally:
            with self._recording_lock:
                self._recording_starting = False
            self._start_settled.set()

    def _cleanup_recording_state(self, session=None):
        """Best-effort cleanup after any recording ends. Safe to call multiple times."""
        session = session if session is not None else self._recording_session
        # Release recording state and capture clients before teardown can block.
        try:
            self._write_recording_status(False)
        except Exception:
            pass
        self._notify_capture("", final=True)
        self._restore_recording_playback(session)

        # Retire this recording's monitor before hide can block long enough for
        # a later recording to install its own monitor in the shared slot.
        self._autostop_stop_silence_monitor()
        try:
            self._hide_mic_osd()
        except Exception:
            pass

        try:
            self._clear_mic_osd_preview_text()
        except Exception:
            pass
        try:
            self._stop_audio_level_monitoring()
        except Exception:
            pass
        try:
            self._restore_recording_playback(session)
        except Exception:
            pass

    def _cancel_recording_muted(self):
        """Cancel recording early due to muted microphone"""
        with self._recording_lock:
            if not self.is_recording:
                return
            session = self._recording_session
            self.is_recording = False
            self._current_language_override = None  # Clear language override on error
            self._ptt_reset()

        log("[MUTE] Recording cancelled - microphone returned silence for 1 second")

        self._cleanup_recording_state(session)
        try:
            self._wait_for_start_settled()
            self.audio_capture.stop_recording()
            self.audio_manager.play_error_sound()
            # Note: No desktop notification - tray will detect muted state via audio level monitoring
        except Exception as e:
            log(f"[ERROR] Error canceling recording: {e}")

    def _cancel_recording(self):
        """Cancel recording and discard audio without transcribing or injecting text"""
        with self._recording_lock:
            if not self.is_recording:
                return
            session = self._recording_session
            self.is_recording = False
            self._current_language_override = None
            self._ptt_reset()

        log("Recording cancelled (discarded)")

        self._cleanup_recording_state(session)
        try:
            # Stop capture and discard the audio data
            self._wait_for_start_settled()
            self.audio_capture.stop_recording()

            # Discard buffered realtime audio but keep the WebSocket alive so the
            # next recording starts without paying a fresh handshake
            self.whisper_manager.discard_realtime_audio()

            self.audio_manager.play_error_sound()
        except Exception as e:
            log(f"[ERROR] Error cancelling recording: {e}")

    _DEBUG_RECORDINGS_KEEP = 3

    def _save_debug_recording(self, audio_data):
        """Keep the last few raw recordings in DEBUG_RECORDINGS_DIR (debug_recordings).

        Called from both the stop path and continuous-mode flushes, which can
        land in the same second and race on pruning.
        """
        if audio_data is None or not self.config.get_setting('debug_recordings', False):
            return
        try:
            DEBUG_RECORDINGS_DIR.mkdir(mode=0o700, exist_ok=True)
            millis = time.time_ns() // 1_000_000 % 1000
            path = DEBUG_RECORDINGS_DIR / f"{time.strftime('%Y%m%d-%H%M%S')}-{millis:03d}.wav"
            self.audio_capture.save_audio_to_wav(audio_data, str(path))
            for old in sorted(DEBUG_RECORDINGS_DIR.glob('*.wav'))[:-self._DEBUG_RECORDINGS_KEEP]:
                old.unlink(missing_ok=True)
        except Exception as e:
            log(f"[WARN] Failed to save debug recording: {e}")

    def _stop_recording(self):
        """Stop voice recording and process audio"""
        with self._recording_lock:
            if not self.is_recording:
                return
            self.is_recording = False
            self._recording_finalizing.set()
            session = self._recording_session
            self._ptt_reset()

        try:
            log("Recording stopped")

            # Tear down the auto-stop silence monitor if it was running (toggle/auto modes)
            self._autostop_stop_silence_monitor()

            self._clear_mic_osd_preview_text()

            # Set visualizer to processing state (keep it visible during transcription)
            self._set_visualizer_state('processing')
            
            # Stop audio level monitoring
            self._stop_audio_level_monitoring()
            
            # Write recording status to file for tray script
            self._write_recording_status(False)

            # Restore system audio if it was ducked or paused
            self._restore_recording_playback(session)

            # Check backend type
            backend = self.config.get_setting('transcription_backend', 'pywhispercpp')
            backend = normalize_backend(backend)
            
            # Stop audio capture
            self._wait_for_start_settled()
            audio_data = self.audio_capture.stop_recording()
            self._save_debug_recording(audio_data)

            # Check for zero-volume or broken stream
            if audio_data is None:
                # Stream was broken - check if we got any callbacks
                self.audio_manager.play_error_sound()
                with self.audio_capture.lock:
                    frames_count = self.audio_capture.frames_since_start
                if frames_count == 0:
                    # No callbacks received - mic disconnected during recording
                    self._notify_zero_volume("Microphone disconnected during recording - no audio captured. Try recording again after reseating.")
                else:
                    # Had callbacks but no data - stream broke mid-recording
                    self._notify_zero_volume("Audio stream broke during recording - no audio data captured. Try recording again after reseating.")
                # Show error state and hide OSD
                self._show_result_and_hide(False)
                self._notify_capture("", final=True)
            elif self._is_zero_volume(audio_data):
                # Audio data exists but is all zeros - mic not producing sound
                # Play error sound and notify user (may be intentional muting, but still inform)
                self.audio_manager.play_error_sound()
                self._notify_zero_volume("Microphone not producing audio (zero volume detected). This may be intentional muting, or the microphone may need to be reseated.")
                # Show error state and hide OSD
                self._show_result_and_hide(False)
                self._notify_capture("", final=True)
            else:
                # Valid audio data - process it
                self.audio_manager.play_stop_sound()
                self._process_audio(audio_data)
                
            # Clear language override after transcription completes
            self._current_language_override = None
                
        except Exception as e:
            log(f"[ERROR] Error stopping recording: {e}")
            self._notify_capture("", final=True)
            # Ensure cleanup even if error occurs
            try:
                self.is_recording = False
                self._current_language_override = None  # Clear language override on cancel
                self._show_result_and_hide(False)
                self._stop_audio_level_monitoring()
                self._write_recording_status(False)
                self._continuous_stop_silence_monitor()
                self._restore_recording_playback(session)

                self.whisper_manager.close_realtime_connection("recording stop error")
            except Exception:
                pass  # Best effort cleanup
        finally:
            self._recording_finalizing.clear()

    def _process_audio(self, audio_data):
        """Process captured audio through Whisper"""
        with self._recording_lock:
            if self.is_processing:
                return
            self.is_processing = True

        success = False
        try:
            # Transcribe audio with language override if set
            transcription = self.whisper_manager.transcribe_audio(
                audio_data,
                sample_rate=self.audio_capture.sample_rate,
                language_override=self._current_language_override,
            )

            if transcription and transcription.strip():
                text = transcription.strip()

                # Filter out Whisper hallucination markers - don't touch clipboard
                if is_hallucination(text, self.config.get_hallucination_markers()):
                    log(f"[INFO] Whisper hallucination detected: {text!r} - ignoring")
                    if self._recording_control_server.is_trace_capture():
                        self._notify_capture(text, final=True)
                    self.audio_manager.play_error_sound()
                    success = False
                    # Explicitly handle cleanup before returning to ensure visualizer state is updated
                    with self._recording_lock:
                        self.is_processing = False
                    self._show_result_and_hide(False)
                    return

                # Inject text
                outcome = self._inject_text(text)
                success = outcome != InjectionOutcome.FAILED
            else:
                log("[WARN] No transcription generated")
                self.audio_manager.play_error_sound()

        except Exception as e:
            log(f"[ERROR] Error processing audio: {e}")
        finally:
            self._notify_capture("", final=True)
            self._clear_mic_osd_preview_text()
            with self._recording_lock:
                self.is_processing = False
            # Show success/error state and hide OSD after delay
            self._show_result_and_hide(success)

    def _notify_capture(self, text="", final=True):
        """Complete a capture request, encoding trace requests as one JSON document."""
        if self._recording_control_server.is_trace_capture():
            text = json.dumps(
                build_processing_trace(text, self.config),
                ensure_ascii=False,
                separators=(',', ':'),
            ) + '\n'
        self._recording_control_server.notify_capture(text, final=final)

    def _release_blocked_capture(self):
        """Finish a capture request whose recording was refused before it began."""
        if self._recording_control_server.has_capture_subscriber():
            self._notify_capture("", final=True)

    def _inject_text(self, text):
        """Inject transcribed text into active application"""

        # Capture mode: route text to client instead of injecting into active app
        if self._recording_control_server.has_capture_subscriber():
            self._notify_capture(text, final=True)
            return InjectionOutcome.INJECTED

        try:
            outcome = self.text_injector.inject_text(text)
            if outcome == InjectionOutcome.FAILED:
                log(f"[ERROR] Text injection failed ({len(text)} chars)")
                notify = True
                if self.config.get_setting('recording_mode', 'toggle') == 'continuous':
                    with self._recording_lock:
                        notify = not self._continuous_delivery_failure_notified
                        self._continuous_delivery_failure_notified = True
                if notify:
                    self._notify_user(
                        "hyprwhspr", "Text delivery failed. Recover with hyprwhspr record copy-last or record paste-last",
                        urgency="normal",
                    )
                return InjectionOutcome.FAILED

            if outcome == InjectionOutcome.CONSUMED:
                log("[INJECT] Post-transcription hook consumed transcription")
            else:
                log(f"[INJECT] Injection dispatched ({len(text)} chars)")

            # Text injection succeeded (or was intentionally consumed) - system is fully healthy
            # Cancel any pending background recovery
            if self._background_recovery_needed.is_set():
                log("[HEALTH] Successful recording detected - canceling background recovery")
                self._background_recovery_needed.clear()
                # Write recovery success result (system self-healed via user activity)
                self._write_recovery_result(True, 'user_activity_validated')
                with self._mic_state_lock:
                    self._mic_disconnected = False
                self._clear_error_state_signals()
            try:
                # Ensure any active recovery is aborted once user activity proves health
                self.audio_capture.abort_recovery()
            except Exception:
                pass
            return outcome
        except Exception as e:
            log(f"[ERROR] Text injection failed: {e}")
            return InjectionOutcome.FAILED

    def _is_zero_volume(self, audio_data) -> bool:
        """Check if audio data has zero or near-zero volume"""
        if np is None:
            # numpy not available, can't check - assume not zero
            return False
        
        if audio_data is None or len(audio_data) == 0:
            return True
        
        try:
            # Check if all samples are zero
            if np.all(audio_data == 0.0):
                return True
            
            # Check RMS level (very quiet = likely broken)
            rms = np.sqrt(np.mean(audio_data**2))
            if rms < 1e-6:  # Extremely quiet threshold
                return True
        except Exception:
            # If check fails, assume not zero (safer)
            return False
        
        return False
