"""Long-form recording state machine and persistence orchestration."""

import threading

try:
    from .paths import LONGFORM_STATE_FILE
    from .segment_manager import SegmentManager
except ImportError:
    from paths import LONGFORM_STATE_FILE
    from segment_manager import SegmentManager


class LongFormController:
    """Own long-form session state independently of the application shell."""

    def __init__(
        self,
        config,
        audio_capture,
        audio_manager,
        whisper_manager,
        inject_text,
        notify_capture,
        set_visualizer_state,
        show_mic_osd,
        hide_mic_osd,
        show_result_and_hide,
        write_recording_status,
        set_processing,
        hallucination_markers,
        segment_manager_factory=SegmentManager,
        timer_factory=threading.Timer,
        state_file=LONGFORM_STATE_FILE,
    ):
        self.config = config
        self.audio_capture = audio_capture
        self.audio_manager = audio_manager
        self.whisper_manager = whisper_manager
        self.inject_text = inject_text
        self.notify_capture = notify_capture
        self.set_visualizer_state = set_visualizer_state
        self.show_mic_osd = show_mic_osd
        self.hide_mic_osd = hide_mic_osd
        self.show_result_and_hide = show_result_and_hide
        self.write_recording_status = write_recording_status
        self.set_processing = set_processing
        self.hallucination_markers = hallucination_markers
        self.segment_manager_factory = segment_manager_factory
        self.timer_factory = timer_factory
        self.state_file = state_file

        self.state = 'IDLE'
        self.language_override = None
        self.lock = threading.Lock()
        self.segment_manager = None
        self.auto_save_timer = None
        self.error_audio = None

    def ensure_initialized(self):
        if self.segment_manager is not None:
            return
        max_size_mb = self.config.get_setting('long_form_temp_limit_mb', 500)
        self.segment_manager = self.segment_manager_factory(max_size_mb=max_size_mb)
        print("[LONGFORM] Segment manager initialized (lazy init)", flush=True)
        self.cleanup_temp_on_startup()

    def primary_shortcut(self):
        self.ensure_initialized()
        with self.lock:
            if self.state == 'IDLE':
                self.start_recording()
            elif self.state == 'RECORDING':
                self.pause_recording()
            elif self.state == 'PAUSED':
                self.resume_recording()
            elif self.state in ('PROCESSING', 'ERROR'):
                print(f"[LONGFORM] Ignoring shortcut in {self.state} state")

    def cancel_shortcut(self):
        self.ensure_initialized()
        with self.lock:
            self.cancel()

    def submit_shortcut(self):
        self.ensure_initialized()
        with self.lock:
            if self.state in ('RECORDING', 'PAUSED'):
                if self.state == 'RECORDING':
                    audio_data = self.audio_capture.pause_recording()
                    pending_audio = None
                    if audio_data is not None and len(audio_data) > 0:
                        if self.segment_manager.save_segment(audio_data) is None:
                            pending_audio = self.segment_manager.concatenate_readable(audio_data)
                    self.submit(audio_data=pending_audio)
                else:
                    self.submit()
            elif self.state == 'ERROR':
                print("[LONGFORM] Retrying submission")
                self.submit(retry=True)
            elif self.state == 'IDLE':
                print("[LONGFORM] Nothing to submit (IDLE state)")
            elif self.state == 'PROCESSING':
                print("[LONGFORM] Already processing, please wait")

    def request_start(self, language_override=None):
        """Handle an external start command using long-form state semantics."""
        self.ensure_initialized()
        lang_info = f" (language: {language_override})" if language_override else ""
        with self.lock:
            if self.state == 'IDLE':
                print(f"[CONTROL] Long-form start requested (immediate){lang_info}", flush=True)
                self.start_recording(language_override=language_override)
            elif self.state == 'PAUSED':
                print(f"[CONTROL] Long-form resume requested (immediate){lang_info}", flush=True)
                self.resume_recording()
            elif self.state == 'RECORDING':
                print("[CONTROL] Long-form already recording, ignoring start request", flush=True)
            else:
                print(f"[CONTROL] Long-form in {self.state} state, ignoring start request", flush=True)

    def request_pause(self):
        """Handle an external stop command as a long-form pause request."""
        self.ensure_initialized()
        with self.lock:
            if self.state == 'RECORDING':
                print("[CONTROL] Long-form pause requested (immediate)", flush=True)
                self.pause_recording()
            elif self.state == 'PAUSED':
                print("[CONTROL] Long-form already paused, ignoring stop request", flush=True)
            elif self.state == 'IDLE':
                print("[CONTROL] Long-form not recording, ignoring stop request", flush=True)
            else:
                print(f"[CONTROL] Long-form in {self.state} state, ignoring stop request", flush=True)

    def request_cancel(self):
        """Handle an external cancel command, including failed sessions."""
        self.ensure_initialized()
        with self.lock:
            if self.state in ('RECORDING', 'PAUSED', 'ERROR'):
                print("[CONTROL] Long-form cancel requested (immediate)", flush=True)
                self.cancel()
            else:
                print(f"[CONTROL] Long-form in {self.state} state, ignoring cancel request", flush=True)

    def start_recording(self, language_override=None):
        lang_info = f" (language: {language_override})" if language_override else ""
        print(f"[LONGFORM] Starting recording session{lang_info}")
        self.language_override = language_override

        if not self.audio_capture.start_recording():
            print("[LONGFORM] Failed to start audio capture")
            return

        self.segment_manager.start_session()
        self._set_state('RECORDING', visualizer='recording')
        self.show_mic_osd()
        self.start_auto_save_timer()
        self.audio_manager.play_start_sound()

    def pause_recording(self):
        print("[LONGFORM] Pausing recording")
        self.stop_auto_save_timer()
        audio_data = self.audio_capture.pause_recording()
        if audio_data is not None and len(audio_data) > 0:
            if self.segment_manager.save_segment(audio_data) is None:
                self.persistence_failed(audio_data)
                return

        self._set_state('PAUSED', visualizer='paused')
        self.audio_manager.play_stop_sound()

    def resume_recording(self):
        print("[LONGFORM] Resuming recording")
        if not self.audio_capture.resume_recording():
            print("[LONGFORM] Failed to resume audio capture")
            self._set_state('ERROR', visualizer='error')
            return

        self._set_state('RECORDING', visualizer='recording')
        self.start_auto_save_timer()
        self.audio_manager.play_start_sound()

    def cancel(self):
        if self.state not in ('RECORDING', 'PAUSED', 'ERROR'):
            return

        print("[LONGFORM] Recording cancelled (discarded)", flush=True)
        self.notify_capture("", final=True)
        try:
            self.stop_auto_save_timer()
            self.audio_capture.stop_recording()
            self.segment_manager.clear_session()
            self.error_audio = None
            self.language_override = None
            self._set_state('IDLE')
            self.hide_mic_osd()
            self.write_recording_status(False)
            self.audio_manager.play_error_sound()
        except Exception as e:
            print(f"[ERROR] Error cancelling long-form recording: {e}", flush=True)
            try:
                self._set_state('IDLE')
                self.hide_mic_osd()
                self.write_recording_status(False)
            except Exception:
                pass

    def persistence_failed(self, audio_data):
        self.stop_auto_save_timer()
        self.error_audio = self.segment_manager.concatenate_readable(audio_data)
        self._set_state('ERROR', visualizer='error')
        self.audio_manager.play_error_sound()

    def submit(self, retry=False, audio_data=None):
        print("[LONGFORM] Submitting for transcription")
        self.stop_auto_save_timer()

        if audio_data is None:
            if retry and self.error_audio is not None:
                audio_data = self.error_audio
            else:
                audio_data = self.segment_manager.concatenate_all()

        if audio_data is None or len(audio_data) == 0:
            print("[LONGFORM] No audio data to process")
            if self.segment_manager.has_segments():
                self.error_audio = self.segment_manager.concatenate_readable()
                self._set_state('ERROR', visualizer='error')
            else:
                self._set_state('IDLE')
                self.hide_mic_osd()
            return

        self._set_state('PROCESSING', visualizer='processing')
        try:
            self.set_processing(True)
            transcription = self.whisper_manager.transcribe_audio(
                audio_data,
                sample_rate=self.audio_capture.sample_rate,
                language_override=self.language_override,
            )

            if transcription and transcription.strip():
                text = transcription.strip()
                normalized = text.lower().replace('_', ' ').strip('[]().!?, ')
                if normalized in self.hallucination_markers or text.startswith('♪'):
                    print(f"[LONGFORM] Whisper hallucination detected: {text!r}")
                    self._submission_failed(audio_data)
                    return

                if not self.inject_text(text):
                    self._submission_failed(audio_data)
                    return

                self.segment_manager.clear_session()
                self.error_audio = None
                self.language_override = None
                self._set_state('IDLE')
                self.show_result_and_hide(True)
            else:
                print("[LONGFORM] No transcription generated")
                self._submission_failed(audio_data)
        except Exception as e:
            print(f"[LONGFORM] Transcription error: {e}", flush=True)
            self._submission_failed(audio_data)
        finally:
            self.notify_capture("", final=True)
            self.set_processing(False)

    def _submission_failed(self, audio_data):
        self.audio_manager.play_error_sound()
        self.error_audio = audio_data
        self._set_state('ERROR', visualizer='error')

    def start_auto_save_timer(self):
        interval = self.config.get_setting('long_form_auto_save_interval', 300)
        if interval <= 0:
            return

        def auto_save_callback():
            with self.lock:
                if self.state == 'RECORDING':
                    audio_data = self.audio_capture.get_current_audio_copy()
                    if audio_data is not None and len(audio_data) > 0:
                        if self.segment_manager.save_segment(audio_data) is None:
                            frozen_audio = self.audio_capture.pause_recording()
                            if frozen_audio is None or len(frozen_audio) == 0:
                                frozen_audio = audio_data
                            self.persistence_failed(frozen_audio)
                            return
                        self.audio_capture.clear_buffer()
                        print(f"[LONGFORM] Auto-saved segment ({len(audio_data) / 16000:.1f}s)")
                    self.start_auto_save_timer()

        self.auto_save_timer = self.timer_factory(interval, auto_save_callback)
        self.auto_save_timer.daemon = True
        self.auto_save_timer.start()

    def stop_auto_save_timer(self):
        if self.auto_save_timer is not None:
            self.auto_save_timer.cancel()
            self.auto_save_timer = None

    def write_state(self, state):
        try:
            self.state_file.parent.mkdir(parents=True, exist_ok=True)
            self.state_file.write_text(state)
        except Exception as e:
            print(f"[LONGFORM] Failed to write state file: {e}", flush=True)

    def _set_state(self, state, visualizer=None):
        self.state = state
        self.write_state(state)
        if visualizer is not None:
            self.set_visualizer_state(visualizer)

    def cleanup_temp_on_startup(self):
        if self.segment_manager is None:
            return
        try:
            total_size = self.segment_manager.get_total_size()
            max_size = self.segment_manager.max_size_bytes
            if total_size > max_size:
                print(f"[LONGFORM] Temp directory over limit ({total_size / 1024 / 1024:.1f}MB > {max_size / 1024 / 1024:.1f}MB)")
                print("[LONGFORM] Cleaning up oldest segments...")
                while self.segment_manager.cleanup_oldest():
                    if self.segment_manager.get_total_size() <= max_size:
                        break
                final_size = self.segment_manager.get_total_size()
                print(f"[LONGFORM] Cleanup complete. New size: {final_size / 1024 / 1024:.1f}MB")
            elif total_size > 0:
                print(f"[LONGFORM] Found {total_size / 1024 / 1024:.1f}MB of previous segments (limit: {max_size / 1024 / 1024:.1f}MB)")
        except Exception as e:
            print(f"[LONGFORM] Error during startup cleanup: {e}", flush=True)
