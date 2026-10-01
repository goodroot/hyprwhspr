"""Continuous-mode paste on pause and auto-stop on silence."""

import threading

from text_injector import InjectionOutcome
from hallucination import is_hallucination
from service_log import log


class SilenceMixin:
    """Continuous-mode paste on pause and auto-stop on silence."""

    # Continuous mode: auto-paste on speech pause
    _POLL_INTERVAL = 0.1  # seconds between silence checks

    def _calibrate_noise_floor(self, stop_event):
        """Sample the mic's noise floor over ~0.6s. Returns None if stopped/recording ended early."""
        samples = []
        for _ in range(6):
            stop_event.wait(0.1)
            if stop_event.is_set() or not self.is_recording:
                return None
            samples.append(self.audio_capture.rolling_avg_level)
        noise_floor = min(samples)
        return max(noise_floor * 2, 2e-4)

    def _continuous_start_silence_monitor(self):
        """Start monitoring for silence to trigger auto-paste in continuous mode"""
        self._continuous_cancelled = False
        with self._recording_lock:
            self._continuous_delivery_failure_notified = False
        self._continuous_stop_silence_monitor()
        self._continuous_silence_stop.clear()

        silence_seconds = self._get_float_setting('continuous_silence_seconds', 2.0)
        configured_threshold = self._get_float_setting('continuous_silence_threshold', 0)
        samples_needed = max(1, int(silence_seconds / self._POLL_INTERVAL))
        # Optional session end, same rules as toggle/auto auto-stop. Kept in this
        # loop so both timers share one threshold and one level reading.
        stop_timeout = self._get_float_setting('silence_timeout', 0)
        stop_samples = max(1, int(stop_timeout / self._POLL_INTERVAL)) if stop_timeout > 0 else 0
        session = self._recording_session

        def monitor():
            silent_count = 0
            speech_since_flush = False   # a quiet room is not a chunk; don't transcribe it
            quiet_count = 0         # silence since last speech; pastes don't reset it
            heard_speech = False    # don't count toward the stop until speech is heard
            try:
                # Auto-calibrate threshold from noise floor if not manually configured
                threshold = configured_threshold
                if threshold <= 0:
                    threshold = self._calibrate_noise_floor(self._continuous_silence_stop)
                    if threshold is None:
                        return
                    log(f"[CONTINUOUS] Auto-calibrated threshold={threshold:.5f}")

                while self.is_recording and not self._continuous_silence_stop.is_set():
                    raw_level = self.audio_capture.rolling_avg_level
                    if raw_level < threshold:
                        silent_count += 1
                        quiet_count += 1
                        if silent_count >= samples_needed:
                            if speech_since_flush:
                                self._continuous_flush_audio()
                                speech_since_flush = False
                            silent_count = 0
                        if stop_samples and heard_speech and quiet_count >= stop_samples:
                            self._continuous_autostop(session, stop_timeout)
                            return
                    else:
                        silent_count = 0
                        quiet_count = 0
                        heard_speech = speech_since_flush = True
                    self._continuous_silence_stop.wait(self._POLL_INTERVAL)
            except Exception as e:
                log(f"[CONTINUOUS] Silence monitor error: {e}")

        self._continuous_silence_thread = threading.Thread(target=monitor, daemon=True)
        self._continuous_silence_thread.start()

    def _continuous_stop_silence_monitor(self):
        """Stop the continuous silence monitor"""
        self._continuous_silence_stop.set()
        thread = self._continuous_silence_thread
        # The monitor itself reaches here via silence_timeout's _stop_recording()
        if thread and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=0.5)
        self._continuous_silence_thread = None

    def _continuous_autostop(self, session, silence_timeout):
        """End a continuous session from its own monitor thread after `silence_timeout`."""
        self._continuous_silence_stop.set()   # no join: this is the monitor thread
        self._continuous_transcription_done.wait(timeout=30)   # let a chunk paste first
        with self._recording_lock:
            if self._recording_session is not session:
                return   # stopped (and maybe restarted) meanwhile - not ours to stop
        log(f"[AUTOSTOP] {silence_timeout:.1f}s of silence - stopping continuous recording")
        self._stop_recording()   # stop beep; transcribes + pastes only the unflushed tail

    def _autostop_start_silence_monitor(self):
        """Auto-stop recording after `silence_timeout` seconds of silence (toggle/auto modes).

        Arms only after speech has been detected, so it can't fire while the user is
        still composing their first sentence. The silence threshold auto-calibrates from
        the noise floor (same approach as continuous mode). No-op when silence_timeout <= 0.
        """
        silence_timeout = self._get_float_setting('silence_timeout', 0)
        configured_threshold = self._get_float_setting('continuous_silence_threshold', 0)

        with self._autostop_lock:
            self._autostop_teardown_locked()
            if silence_timeout <= 0:
                return  # feature disabled (default)

            stop_event = threading.Event()
            samples_needed = max(1, int(silence_timeout / self._POLL_INTERVAL))

            def monitor():
                thread = threading.current_thread()
                try:
                    threshold = configured_threshold
                    if threshold <= 0:
                        threshold = self._calibrate_noise_floor(stop_event)
                        if threshold is None:
                            return

                    armed = False          # don't count silence until speech has been heard
                    silent_count = 0
                    while self.is_recording and not stop_event.is_set():
                        level = self.audio_capture.rolling_avg_level
                        if level >= threshold:
                            armed = True
                            silent_count = 0
                        elif armed:
                            silent_count += 1
                            if silent_count >= samples_needed:
                                stop_event.set()
                                # Deregister ourselves before handing off to _stop_recording().
                                # If a newer generation already replaced us here, our session
                                # already ended some other way - don't stop whatever is
                                # recording now, it isn't ours.
                                with self._autostop_lock:
                                    still_current = self._autostop_silence_thread is thread
                                    if still_current:
                                        self._autostop_silence_thread = None
                                        self._autostop_silence_stop = None
                                if still_current:
                                    log(f"[AUTOSTOP] {silence_timeout:.1f}s of silence - stopping recording")
                                    self._stop_recording()   # plays the stop beep; transcribes + pastes
                                return
                        stop_event.wait(self._POLL_INTERVAL)
                except Exception as e:
                    log(f"[AUTOSTOP] Silence monitor error: {e}")

            self._autostop_silence_thread = threading.Thread(target=monitor, daemon=True)
            self._autostop_silence_stop = stop_event
            self._autostop_silence_thread.start()

    def _autostop_stop_silence_monitor(self):
        """Stop the auto-stop silence monitor (safe to call from anywhere, incl. the monitor)."""
        with self._autostop_lock:
            self._autostop_teardown_locked()

    def _autostop_teardown_locked(self):
        """Tear down whatever autostop monitor is currently registered. Caller must hold `_autostop_lock`.

        No generation check here (unlike the monitor's self-stop path) - accepted edge case.
        """
        stop_event = self._autostop_silence_stop
        thread = self._autostop_silence_thread
        if stop_event is not None:
            stop_event.set()
        # Avoid self-join deadlock when the monitor thread is the one triggering the stop
        if thread and thread.is_alive() and threading.current_thread() is not thread:
            thread.join(timeout=0.5)
        self._autostop_silence_thread = None
        self._autostop_silence_stop = None

    def _continuous_stop_and_wait(self):
        """Stop the silence monitor and wait for any in-progress transcription"""
        self._continuous_stop_silence_monitor()
        self._continuous_transcription_done.wait(timeout=30)

    def _continuous_flush_audio(self):
        """Flush accumulated audio: transcribe and paste without stopping recording"""
        if not self._continuous_flush_lock.acquire(blocking=False):
            return  # another flush/transcription in progress

        # Lock is now held — all paths must go through the finally that releases it.
        self._continuous_transcription_done.clear()
        should_transcribe = False
        audio_data = None
        try:
            audio_data = self.audio_capture.flush_buffer()
            if audio_data is None or len(audio_data) == 0:
                return

            duration = len(audio_data) / self.audio_capture.sample_rate
            if duration < 0.5 or self._is_zero_volume(audio_data):
                return

            log(f"[CONTINUOUS] Flushing {duration:.1f}s of audio for transcription")
            should_transcribe = True
        except Exception as e:
            log(f"[CONTINUOUS] Flush error: {e}")
        finally:
            if not should_transcribe:
                self._continuous_flush_lock.release()
                self._continuous_transcription_done.set()

        if not should_transcribe:
            return

        # Transcribe in background thread; lock is held until transcription
        # completes so the next flush is blocked until this one finishes.
        def process():
            try:
                transcription = self.whisper_manager.transcribe_audio(
                    audio_data,
                    sample_rate=self.audio_capture.sample_rate,
                    language_override=self._current_language_override,
                )
                if transcription and transcription.strip():
                    text = transcription.strip()
                    if is_hallucination(text, self.config.get_hallucination_markers()):
                        log(f"[CONTINUOUS] Hallucination ignored: {text!r}")
                        return
                    if self._continuous_cancelled:
                        log("[CONTINUOUS] Cancelled — discarding transcription")
                        return
                    outcome = self._inject_text(text)
                    preview = f"{text[:80]}{'...' if len(text) > 80 else ''}"
                    if outcome == InjectionOutcome.INJECTED:
                        log(f"[CONTINUOUS] Pasted: {preview}")
                    elif outcome == InjectionOutcome.CONSUMED:
                        log(f"[CONTINUOUS] Consumed by hook: {preview}")
                    else:
                        log(f"[CONTINUOUS] Injection failed: {preview}")
                else:
                    log("[CONTINUOUS] No transcription from flushed audio")
            except Exception as e:
                log(f"[CONTINUOUS] Transcription error: {e}")
            finally:
                self._notify_capture("", final=True)
                self._continuous_flush_lock.release()
                self._continuous_transcription_done.set()

        self._save_debug_recording(audio_data)
        threading.Thread(target=process, daemon=True).start()
