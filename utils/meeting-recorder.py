#!/usr/bin/env python3
"""
meeting-recorder - Capture system audio via PipeWire, transcribe with hyprwhspr backends.

Records what's playing through your speakers (meeting audio from other participants)
and optionally your microphone, transcribes in chunks using whatever backend you
have configured in hyprwhspr (pywhispercpp, onnx-asr, rest-api), and serves the
transcript over HTTP.

Requirements:
    - PipeWire (pw-record, pactl)
    - hyprwhspr installed and configured (any transcription backend)

Usage:
    python3 utils/meeting-recorder.py

    curl -X POST localhost:8765/start                     # system audio only
    curl -X POST localhost:8765/start -d '{"mic": true}'  # system audio + mic
    curl -X POST localhost:8765/stop                      # stop & transcribe
    curl localhost:8765/transcript                         # get full transcript
    curl localhost:8765/status                             # check state

Environment variables:
    MEETING_PORT            HTTP port (default: 8765)
    MEETING_CHUNK_SECS      Transcription interval in seconds (default: 300)
    MEETING_TRANSCRIPT_DIR  Where to save transcripts
    MEETING_LANGUAGE        Language hint for Whisper (e.g. "en", "de", "it")
"""

import json
import os
import signal
import subprocess
import sys
import threading
import time
from datetime import datetime
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

# Add hyprwhspr's lib directories to the import path
_script_dir = Path(__file__).resolve().parent
_lib_dir = _script_dir.parent / "lib"
_src_dir = _lib_dir / "src"
for p in [str(_lib_dir), str(_src_dir)]:
    if p not in sys.path:
        sys.path.insert(0, p)

import numpy as np
from config_manager import ConfigManager
from whisper_manager import WhisperManager

SAMPLE_RATE = 16000
CHANNELS = 1
BYTES_PER_SAMPLE = 2  # s16le


class MeetingRecorder:
    def __init__(self):
        self.recording = False
        self.lock = threading.Lock()
        self.procs = []
        self.system_buf = bytearray()
        self.mic_buf = bytearray()
        self.transcripts = []
        self.start_time = None
        self.include_mic = False
        self._threads = []

        self.chunk_secs = int(os.environ.get("MEETING_CHUNK_SECS", "300"))
        self.language = os.environ.get("MEETING_LANGUAGE", "")
        self.transcript_dir = Path(
            os.environ.get(
                "MEETING_TRANSCRIPT_DIR",
                Path.home() / ".local/share/meeting-recorder",
            )
        )
        self.transcript_dir.mkdir(parents=True, exist_ok=True)

        # Use hyprwhspr's transcription infrastructure
        self.config = ConfigManager()
        self.whisper = WhisperManager(self.config)
        print("[init] Initializing transcription backend...", flush=True)
        if not self.whisper.initialize():
            raise RuntimeError(
                "Failed to initialize transcription backend. "
                "Check your hyprwhspr config: hyprwhspr config show"
            )
        self.backend_name = self.whisper.get_backend_info()
        print(f"[init] Backend ready: {self.backend_name}", flush=True)

    # -- PipeWire helpers --

    def _get_default_sink(self):
        r = subprocess.run(
            ["pactl", "get-default-sink"], capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            raise RuntimeError(f"pactl get-default-sink failed: {r.stderr.strip()}")
        return r.stdout.strip()

    def _get_default_source(self):
        r = subprocess.run(
            ["pactl", "get-default-source"], capture_output=True, text=True, timeout=5
        )
        if r.returncode != 0:
            raise RuntimeError(f"pactl get-default-source failed: {r.stderr.strip()}")
        return r.stdout.strip()

    def _get_monitor_source(self):
        """Get the monitor source name for the default sink.

        PipeWire exposes a .monitor source for every sink, which captures
        whatever audio is being played to that sink (system/desktop audio).
        """
        sink = self._get_default_sink()
        monitor = f"{sink}.monitor"

        # Verify the monitor source exists
        r = subprocess.run(
            ["pactl", "list", "short", "sources"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if monitor in r.stdout:
            return monitor

        # Fallback: target the sink directly. PipeWire connects capture
        # streams to a sink's monitor ports automatically.
        print(f"[warn] Monitor source '{monitor}' not found, targeting sink directly")
        return sink

    def _spawn_capture(self, target):
        """Spawn parec to capture raw s16le PCM from a PipeWire/PulseAudio source."""
        cmd = [
            "parec",
            f"--device={target}",
            "--rate=%d" % SAMPLE_RATE,
            "--channels=%d" % CHANNELS,
            "--format=s16le",
            "--raw",
        ]
        proc = subprocess.Popen(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL
        )
        time.sleep(0.3)
        if proc.poll() is not None:
            raise RuntimeError(f"parec exited immediately (device: {target})")
        return proc

    # -- Buffer management --

    def _reader(self, proc, buf_name):
        """Read raw PCM from a pw-record process into the named buffer."""
        buf = getattr(self, buf_name)
        chunk_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE  # 1 second
        try:
            while self.recording and proc.poll() is None:
                data = proc.stdout.read(chunk_bytes)
                if data:
                    with self.lock:
                        buf.extend(data)
        except Exception as e:
            print(f"[reader:{buf_name}] {e}", flush=True)

    def _drain_buffers(self):
        """Drain audio buffers and return (system_pcm, mic_pcm) tuple."""
        with self.lock:
            system = bytes(self.system_buf)
            self.system_buf.clear()
            mic = bytes(self.mic_buf) if self.include_mic else b""
            self.mic_buf.clear()
        return (system or None, mic or None)

    # -- Transcription --

    def _pcm_to_float32(self, pcm):
        """Convert raw s16le PCM bytes to float32 numpy array."""
        return np.frombuffer(pcm, dtype=np.int16).astype(np.float32) / 32768.0

    def _transcribe_pcm(self, pcm):
        """Transcribe a single PCM buffer. Returns text or empty string."""
        audio = self._pcm_to_float32(pcm)
        kwargs = {}
        if self.language:
            kwargs["language_override"] = self.language
        return self.whisper.transcribe_audio(audio, sample_rate=SAMPLE_RATE, **kwargs)

    def _transcribe_chunk(self, system_pcm, mic_pcm):
        """Transcribe system and mic audio separately, return combined text."""
        min_bytes = SAMPLE_RATE * BYTES_PER_SAMPLE  # 1 second minimum
        parts = []

        if system_pcm and len(system_pcm) > min_bytes:
            dur = len(system_pcm) / SAMPLE_RATE / BYTES_PER_SAMPLE
            print(f"[transcribe] System audio: {dur:.0f}s", flush=True)
            text = self._transcribe_pcm(system_pcm)
            if text:
                parts.append(f"[System] {text}")

        if mic_pcm and len(mic_pcm) > min_bytes:
            dur = len(mic_pcm) / SAMPLE_RATE / BYTES_PER_SAMPLE
            print(f"[transcribe] Mic audio: {dur:.0f}s", flush=True)
            text = self._transcribe_pcm(mic_pcm)
            if text:
                parts.append(f"[You] {text}")

        return "\n".join(parts)

    def _chunk_loop(self):
        """Periodically drain buffers and transcribe."""
        while self.recording:
            time.sleep(self.chunk_secs)
            if not self.recording:
                break
            system_pcm, mic_pcm = self._drain_buffers()
            text = self._transcribe_chunk(system_pcm, mic_pcm)
            if text:
                self.transcripts.append(text)
                print(f"[chunk] #{len(self.transcripts)} done", flush=True)

    # -- Control --

    def start(self, include_mic=False):
        if self.recording:
            return {"error": "already recording"}

        self.include_mic = include_mic
        self.transcripts = []
        self.system_buf.clear()
        self.mic_buf.clear()
        self._threads = []
        self.procs = []

        # System audio: default sink's monitor
        try:
            monitor = self._get_monitor_source()
            print(f"[start] System audio: {monitor}", flush=True)
            self.procs.append(self._spawn_capture(monitor))
        except Exception as e:
            return {"error": f"System audio capture failed: {e}"}

        # Optional: microphone
        if include_mic:
            try:
                source = self._get_default_source()
                print(f"[start] Microphone: {source}", flush=True)
                self.procs.append(self._spawn_capture(source))
            except Exception as e:
                self.procs[0].terminate()
                self.procs.clear()
                return {"error": f"Mic capture failed: {e}"}

        self.recording = True
        self.start_time = time.time()

        # Reader threads
        t = threading.Thread(
            target=self._reader, args=(self.procs[0], "system_buf"), daemon=True
        )
        t.start()
        self._threads.append(t)

        if include_mic and len(self.procs) > 1:
            t = threading.Thread(
                target=self._reader, args=(self.procs[1], "mic_buf"), daemon=True
            )
            t.start()
            self._threads.append(t)

        # Chunk timer
        t = threading.Thread(target=self._chunk_loop, daemon=True)
        t.start()
        self._threads.append(t)

        mode = "system+mic" if include_mic else "system"
        return {"status": "recording", "mode": mode, "chunk_secs": self.chunk_secs}

    def stop(self):
        if not self.recording:
            return {"error": "not recording"}

        duration = time.time() - self.start_time
        self.recording = False

        for p in self.procs:
            p.terminate()
        for p in self.procs:
            try:
                p.wait(timeout=5)
            except subprocess.TimeoutExpired:
                p.kill()

        for t in self._threads:
            t.join(timeout=5)

        # Transcribe remaining audio
        system_pcm, mic_pcm = self._drain_buffers()
        text = self._transcribe_chunk(system_pcm, mic_pcm)
        if text:
            self.transcripts.append(text)

        full = "\n\n".join(self.transcripts)

        if full.strip():
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            path = self.transcript_dir / f"meeting_{ts}.txt"
            path.write_text(full)
            print(f"[stop] Saved: {path}", flush=True)

        self.procs.clear()
        return {
            "status": "stopped",
            "duration_secs": round(duration),
            "chunks": len(self.transcripts),
            "transcript": full,
        }

    def status(self):
        s = {"recording": self.recording, "chunks": len(self.transcripts)}
        if self.recording and self.start_time:
            s["duration_secs"] = round(time.time() - self.start_time)
            s["mode"] = "system+mic" if self.include_mic else "system"
        return s

    def transcript(self):
        return "\n\n".join(self.transcripts)


# -- HTTP server --


class Handler(BaseHTTPRequestHandler):
    recorder: MeetingRecorder = None  # type: ignore

    def do_POST(self):
        body = self._read_body()
        if self.path == "/start":
            mic = body.get("mic", False) if body else False
            self._json(self.recorder.start(include_mic=mic))
        elif self.path == "/stop":
            self._json(self.recorder.stop())
        else:
            self._json({"error": "not found"}, 404)

    def do_GET(self):
        if self.path == "/status":
            self._json(self.recorder.status())
        elif self.path == "/transcript":
            self._json({"transcript": self.recorder.transcript()})
        else:
            self._json({"error": "not found"}, 404)

    def _read_body(self):
        length = int(self.headers.get("Content-Length", 0))
        if length:
            try:
                return json.loads(self.rfile.read(length))
            except Exception:
                pass
        return None

    def _json(self, data, code=200):
        payload = json.dumps(data, indent=2).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, fmt, *args):
        pass  # quiet


def main():
    port = int(os.environ.get("MEETING_PORT", "8765"))
    recorder = MeetingRecorder()
    Handler.recorder = recorder

    server = HTTPServer(("127.0.0.1", port), Handler)

    def shutdown(sig, frame):
        print("\nShutting down...", flush=True)
        if recorder.recording:
            recorder.stop()
        threading.Thread(target=server.shutdown, daemon=True).start()

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    print(
        f"\nmeeting-recorder  http://127.0.0.1:{port}\n"
        f"\n"
        f'  POST /start           Begin recording ({{"mic": true}} for mic+system)\n'
        f"  POST /stop            Stop and transcribe\n"
        f"  GET  /status          Recording state\n"
        f"  GET  /transcript      Full transcript\n"
        f"\n"
        f"  Backend: {recorder.backend_name}\n"
        f"  Chunks:  every {recorder.chunk_secs}s\n"
        f"  Save:    {recorder.transcript_dir}\n",
        flush=True,
    )

    server.serve_forever()


if __name__ == "__main__":
    main()
