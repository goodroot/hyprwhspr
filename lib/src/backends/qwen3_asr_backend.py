"""Qwen3-ASR batch backend using a private pinned llama-server sidecar."""

import ctypes
import http.client
import json
import os
import re
import signal
import socket
import subprocess
import time
import uuid
from typing import Optional

try:
    from ..dependencies import require_package
    from ..text_script import join_segments
except ImportError:
    from dependencies import require_package
    from text_script import join_segments

np = require_package('numpy')

try:
    from ..backend_utils import language_name
    from ..qwen3_asr_runtime import (DEFAULT_MODEL, QWEN3_ASR_LOG,
                                     QWEN3_ASR_MAX_AUDIO_SECONDS, QWEN3_ASR_MODELS,
                                     QWEN3_ASR_SOCKET, library_dir, model_paths,
                                     resolve_device, server_path)
except ImportError:
    from backend_utils import language_name
    from qwen3_asr_runtime import (DEFAULT_MODEL, QWEN3_ASR_LOG,
                                   QWEN3_ASR_MAX_AUDIO_SECONDS, QWEN3_ASR_MODELS,
                                   QWEN3_ASR_SOCKET, library_dir, model_paths,
                                   resolve_device, server_path)

from .base import TranscriptionBackend


# llama-server leaks the model's own preamble into /v1/audio/transcriptions
# responses ("language English<asr_text>…"), which clients are not expected to
# handle — see ggml-org/llama.cpp#26749 (open). vLLM strips the same markers in
# post-processing. Remove this once upstream does it server-side.
_PREFIX = re.compile(r"^\s*language\s+([^<\r\n]+?)\s*<asr_text>\s*", re.IGNORECASE)
_TRAILING_TOKENS = re.compile(
    r"\s*(?:<\|(?:endoftext|im_end|end)\|>|</s>|<asr_text>)\s*$", re.IGNORECASE)

_PR_SET_PDEATHSIG = 1

# Minimum gap between sidecar restarts. This bounds a crash loop — including the
# retry inside a single transcribe() — without ever permanently disabling the
# backend: two unrelated crashes hours apart each get their own restart, where a
# one-shot flag would have refused the second until the service was restarted.
_RESTART_COOLDOWN_SECONDS = 60

# sockaddr_un.sun_path is 108 bytes on Linux, NUL included.
_MAX_UNIX_SOCKET_BYTES = 107

# Target chunk length, with headroom under QWEN3_ASR_MAX_AUDIO_SECONDS so a
# boundary nudged later by the quiet-point search still fits.
_CHUNK_TARGET_SECONDS = 110
# How far we hunt for a natural pause. Asymmetric on purpose: forward room is
# limited by the hard cap, but reaching further back costs only chunk length, so
# the window is wide enough to contain a sentence gap at normal speaking pace.
# Too narrow and a seam lands mid-utterance, where the decoder tends to complete
# or restart the sentence and inflate the transcript.
_CHUNK_SEARCH_BACK_SECONDS = 25
_CHUNK_SEARCH_FORWARD_SECONDS = 10
# Resolution of that hunt. 100 ms is short enough to land inside a real pause
# and long enough that one quiet sample cannot masquerade as silence.
_CHUNK_BUCKET_SECONDS = 0.1
# Never leave a sliver as the final chunk: cuts land a shade under the cap, so a
# clean multiple of the cap can otherwise strand a fraction of a second into its
# own sidecar round trip that transcribes to nothing.
_CHUNK_MIN_TAIL_SECONDS = 3

# Resolved at import, never inside preexec_fn: that hook runs between fork and
# exec, where dlopen() can deadlock if another thread held the loader lock at
# fork time — and the service is multithreaded.
try:
    _LIBC = ctypes.CDLL("libc.so.6", use_errno=True)
except OSError:
    _LIBC = None


class _UnixHTTPConnection(http.client.HTTPConnection):
    def __init__(self, socket_path: str, timeout: float):
        super().__init__("localhost", timeout=timeout)
        self.socket_path = socket_path

    def connect(self):
        self.sock = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        self.sock.settimeout(self.timeout)
        self.sock.connect(self.socket_path)


def _pause_offset(window, sample_rate: int):
    """Offset of the LAST clear pause in `window`, or None if there is none.

    Returning None rather than a midpoint lets the caller fall back to its own
    target, which is not the window's centre now that the search is asymmetric.

    Same bucketed-RMS reduction the mic-OSD meter uses (audio_capture.py), applied
    here to find a pause rather than to draw one.

    Latest rather than quietest: any real pause is a fine place to cut, so the one
    nearest the cap wins and keeps chunks long. Picking the global minimum would
    cut early whenever an earlier pause happened to be quieter.
    """
    bucket = max(1, int(_CHUNK_BUCKET_SECONDS * sample_rate))
    count = window.size // bucket
    if count < 2:
        return None
    usable = count * bucket
    levels = np.sqrt(np.mean(window[:usable].reshape(count, -1).astype(np.float64) ** 2, axis=1))
    average = float(levels.mean())
    if average <= 0:
        return None
    # "Clear" means well below the window's own level, so this adapts to quiet
    # and loud recordings alike rather than using an absolute amplitude.
    quiet = np.flatnonzero(levels < average * 0.5)
    if quiet.size == 0:
        return None
    # Centre of that bucket keeps the cut furthest from the adjacent speech.
    return int(quiet[-1] * bucket + bucket // 2)


def split_for_transcription(audio_data, sample_rate: int, max_seconds: int):
    """Split audio into chunks under `max_seconds`, cutting at quiet points.

    Returns [audio_data] unchanged when it already fits, so the common case runs
    exactly the path it did before chunking existed.
    """
    if sample_rate <= 0:
        return [audio_data]
    if len(audio_data) / sample_rate <= max_seconds:
        return [audio_data]

    target = int(_CHUNK_TARGET_SECONDS * sample_rate)
    search_back = int(_CHUNK_SEARCH_BACK_SECONDS * sample_rate)
    search_forward = int(_CHUNK_SEARCH_FORWARD_SECONDS * sample_rate)
    hard_limit = int(max_seconds * sample_rate)
    chunks = []
    start = 0
    total = len(audio_data)
    while start < total:
        if total - start <= hard_limit:
            chunks.append(audio_data[start:])
            break
        # Hunt for a pause around the target, clamped so the cut can never
        # produce a chunk over the hard limit or fail to advance.
        ideal = start + target
        low = max(start + 1, ideal - search_back)
        high = min(start + hard_limit, ideal + search_forward, total)
        # Keep the runt constraint inside the search window rather than nudging
        # the result afterwards: a post-hoc shift walks the cut off the pause it
        # just found, which is exactly what inflates the transcript.
        min_tail = int(_CHUNK_MIN_TAIL_SECONDS * sample_rate)
        if total - high < min_tail:
            high = min(high, max(low, total - min_tail))
        if high <= low:
            end = min(start + hard_limit, total)
        else:
            offset = _pause_offset(audio_data[low:high], sample_rate)
            # No clear pause: cut at the target rather than anywhere arbitrary.
            end = min(ideal, high) if offset is None else max(low, min(low + offset, high))
        chunks.append(audio_data[start:end])
        start = end
    return chunks


def _die_with_parent():
    """Ask the kernel to SIGTERM the sidecar if hyprwhspr dies abruptly.

    Without this a SIGKILL of the service orphans llama-server holding the whole
    model resident; the next start binds a fresh socket, so the orphan leaks
    invisibly.
    """
    if _LIBC is None:
        return
    try:
        _LIBC.prctl(_PR_SET_PDEATHSIG, signal.SIGTERM)
    except Exception:
        pass


class Qwen3AsrBackend(TranscriptionBackend):
    name = "qwen3-asr"
    # Idle reinit exists to refresh a GPU context invalidated by suspend. The
    # sidecar holds its own context out of process, so a long idle costs nothing
    # and reloading would only make the next recording pay a full model load.
    reinit_on_idle = False
    reinit_on_resume = True

    def __init__(self, manager):
        super().__init__(manager)
        self._process = None
        self._socket_path = QWEN3_ASR_SOCKET
        self._last_restart = None

    @property
    def is_loaded(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def _device(self) -> str:
        return resolve_device(self.config)

    @staticmethod
    def _log_tail(limit: int = 4) -> str:
        """Last few sidecar log lines, for an otherwise unexplainable failure."""
        try:
            lines = QWEN3_ASR_LOG.read_text(encoding="utf-8", errors="replace").splitlines()
        except OSError:
            return ""
        return " | ".join(line.strip() for line in lines[-limit:] if line.strip())

    def _start(self) -> bool:
        model_name = self.current_model
        decoder, projector = model_paths(model_name)
        device = self._device()
        server = server_path(device)
        for path in (server, decoder, projector):
            if not path.exists():
                print(f"[ERROR] Qwen3-ASR component not found: {path}", flush=True)
                print("[ERROR] Run 'hyprwhspr setup' to install the Qwen3-ASR backend", flush=True)
                return False
        # AF_UNIX sun_path holds 108 bytes including the NUL. Over that, the
        # bind fails inside llama-server and surfaces only as its own opaque
        # "couldn't bind HTTP server socket … port: 8080", which reads like a
        # port conflict and sends people hunting for the wrong thing.
        socket_bytes = len(str(self._socket_path).encode("utf-8"))
        if socket_bytes > _MAX_UNIX_SOCKET_BYTES:
            print(f"[ERROR] Qwen3-ASR socket path is {socket_bytes} bytes; the kernel "
                  f"limit is {_MAX_UNIX_SOCKET_BYTES}: {self._socket_path}", flush=True)
            print("[ERROR] Set a shorter XDG_RUNTIME_DIR for hyprwhspr", flush=True)
            return False
        self._socket_path.parent.mkdir(parents=True, exist_ok=True)
        self._socket_path.unlink(missing_ok=True)
        # The .sock suffix is load-bearing: llama-server binds an AF_UNIX socket
        # only when --host ends in .sock (llama.cpp#12613), and otherwise tries
        # to resolve the value as a hostname.
        args = [str(server), "-m", str(decoder), "--mmproj", str(projector),
                "--host", str(self._socket_path), "--threads",
                str(self.config.get_setting("threads", 4)), "--parallel", "1"]
        env = os.environ.copy()
        lib_dir = library_dir(device)
        env["LD_LIBRARY_PATH"] = str(lib_dir) + (
            ":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")
        log_handle = None
        try:
            # Never an unread PIPE: llama-server is verbose at load, and once the
            # 64 KiB pipe buffer filled it would block on write forever.
            QWEN3_ASR_LOG.parent.mkdir(parents=True, exist_ok=True)
            log_handle = QWEN3_ASR_LOG.open("wb")
            self._process = subprocess.Popen(args, stdin=subprocess.DEVNULL,
                                             stdout=subprocess.DEVNULL,
                                             stderr=log_handle, env=env,
                                             preexec_fn=_die_with_parent)
            # Floor of 60 s, but a raised qwen3_asr_timeout must be able to
            # extend it: a cold 2.4 GB Q8 load or a first-run Vulkan pipeline
            # build can exceed a minute.
            deadline = time.monotonic() + max(60, self._timeout())
            while time.monotonic() < deadline:
                if self._process.poll() is not None:
                    raise RuntimeError("llama-server exited during startup")
                try:
                    status, _ = self._request("GET", "/health", timeout=1)
                    if status == 200:
                        return True
                except (OSError, http.client.HTTPException):
                    pass
                time.sleep(0.1)
            raise TimeoutError("llama-server health check timed out")
        except Exception as exc:
            detail = self._log_tail()
            print(f"[ERROR] Qwen3-ASR initialization failed: {exc}", flush=True)
            if detail:
                print(f"[ERROR] llama-server: {detail}", flush=True)
            self._stop()
            return False
        finally:
            if log_handle is not None:
                log_handle.close()

    def _timeout(self) -> int:
        value = int(self.config.get_setting("qwen3_asr_timeout", 180))
        return max(1, min(600, value))

    def initialize(self) -> bool:
        self.cleanup()
        self.current_model = self.config.get_setting("qwen3_asr_model", DEFAULT_MODEL)
        if self.current_model not in QWEN3_ASR_MODELS:
            print(f"[ERROR] Unsupported Qwen3-ASR model: {self.current_model}", flush=True)
            return False
        self._last_restart = None
        self.ready = self._start()
        if self.ready:
            self._last_use_time = time.monotonic()
        return self.ready

    def _request(self, method, path, body=None, headers=None, timeout=None):
        conn = _UnixHTTPConnection(str(self._socket_path), timeout or self._timeout())
        try:
            conn.request(method, path, body=body, headers=headers or {})
            response = conn.getresponse()
            return response.status, response.read()
        finally:
            conn.close()

    @staticmethod
    def _split_preamble(text: str):
        """Return (detected language name or None, cleaned transcript).

        llama-server hands back the model's raw output, preamble included — see
        `to_json_oaicompat_asr` in tools/server/server-task.cpp, which does no
        post-processing. That leak is ggml-org/llama.cpp#26749, and it is the
        only channel through which the detected language is observable.

        We use it opportunistically and never depend on it: when upstream stops
        leaking the preamble there is simply nothing to capture, the caller pins
        nothing, and behaviour is what it would have been anyway.
        """
        match = _PREFIX.match(text or "")
        language = None
        if match:
            language = (match.group(1) or "").strip() or None
            text = text[match.end():]
        previous = None
        while previous != text:
            previous = text
            text = _TRAILING_TOKENS.sub("", text)
        return language, text.strip()

    @classmethod
    def _clean_text(cls, text: str) -> str:
        return cls._split_preamble(text)[1]

    def _multipart(self, wav_bytes: bytes, language: Optional[str]):
        boundary = "hyprwhspr-" + uuid.uuid4().hex
        chunks = []
        def field(name, value):
            chunks.extend([f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n".encode(),
                           str(value).encode(), b"\r\n"])
        field("model", "qwen3-asr")
        field("response_format", "json")
        if language:
            field("language", language)
        chunks.extend([f"--{boundary}\r\nContent-Disposition: form-data; name=\"file\"; filename=\"audio.wav\"\r\nContent-Type: audio/wav\r\n\r\n".encode(),
                       wav_bytes, b"\r\n", f"--{boundary}--\r\n".encode()])
        return b"".join(chunks), f"multipart/form-data; boundary={boundary}"

    def transcribe(self, audio_data, sample_rate=16000,
                   language_override: Optional[str] = None) -> str:
        language = language_override if language_override is not None else self.config.get_setting("language", None)
        # Resolve to a name once: llama.cpp appends this verbatim to the ASR
        # prompt and Qwen reasons in language names, not ISO codes.
        hint = language_name(language)
        # llama.cpp returns nothing for audio past ~2 minutes (llama.cpp#21847),
        # so long recordings are split at pauses and rejoined. Every caller —
        # live, long-form, continuous, file CLI — arrives here, so this is the
        # one place that has to handle it.
        chunks = split_for_transcription(audio_data, sample_rate, QWEN3_ASR_MAX_AUDIO_SECONDS)
        if len(chunks) == 1:
            return self._transcribe_one(chunks[0], sample_rate, hint)[0]

        total_seconds = len(audio_data) / sample_rate
        print(f"[QWEN3-ASR] Splitting {total_seconds:.1f}s into {len(chunks)} chunks", flush=True)
        parts = []
        failed = 0
        for index, chunk in enumerate(chunks, start=1):
            print(f"[QWEN3-ASR] chunk {index}/{len(chunks)} "
                  f"({len(chunk) / sample_rate:.1f}s)", flush=True)
            text, detected = self._transcribe_one(chunk, sample_rate, hint)
            if text:
                parts.append(text)
                # Pin the language the model reported, so one ambiguous chunk
                # later on cannot switch the transcript mid-way. Only from a
                # chunk that actually produced text: an opening stretch of
                # silence or music would otherwise mispin the whole session.
                if hint is None and detected:
                    hint = detected
                    print(f"[QWEN3-ASR] pinning detected language: {detected}", flush=True)
            else:
                # Keep what we have: partial output beats discarding a long
                # dictation because one chunk failed.
                failed += 1
                print(f"[QWEN3-ASR] chunk {index}/{len(chunks)} produced no text", flush=True)
        # join_segments omits the space between CJK neighbours, which matters
        # for the languages this backend exists to serve.
        result = join_segments(parts)
        if failed and result:
            # A truncated transcript otherwise lands in the user's document with
            # nothing to say part of it is missing. A total failure needs no
            # notification: "" already triggers the error sound and failed OSD.
            self._notify_incomplete(failed, len(chunks))
        return result

    @staticmethod
    def _notify_incomplete(failed: int, total: int) -> None:
        try:
            try:
                from ..desktop_notify import notify
            except ImportError:
                from desktop_notify import notify
            notify("hyprwhspr",
                   f"Qwen3-ASR: {failed} of {total} segments failed. "
                   "The transcript is incomplete.",
                   urgency="critical")
        except Exception:
            pass

    def _transcribe_one(self, audio_data, sample_rate: int, hint):
        """One request for one already-short-enough chunk.

        Returns (text, detected_language). `hint` is already a language name, so
        it is sent as-is rather than mapped again.
        """
        body = content_type = None
        for attempt in range(2):
            if not self.is_loaded:
                if not self.ready:
                    # Deliberately unloaded (`model unload`). WhisperManager
                    # already refuses to dispatch in this state; this keeps the
                    # backend correct on its own terms rather than relying on a
                    # caller's invariant.
                    print("[ERROR] Qwen3-ASR model is unloaded; run 'hyprwhspr model reload'", flush=True)
                    return "", None
                since = (None if self._last_restart is None
                         else time.monotonic() - self._last_restart)
                if since is not None and since < _RESTART_COOLDOWN_SECONDS:
                    print(f"[ERROR] Qwen3-ASR sidecar restarted {since:.0f}s ago; "
                          f"not restarting again within {_RESTART_COOLDOWN_SECONDS}s", flush=True)
                    return "", None
                self._last_restart = time.monotonic()
                self._stop()
                if not self._start():
                    return "", None
            if body is None:
                # Built lazily so a chunk we end up refusing never costs a
                # multi-megabyte WAV encode. llama.cpp appends `language`
                # verbatim to the ASR prompt, and Qwen reasons in language names
                # rather than ISO codes.
                body, content_type = self._multipart(
                    self._numpy_to_wav_bytes(audio_data, sample_rate), hint)
            try:
                status, payload = self._request("POST", "/v1/audio/transcriptions", body,
                                                {"Content-Type": content_type,
                                                 "Content-Length": str(len(body))})
                if status != 200:
                    print(f"[ERROR] Qwen3-ASR server returned HTTP {status}", flush=True)
                    return "", None
                result = json.loads(payload.decode("utf-8"))
                text = result.get("text", "") if isinstance(result, dict) else ""
                self._last_use_time = time.monotonic()
                detected, cleaned = self._split_preamble(text)
                return cleaned, detected
            except (OSError, http.client.HTTPException) as exc:
                # Retry only when the child actually died. A timeout from a live
                # server must not create two expensive inference jobs.
                if attempt == 0 and not self.is_loaded:
                    continue
                print(f"[ERROR] Qwen3-ASR transcription failed: {exc}", flush=True)
                return "", None
            except (ValueError, json.JSONDecodeError) as exc:
                print(f"[ERROR] Qwen3-ASR returned invalid JSON: {exc}", flush=True)
                return "", None
        return "", None

    def _stop(self):
        process, self._process = self._process, None
        if process is not None:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait(timeout=5)
            else:
                process.wait()
        self._socket_path.unlink(missing_ok=True)

    def unload(self) -> None:
        self._stop()
        self.ready = False

    def reinitialize(self) -> bool:
        self._stop()
        # A deliberate reinit is not a crash restart, so it clears the cooldown.
        self._last_restart = None
        self.ready = self._start()
        return self.ready

    def cleanup(self) -> None:
        self._stop()
