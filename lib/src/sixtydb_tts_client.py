"""
60db Text-to-Speech client (library + CLI helper).

Synthesizes speech from text over 60db's WebSocket TTS API
(wss://api.60db.ai/ws/tts) and lists the caller's voices via the REST
endpoint (GET https://api.60db.ai/myvoices).

This is a standalone capability: hyprwhspr's dictation pipeline does not consume
TTS, so nothing here is wired to a hotkey. It exists for the `hyprwhspr 60db`
CLI commands and for programmatic use.

Protocol summary (see https://docs.60db.ai/websocket-api/tts):
  - Auth via ?apiKey=... query param on the socket URL.
  - Server emits {"connection_established": {...}} after auth.
  - Client sends create_context (context_id, voice_id, audio_config, tuning).
  - Server replies context_created.
  - Client sends send_text, then flush_context to trigger synthesis.
  - Server streams audio_chunk messages ({audioContent: <base64>}), then
    flush_completed when the flushed text is fully synthesized.
  - Client sends close_context; server replies context_closed and closes.

Audio is 16-bit signed little-endian PCM (LINEAR16), mono, at the requested
sample rate (8k/16k/24k/48k). Chunks concatenate directly.
"""

import io
import json
import sys
import time
import uuid
import wave
from typing import List, Optional

try:
    import websocket  # websocket-client
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: websocket-client is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    print("\nThis is a required dependency. Please install it:", file=sys.stderr)
    print("  pip install websocket-client>=1.6.0", file=sys.stderr)
    raise

try:
    import requests
except (ImportError, ModuleNotFoundError) as e:
    print("ERROR: requests is not available in this Python environment.", file=sys.stderr)
    print(f"ImportError: {e}", file=sys.stderr)
    raise


DEFAULT_WS_URL = 'wss://api.60db.ai/ws/tts'
DEFAULT_VOICES_URL = 'https://api.60db.ai/myvoices'
VALID_SAMPLE_RATES = (8000, 16000, 24000, 48000)


class SixtyDbTTSError(Exception):
    """Raised on a 60db TTS synthesis or API failure."""


def list_my_voices(api_key: str, url: str = DEFAULT_VOICES_URL, timeout: float = 30.0) -> List[dict]:
    """
    Return the caller's available 60db voices.

    Args:
        api_key: 60db API key (sent as 'Authorization: Bearer ...').

    Returns:
        A list of voice dicts (voice_id, name, category, model, labels, ...).
        Empty list if the account has no voices.
    """
    resp = requests.get(
        url,
        headers={'Authorization': f'Bearer {api_key}'},
        timeout=timeout,
    )
    if resp.status_code != 200:
        raise SixtyDbTTSError(f'Voices request failed (HTTP {resp.status_code}): {resp.text[:200]}')

    try:
        payload = resp.json()
    except ValueError as e:
        raise SixtyDbTTSError(f'Voices response was not valid JSON: {e}')

    if isinstance(payload, dict):
        if payload.get('success') is False:
            raise SixtyDbTTSError(payload.get('message', 'Voices request unsuccessful'))
        data = payload.get('data', [])
    else:
        data = payload
    return data if isinstance(data, list) else []


class SixtyDbTTSClient:
    """One-shot / reusable synchronous client for 60db WebSocket TTS."""

    def __init__(self, api_key: str, ws_url: str = DEFAULT_WS_URL):
        self.api_key = api_key
        self.ws_url = ws_url

    def _auth_url(self) -> str:
        sep = '&' if ('?' in self.ws_url) else '?'
        return f'{self.ws_url}{sep}apiKey={self.api_key}'

    def synthesize(
        self,
        text: str,
        voice_id: str,
        sample_rate: int = 24000,
        speed: float = 1.0,
        stability: float = 50,
        similarity: float = 75,
        timeout: float = 60.0,
    ) -> bytes:
        """
        Synthesize `text` with `voice_id` and return raw PCM16 (mono) bytes.

        Use pcm16_to_wav() to wrap the result in a playable WAV container.
        """
        if not text or not text.strip():
            raise SixtyDbTTSError('text is required')
        if not voice_id:
            raise SixtyDbTTSError('voice_id is required')
        if sample_rate not in VALID_SAMPLE_RATES:
            raise SixtyDbTTSError(
                f'sample_rate must be one of {VALID_SAMPLE_RATES}, got {sample_rate}'
            )

        context_id = uuid.uuid4().hex
        ws = websocket.create_connection(self._auth_url(), timeout=timeout)
        audio = bytearray()
        deadline = time.time() + timeout

        try:
            # 1) Open a synthesis context.
            ws.send(json.dumps({
                'type': 'create_context',
                'context_id': context_id,
                'voice_id': voice_id,
                'audio_config': {
                    'audio_encoding': 'LINEAR16',
                    'sample_rate_hertz': sample_rate,
                },
                'speed': speed,
                'stability': stability,
                'similarity': similarity,
            }))

            # 2) Send the text and flush to trigger synthesis.
            ws.send(json.dumps({
                'type': 'send_text',
                'context_id': context_id,
                'text': text,
            }))
            ws.send(json.dumps({
                'type': 'flush_context',
                'context_id': context_id,
            }))

            # 3) Collect audio_chunk messages until flush completes.
            flushed = False
            while time.time() < deadline:
                remaining = max(0.1, deadline - time.time())
                ws.settimeout(remaining)
                try:
                    raw = ws.recv()
                except websocket.WebSocketTimeoutException:
                    break
                if raw is None or raw == '':
                    continue
                if isinstance(raw, bytes):
                    # Defensive: a server that streams binary frames sends raw PCM.
                    audio.extend(raw)
                    continue

                msg = self._parse(raw)
                if msg is None:
                    continue

                mtype = msg.get('type') or self._implicit_type(msg)
                if mtype == 'audio_chunk':
                    chunk_b64 = msg.get('audioContent') or msg.get('audio') or ''
                    if chunk_b64:
                        import base64
                        audio.extend(base64.b64decode(chunk_b64))
                elif mtype == 'flush_completed':
                    flushed = True
                    break
                elif mtype == 'error':
                    raise SixtyDbTTSError(msg.get('message', 'Unknown TTS error'))

            if not flushed and not audio:
                raise SixtyDbTTSError('No audio received before timeout')

            # 4) Close the context cleanly.
            try:
                ws.send(json.dumps({'type': 'close_context', 'context_id': context_id}))
            except Exception:
                pass

            return bytes(audio)

        finally:
            try:
                ws.close()
            except Exception:
                pass

    @staticmethod
    def _parse(raw) -> Optional[dict]:
        try:
            obj = json.loads(raw)
            return obj if isinstance(obj, dict) else None
        except (json.JSONDecodeError, TypeError):
            return None

    @staticmethod
    def _implicit_type(msg: dict) -> str:
        """Map unkeyed server frames (e.g. {"context_created": {...}}) to a type."""
        for key in ('audio_chunk', 'flush_completed', 'context_created', 'context_closed', 'error'):
            if key in msg:
                return key
        return ''


def pcm16_to_wav(pcm_bytes: bytes, sample_rate: int = 24000, channels: int = 1) -> bytes:
    """Wrap raw 16-bit little-endian PCM in a WAV container."""
    buf = io.BytesIO()
    with wave.open(buf, 'wb') as wf:
        wf.setnchannels(channels)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)
    return buf.getvalue()
