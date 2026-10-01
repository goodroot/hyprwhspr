"""Phonon protocol client and backend lifecycle without hardware or network.

Ported from James Wolfley's Phonon-2 provider branch (#270).
"""

import importlib
import json
import socket
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))
sys.modules.setdefault('websocket', types.SimpleNamespace(WebSocketApp=object))

from phonon_realtime_client import PhononRealtimeClient  # noqa: E402
from backends.realtime_ws_backend import RealtimeWsBackend  # noqa: E402


def _real_websocket(test):
    """The installed websocket-client, loaded without disturbing other tests' stub."""
    with mock.patch.dict(sys.modules):
        sys.modules.pop('websocket', None)
        try:
            return importlib.import_module('websocket')
        except ImportError:
            test.skipTest('websocket-client not installed')


class FakeConfig:
    def __init__(self, **values):
        self.values = values

    def get_setting(self, key, default=None):
        return self.values.get(key, default)

    def migrate_api_key_to_credential_manager(self):
        pass


def manager(config, preview=None):
    return types.SimpleNamespace(config=config, ready=False, current_model=None,
                                 _last_use_time=0, _realtime_partial_callback=preview)


class FakeSocket:
    """Calls real transport callbacks; binary sends can be held in flight."""

    def __init__(self, url, on_open, on_message, on_error, on_close, **kwargs):
        self.url = url
        self.headers = kwargs.get('header', [])
        self.on_open = on_open
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.sent = []
        self.closed = threading.Event()
        self.binary_started = threading.Event()
        self.allow_binary = threading.Event()
        self.allow_binary.set()
        self.complete = True
        self.fail_binary = False

    def run_forever(self):
        self.on_open(self)
        self.closed.wait(5)

    def event(self, **event):
        self.on_message(self, json.dumps(event))

    def send(self, payload, opcode=1):
        if opcode == 2:
            self.binary_started.set()
            if not self.allow_binary.wait(2):
                raise TimeoutError('test sender stalled')
            if self.fail_binary:
                raise OSError('send failed')
        self.sent.append((opcode, payload))
        if opcode == 1 and json.loads(payload)['type'] == 'end' and self.complete:
            self.event(type='final', segment=1, text='Hello world.')
            self.event(type='done', text='Hello world.')
            self.on_close(self, 1000, '')

    def close(self):
        self.closed.set()
        self.on_close(self, 1000, '')


class PhononClientTests(unittest.TestCase):
    def test_busy_rejection_during_success_finalization_returns_false(self):
        client = PhononRealtimeClient()
        client._websocket_transport = types.SimpleNamespace(WebSocketApp=FakeSocket)
        self.addCleanup(client.close)
        original = client._on_connect_success
        def reject():
            client.ws.event(type='error', message='busy')
            original()
        with mock.patch.object(client, '_on_connect_success', side_effect=reject):
            self.assertFalse(client.connect('ws://asr.example.test/stream', None, 'phonon-2'))
        self.assertFalse(client.connected)

    def test_abandoned_attempt_interrupts_blocked_config_write(self):
        client = PhononRealtimeClient()
        self.addCleanup(client.close)
        entered, release = threading.Event(), threading.Event()
        ws = FakeSocket('ws://asr.example.test/stream', client._on_open,
                        client._on_message, client._on_error, client._on_close)
        ws.sock = types.SimpleNamespace(shutdown=release.set)
        original = ws.send
        def send(payload, opcode=1):
            entered.set()
            release.wait(2)
            original(payload, opcode)
        ws.send = send
        client.ws = ws
        self.addCleanup(release.set)
        worker = threading.Thread(target=client._on_open, args=(ws,))
        worker.start()
        self.assertTrue(entered.wait(1))
        client._abandon_attempt(ws)
        self.assertTrue(release.is_set())
        worker.join(1)
        self.assertFalse(worker.is_alive())
        self.assertFalse(client.connected)
        self.assertIsNone(client.ws)

    def test_connect_timeout_interrupts_real_blocked_config_write(self):
        websocket = _real_websocket(self)
        writer, reader = socket.socketpair()
        self.addCleanup(reader.close)
        self.addCleanup(writer.close)
        writer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        writer.setblocking(False)
        try:
            while True:
                writer.send(b'x' * 4096)
        except BlockingIOError:
            pass
        writer.setblocking(True)
        transport = websocket.WebSocket(enable_multithread=True)
        transport.sock = writer
        transport.connected = True
        entered = threading.Event()
        class ConfigSocket(FakeSocket):
            def __init__(self, *args, **kwargs):
                super().__init__(*args, **kwargs)
                self.sock = transport
            def send(self, payload, opcode=1):
                entered.set()
                transport.send(payload, opcode)
            def close(self):
                transport.close()
                super().close()
        client = PhononRealtimeClient()
        client._websocket_transport = types.SimpleNamespace(WebSocketApp=ConfigSocket)
        self.addCleanup(client.close)
        calls = []
        def clock():
            self.assertTrue(entered.wait(1))
            calls.append(True)
            return 0 if len(calls) == 1 else 11
        start = time.monotonic()
        # Advance only the inherited handshake clock, not real cleanup timers.
        with mock.patch.dict(client._connect_internal.__globals__, time=types.SimpleNamespace(time=clock)):
            self.assertFalse(client.connect('ws://asr.example.test/stream', None, 'phonon-2'))
        self.assertLess(time.monotonic() - start, 0.5)
        for worker in client._ws_threads:
            worker.join(1)
            self.assertFalse(worker.is_alive())

    def test_cancelled_waiter_cannot_consume_or_close_new_recording(self):
        client = self.client()
        old_ws = client.ws
        old_ws.complete = False
        entered = threading.Event()
        release = threading.Event()
        original_wait = client.response_event.wait
        def wait(timeout):
            entered.set()
            original_wait(timeout)
            release.wait(2)
            return True
        errors = []
        with mock.patch.object(client.response_event, 'wait', side_effect=wait):
            thread = threading.Thread(target=lambda: self._commit_error(client, errors))
            thread.start()
            self.assertTrue(entered.wait(1))
            client.discard_audio()
            self.assertTrue(client.connect('ws://asr.example.test/stream', None, 'phonon-2'))
            new_ws = client.ws
            release.set()
            thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertTrue(client.connected)
        self.assertFalse(new_ws.closed.is_set())

    @staticmethod
    def _commit_error(client, errors):
        try:
            client.commit_and_get_text(0.1)
        except RuntimeError as exc:
            errors.append(str(exc))

    def test_blocked_write_is_aborted_at_completion_timeout(self):
        client = self.client()
        ws = client.ws
        gate = threading.Event()
        ws.sock = types.SimpleNamespace(shutdown=gate.set)
        original_send = ws.send
        def send(payload, opcode=1):
            if opcode == 2:
                ws.binary_started.set()
                gate.wait(3)
            return original_send(payload, opcode)
        ws.send = send
        self.addCleanup(gate.set)
        client.append_audio(np.zeros(160, dtype=np.float32))
        self.assertTrue(ws.binary_started.wait(1))
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'Timeout'):
            client.commit_and_get_text(0.01)
        self.assertTrue(gate.is_set())
        self.assertLess(time.monotonic() - start, 0.5)

    def test_blocked_end_write_is_aborted_at_completion_timeout(self):
        client = self.client()
        ws = client.ws
        gate = threading.Event()
        ws.sock = types.SimpleNamespace(shutdown=gate.set)
        self.addCleanup(gate.set)
        original_send = ws.send
        def send(payload, opcode=1):
            if opcode == 1 and json.loads(payload)['type'] == 'end':
                gate.wait(2)
            return original_send(payload, opcode)
        ws.send = send
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'Timeout'):
            client.commit_and_get_text(0.01)
        self.assertTrue(gate.is_set())
        self.assertLess(time.monotonic() - start, 0.5)

    def test_real_websocket_blocked_write_is_interrupted_by_shutdown(self):
        websocket = _real_websocket(self)
        client = self.client()
        ws = client.ws
        writer, reader = socket.socketpair()
        self.addCleanup(reader.close)
        self.addCleanup(writer.close)
        writer.setsockopt(socket.SOL_SOCKET, socket.SO_SNDBUF, 4096)
        transport = websocket.WebSocket(enable_multithread=True)
        transport.sock = writer
        transport.connected = True
        ws.sock = transport
        def send(payload, opcode=1):
            ws.binary_started.set()
            transport.send(payload, opcode)
        ws.send = send
        client.set_max_buffer_seconds(60)
        client.append_audio(np.zeros(16000 * 40, dtype=np.float32))
        self.assertTrue(ws.binary_started.wait(1))
        start = time.monotonic()
        with self.assertRaisesRegex(RuntimeError, 'Timeout'):
            client.commit_and_get_text(0.01)
        self.assertLess(time.monotonic() - start, 0.5)
        self.assertFalse(client._sender_thread.is_alive())

    def test_cancel_clear_is_delivered_after_inflight_preview(self):
        client = self.client()
        ws = client.ws
        entered, release = threading.Event(), threading.Event()
        previews = []
        def callback(text):
            if text:
                entered.set()
                release.wait(2)
            previews.append(text)
        client.set_partial_transcript_callback(callback)
        event_thread = threading.Thread(target=lambda: ws.event(type='partial', text='stale'))
        event_thread.start()
        self.assertTrue(entered.wait(1))
        cancel_thread = threading.Thread(target=client.discard_audio)
        cancel_thread.start()
        cancel_thread.join(0.05)
        release.set()
        event_thread.join(2)
        cancel_thread.join(2)
        self.assertFalse(event_thread.is_alive() or cancel_thread.is_alive())
        self.assertEqual(previews[-1], '')

    def client(self, token=None):
        client = PhononRealtimeClient()
        client._websocket_transport = types.SimpleNamespace(WebSocketApp=FakeSocket)
        self.addCleanup(client.close)
        self.assertTrue(client.connect('wss://asr.example.test/v1/audio/stream', token, 'phonon-2'))
        return client

    def test_config_and_optional_bearer_header(self):
        for token in (None, 'proxy-token'):
            with self.subTest(token=token):
                client = self.client(token)
                self.assertEqual(client.ws.headers, [f'Authorization: Bearer {token}'] if token else [])
                self.assertEqual(json.loads(client.ws.sent[0][1]), {
                    'type': 'config', 'sample_rate': 16000, 'format': 'pcm_s16le',
                })

    def test_stop_drains_inflight_binary_before_end_and_returns_done_once(self):
        client = self.client()
        ws = client.ws
        ws.allow_binary.clear()
        self.addCleanup(ws.allow_binary.set)
        client.append_audio(np.array([-1, 0, 1], dtype=np.float32))
        self.assertTrue(ws.binary_started.wait(1))
        results = []
        thread = threading.Thread(target=lambda: results.append(client.commit_and_get_text(2)))
        thread.start()
        with client.lock:
            self.assertTrue(client._sending_audio)
        self.assertEqual(len(ws.sent), 1)
        ws.allow_binary.set()
        thread.join(3)
        self.assertFalse(thread.is_alive())
        self.assertEqual(results, ['Hello world.'])
        self.assertEqual([opcode for opcode, _ in ws.sent], [1, 2, 1])
        self.assertEqual(np.frombuffer(ws.sent[1][1], dtype='<i2').tolist(), [-32767, 0, 32767])
        self.assertEqual(json.loads(ws.sent[-1][1]), {'type': 'end'})
        self.assertEqual(client.commit_and_get_text(0), '')
        self.assertFalse(client.connected)

    def test_partial_replacement_and_segment_deduplication(self):
        client = self.client()
        previews = []
        client.set_partial_transcript_callback(previews.append)
        client.ws.event(type='partial', text='Hel')
        client.ws.event(type='partial', text='Hello')
        client.ws.event(type='final', segment=1, text='Hello.')
        client.ws.event(type='final', segment=1, text='Hello.')
        client.ws.event(type='partial', text='World')
        client.ws.event(type='final', segment=1, text='Hello.')
        self.assertEqual(previews, ['Hel', 'Hello', 'Hello.', 'Hello.', 'Hello. World', 'Hello. World'])

    def test_cancellation_ignores_late_results_and_wakes_waiter(self):
        client = self.client()
        ws = client.ws
        ws.complete = False
        errors = []
        def commit():
            try:
                client.commit_and_get_text(3)
            except RuntimeError as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=commit)
        thread.start()
        client.discard_audio()
        ws.event(type='done', text='Must not paste')
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertTrue(errors)
        self.assertFalse(client.response_complete)

    def test_timeout_does_not_return_segment_or_partial_fallback(self):
        client = self.client()
        client.ws.complete = False
        client.ws.event(type='final', segment=1, text='Incomplete')
        client.ws.event(type='partial', text='Provisional')
        with self.assertRaisesRegex(RuntimeError, 'Timeout'):
            client.commit_and_get_text(0.01)
        self.assertFalse(client.connected)

    def test_cancel_discards_done_that_has_not_been_consumed(self):
        client = self.client()
        client._ending = True
        client.ws.event(type='done', text='Must not paste')
        client.discard_audio()
        with self.assertRaisesRegex(RuntimeError, 'cancelled'):
            client.commit_and_get_text(0.01)

    def test_busy_idle_and_midstream_close_never_reconnect_automatically(self):
        for code in (1013, 1000, 1006):
            with self.subTest(code=code):
                client = self.client()
                with mock.patch.object(client, '_attempt_reconnect') as reconnect:
                    client.ws.on_close(client.ws, code, 'busy or idle')
                    with self.assertRaisesRegex(RuntimeError, 'before completion'):
                        client.commit_and_get_text(0.01)
                reconnect.assert_not_called()

    def test_error_and_malformed_events_fail_without_text(self):
        for event in ({'type': 'error', 'message': 'busy'},
                      {'type': 'final', 'text': 'Missing segment'},
                      {'type': 'done', 'text': 'Premature'},
                      {'type': 'partial', 'text': 42}):
            with self.subTest(event=event):
                client = self.client()
                client.ws.event(**event)
                with self.assertRaises(RuntimeError):
                    client.commit_and_get_text(0.01)

    def test_queue_overflow_fails_instead_of_dropping_audio(self):
        client = self.client()
        client.append_audio(np.zeros(16000 * 6, dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, 'overflow'):
            client.commit_and_get_text(0.01)

    def test_send_failure_discards_recording(self):
        client = self.client()
        client.ws.fail_binary = True
        client.append_audio(np.zeros(160, dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, 'send failed'):
            client.commit_and_get_text(1)

    def test_resamples_capture_rate_on_sender(self):
        client = self.client()
        client.set_input_sample_rate(48000)
        resample = mock.Mock(return_value=np.zeros(160, dtype=np.float32))
        # Some isolation tests reload realtime_base. Patch the globals actually
        # owned by this inherited method, not a potentially newer module object.
        with mock.patch.dict(client._resample_for_output.__globals__, resample_audio=resample):
            client.append_audio(np.zeros(480, dtype=np.float32))
            ws = client.ws
            self.assertEqual(client.commit_and_get_text(1), 'Hello world.')
        self.assertEqual(resample.call_args.args[1:], (48000, 16000))
        self.assertEqual(len(ws.sent[1][1]), 320)

    def test_invalid_audio_fails_instead_of_sending_wrong_format(self):
        client = self.client()
        client.append_audio(np.array([np.nan], dtype=np.float32))
        with self.assertRaisesRegex(RuntimeError, 'non-finite'):
            client.commit_and_get_text(1)

    def test_old_socket_events_cannot_change_reconnected_recording(self):
        client = self.client()
        old_ws = client.ws
        client.discard_audio()
        self.assertTrue(client.connect('ws://asr.example.test/stream', None, 'phonon-2'))
        old_ws.event(type='partial', text='stale')
        old_ws.event(type='error', message='stale failure')
        old_ws.on_close(old_ws, 1013, 'stale rejection')
        self.assertTrue(client.connected)
        self.assertEqual(client._partial_transcript, '')
        self.assertIsNone(client._failure)
        self.assertEqual(client.commit_and_get_text(1), 'Hello world.')


class PhononBackendTests(unittest.TestCase):
    def test_invalid_websocket_urls_rejected_before_connection(self):
        for url in ('', 'ws://', 'https://asr.example.test/stream', 'ws://host:bad', 'ws://[invalid', 'ws://host:0',
                    'WSS://asr.example.test/stream', '\x00wss://asr.example.test/stream',
                    'ws://asr.example.test/\x00stream', 'ws://asr.example.test/\x7fstream'):
            with self.subTest(url=url):
                backend, _ = self.backend(websocket_url=url)
                with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None), \
                     mock.patch.object(PhononRealtimeClient, 'connect') as connect:
                    self.assertFalse(backend.initialize())
                    connect.assert_not_called()

    def test_long_form_rejected_before_connection(self):
        backend, _ = self.backend(recording_mode='long_form')
        with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None), \
             mock.patch.object(PhononRealtimeClient, 'connect') as connect:
            self.assertFalse(backend.initialize())
            connect.assert_not_called()

    def test_new_callback_does_not_cancel_pending_completion(self):
        backend, _ = self.backend()
        with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None), \
             mock.patch('websocket.WebSocketApp', FakeSocket):
            self.assertTrue(backend.initialize())
            backend.get_streaming_callback()(np.zeros(160, dtype=np.float32))
            client = backend._realtime_client
            ws = client.ws
            ws.complete = False
            ended = threading.Event()
            send = ws.send
            def signal(payload, opcode=1):
                send(payload, opcode)
                if opcode == 1 and json.loads(payload)['type'] == 'end':
                    ended.set()
            ws.send = signal
            result = []
            worker = threading.Thread(target=lambda: result.append(backend.transcribe(np.zeros(160))))
            worker.start()
            self.assertTrue(ended.wait(1))
            self.assertIsNone(backend.get_streaming_callback())
            self.assertFalse(ws.closed.is_set())
            ws.event(type='done', text='Owned result')
            worker.join(2)
            self.assertFalse(worker.is_alive())
            self.assertEqual(result, ['Owned result'])

    def test_idle_close_between_ready_and_clear_recovers_before_capture(self):
        backend, _ = self.backend()
        with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None), \
             mock.patch('websocket.WebSocketApp', FakeSocket):
            self.assertTrue(backend.initialize())
            client = backend._realtime_client
            original_clear = client.clear_audio_buffer
            def clear():
                client.ws.on_close(client.ws, 1000, 'idle')
                original_clear()
            with mock.patch.object(client, 'clear_audio_buffer', side_effect=clear):
                self.assertIsNone(backend.get_streaming_callback())
            self.assertTrue(callable(backend.get_streaming_callback()))

    def test_continuous_mode_rejected_before_connection(self):
        backend, _ = self.backend(recording_mode='continuous')
        with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None), \
             mock.patch.object(PhononRealtimeClient, 'connect') as connect:
            self.assertFalse(backend.initialize())
            connect.assert_not_called()

    def backend(self, **extra):
        values = dict(transcription_backend='realtime-ws', websocket_provider='custom', websocket_protocol='phonon',
                      websocket_model='phonon-2', websocket_url='ws://asr.example.test/v1/audio/stream')
        values.update(extra)
        preview = mock.Mock()
        backend = RealtimeWsBackend(manager(FakeConfig(**values), preview))
        self.addCleanup(backend.close)
        return backend, preview

    def test_repeated_recordings_and_cancel_reconnect_without_credentials(self):
        backend, preview = self.backend()
        with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None), \
             mock.patch('websocket.WebSocketApp', FakeSocket):
            self.assertTrue(backend.initialize())
            first_ws = backend._realtime_client.ws
            for _ in range(2):
                callback = backend.get_streaming_callback()
                self.assertTrue(callable(callback))
                self.assertTrue(callable(callback.set_input_sample_rate))
                callback(np.zeros(160, dtype=np.float32))
                backend._realtime_client.ws.event(type='partial', text='Preview')
                self.assertEqual(backend.transcribe(np.zeros(160)), 'Hello world.')
            self.assertIsNot(backend._realtime_client.ws, first_ws)
            self.assertIn(mock.call('Preview'), preview.call_args_list)
            backend.get_streaming_callback()(np.zeros(160, dtype=np.float32))
            backend.discard_audio()
            self.assertFalse(backend._realtime_client.connected)
            self.assertTrue(callable(backend.get_streaming_callback()))

    def test_invalid_configuration_fails_before_connect(self):
        for settings in ({'websocket_url': None}, {'websocket_url': 'https://wrong'},
                         {'language': 'fr'}):
            with self.subTest(settings=settings):
                backend, _ = self.backend(**settings)
                with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None), \
                     mock.patch.object(PhononRealtimeClient, 'connect') as connect:
                    self.assertFalse(backend.initialize())
                    connect.assert_not_called()

    def test_busy_connection_attempt_fails_without_waiting_full_timeout(self):
        class BusySocket(FakeSocket):
            def run_forever(self):
                self.on_open(self)
                self.event(type='error', message='Another stream is active')
                self.on_close(self, 1013, 'busy')
        backend, _ = self.backend()
        with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None), \
             mock.patch('websocket.WebSocketApp', BusySocket):
            self.assertFalse(backend.initialize())
            self.assertIsNone(backend._realtime_client)

    def test_skipped_transcription_gets_new_connection_on_next_recording(self):
        backend, _ = self.backend()
        with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None), \
             mock.patch('websocket.WebSocketApp', FakeSocket):
            self.assertTrue(backend.initialize())
            callback = backend.get_streaming_callback()
            old_ws = backend._realtime_client.ws
            callback(np.zeros(160, dtype=np.float32))
            self.assertTrue(callable(backend.get_streaming_callback()))
            self.assertIsNot(backend._realtime_client.ws, old_ws)
            self.assertTrue(old_ws.closed.is_set())

    def test_non_english_override_discards_stream_and_can_recover(self):
        backend, _ = self.backend()
        with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None), \
             mock.patch('websocket.WebSocketApp', FakeSocket):
            self.assertTrue(backend.initialize())
            backend.get_streaming_callback()(np.zeros(160, dtype=np.float32))
            self.assertEqual(backend.transcribe(np.zeros(160), language_override='fr'), '')
            self.assertFalse(backend._realtime_client.connected)
            self.assertTrue(callable(backend.get_streaming_callback()))

    def test_converse_is_coerced_to_transcribe(self):
        backend, _ = self.backend(realtime_mode='converse')
        with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None),              mock.patch.object(PhononRealtimeClient, 'connect', return_value=True):
            self.assertTrue(backend.initialize())
        self.assertEqual(backend._realtime_client.mode, 'transcribe')

    def test_busy_client_blocks_the_next_start(self):
        backend, _ = self.backend()
        client = mock.Mock(busy=True)
        backend._realtime_client = client
        self.assertTrue(backend.is_busy)
        self.assertIsNone(backend.get_streaming_callback())
        self.assertEqual(backend.last_connect_failure, 'processing')

    def test_configured_sample_rate_reaches_the_config_message(self):
        backend, _ = self.backend(websocket_sample_rate=24000)
        with mock.patch('backends.realtime_ws_backend.get_credential', return_value=None),              mock.patch('websocket.WebSocketApp', FakeSocket):
            self.assertTrue(backend.initialize())
            opcode, payload = backend._realtime_client.ws.sent[0]
        self.assertEqual(json.loads(payload)['sample_rate'], 24000)


if __name__ == '__main__':
    unittest.main()
