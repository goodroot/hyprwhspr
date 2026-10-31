"""Deepgram protocol client without hardware or network."""

import json
import struct
import sys
import threading
import time
import types
import unittest
from pathlib import Path
from urllib.parse import parse_qs, parse_qsl, urlsplit

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'lib' / 'src'))
sys.modules.setdefault('websocket', types.SimpleNamespace(WebSocketApp=object))

from deepgram_realtime_client import DeepgramRealtimeClient  # noqa: E402
from provider_registry import get_models_for_backend  # noqa: E402
from realtime_protocols import live_text_mode, protocol_for_provider, resolve_protocol  # noqa: E402

URL = 'wss://api.deepgram.com/v1/listen'


def results(text, is_final=True, from_finalize=False):
    return {'type': 'Results', 'is_final': is_final, 'from_finalize': from_finalize,
            'channel': {'alternatives': [{'transcript': text}]}}


class ConvertedCloseError(Exception):
    """websocket-client >= 1.9.2's on_error argument for a server close frame."""

    def __init__(self, status_code, reason):
        super().__init__(f'Connection closed (code {status_code}): {reason}')
        self.status_code = status_code
        self.reason = reason


class BadStatusError(Exception):
    """websocket-client's handshake failure: a status_code, but no close reason."""

    def __init__(self, status_code):
        super().__init__(f'Handshake status {status_code} Unauthorized')
        self.status_code = status_code


def close_frame(code, reason=''):
    """websocket-client 1.9.0/1.9.1's on_error argument: the raw ABNF close frame."""
    return types.SimpleNamespace(opcode=0x8, data=struct.pack('!H', code) + reason.encode())


class FakeSocket:
    """Answers Finalize and CloseStream the way Deepgram does, unless scripted otherwise.

    An int step is a server close frame, reported the way the configured
    websocket-client version does (both checked against the real library):
    1.9.2 calls on_error(converted exception), then on_close(code, reason);
    1.9.0 calls on_error(raw close frame), then on_close(None, None).
    """

    library = '1.9.2'

    script = ()
    finalize_script = (results('', from_finalize=True),)

    def __init__(self, url, on_open, on_message, on_error, on_close, **kwargs):
        self.url = url
        self.headers = kwargs.get('header', [])
        self.on_open = on_open
        self.on_message = on_message
        self.on_error = on_error
        self.on_close = on_close
        self.sent = []
        self.closed = threading.Event()
        self.audio_sent = threading.Event()
        self.finalize_sent = threading.Event()

    def run_forever(self):
        self.on_open(self)
        self.closed.wait(5)

    def event(self, event):
        self.on_message(self, json.dumps(event))

    def send(self, payload, opcode=1):
        self.sent.append((opcode, payload))
        if opcode == 2:
            self.audio_sent.set()
        elif json.loads(payload) == {'type': 'Finalize'}:
            self.finalize_sent.set()
            self.play(self.finalize_script)
        elif json.loads(payload) == {'type': 'CloseStream'}:
            self.play(self.script)

    def play(self, steps):
        for step in steps:
            if isinstance(step, int):
                self.server_close(step)
            else:
                self.event(step)

    def server_close(self, code, reason='bye'):
        if self.library == '1.9.0':
            self.on_error(self, close_frame(code, reason))
            self.on_close(self, None, None)
        else:
            self.on_error(self, ConvertedCloseError(code, reason))
            self.on_close(self, code, reason)

    def control_messages(self):
        return [json.loads(payload)['type'] for opcode, payload in self.sent if opcode == 1]

    def close(self):
        self.closed.set()
        self.on_close(self, 1000, '')


class DeepgramClientTests(unittest.TestCase):
    def client(self, script=(), language=None, url=URL, key='secret', finalize_script=None,
               keepalive_seconds=None, library='1.9.2'):
        attributes = {'script': script, 'library': library}
        if finalize_script is not None:
            attributes['finalize_script'] = finalize_script
        fake = type('Fake', (FakeSocket,), attributes)
        client = DeepgramRealtimeClient()
        if keepalive_seconds is not None:
            client.KEEPALIVE_SECONDS = keepalive_seconds
        client._websocket_transport = types.SimpleNamespace(WebSocketApp=fake)
        client.update_language(language)
        self.addCleanup(client.close)
        self.assertTrue(client.connect(url, key, 'nova-3'))
        return client

    def query(self, client):
        return {key: values[0] for key, values in parse_qs(urlsplit(client.ws.url).query).items()}

    def query_pairs(self, client):
        return parse_qsl(urlsplit(client.ws.url).query, keep_blank_values=True)

    def speak(self, client):
        client.append_audio(np.zeros(1600, dtype=np.float32))
        self.assertTrue(client.ws.audio_sent.wait(2))

    def test_connection_carries_session_options_and_token_auth(self):
        client = self.client(language='fr')
        self.assertEqual(client.ws.headers, ['Authorization: Token secret'])
        query = self.query(client)
        self.assertEqual(query['model'], 'nova-3')
        self.assertEqual(query['language'], 'fr')
        self.assertEqual(query['encoding'], 'linear16')
        self.assertEqual(query['sample_rate'], '16000')
        self.assertEqual(query['interim_results'], 'true')
        self.assertEqual(query['smart_format'], 'true')
        self.assertNotIn('endpointing', query)

    def test_unset_language_streams_multilingual(self):
        query = self.query(self.client())
        self.assertEqual(query['language'], 'multi')
        self.assertEqual(query['endpointing'], '100')

    def test_url_query_overrides_defaults_and_keyless_sends_no_header(self):
        client = self.client(url=f'{URL}?smart_format=false&tag=me', key=None)
        query = self.query(client)
        self.assertEqual((query['smart_format'], query['tag']), ('false', 'me'))
        self.assertEqual(client.ws.headers, [])
        self.assertEqual([key for key, _ in self.query_pairs(client)].count('smart_format'), 1)

    def test_url_query_keeps_repeated_and_blank_params(self):
        client = self.client(url=f'{URL}?keyterm=Hyprland&keyterm=Wayland&tag=')
        pairs = self.query_pairs(client)
        self.assertEqual([value for key, value in pairs if key == 'keyterm'], ['Hyprland', 'Wayland'])
        self.assertIn(('tag', ''), pairs)

    def test_final_results_join_and_metadata_completes(self):
        client = self.client(script=(results('la fin.'), results(''), {'type': 'Metadata'}))
        self.speak(client)
        client.ws.event(results('bonjour', is_final=False))
        client.ws.event(results('Bonjour tout le monde.'))
        self.assertEqual(client.commit_and_get_text(2), 'Bonjour tout le monde. la fin.')

    def test_normal_close_without_metadata_still_completes(self):
        client = self.client(script=(results('Bonjour.'), 1000))
        self.speak(client)
        self.assertEqual(client.commit_and_get_text(2), 'Bonjour.')

    def test_metadata_before_stop_does_not_end_the_recording(self):
        client = self.client()
        client.ws.event({'type': 'Metadata'})
        self.assertFalse(client.response_complete)

    def test_previews_report_interim_text_after_committed_text(self):
        client = self.client(script=({'type': 'Metadata'},))
        previews = []
        client.set_partial_transcript_callback(previews.append)
        self.speak(client)
        client.ws.event(results('Bonjour.'))
        client.ws.event(results('comment', is_final=False))
        self.assertEqual(previews[-1], 'Bonjour. comment')

    def test_stop_sends_finalize_then_closestream_without_waiting(self):
        # No from_finalize reply: Deepgram skips it when nothing is pending
        client = self.client(script=(results('Bonjour.'), {'type': 'Metadata'}), finalize_script=())
        ws = client.ws
        self.speak(client)
        self.assertEqual(client.commit_and_get_text(0.5), 'Bonjour.')
        self.assertEqual(ws.control_messages(), ['Finalize', 'CloseStream'])

    def test_normal_close_completes_on_every_supported_library(self):
        for library in ('1.9.0', '1.9.2'):
            with self.subTest(library=library):
                client = self.client(script=(results('Bonjour.'), 1000), library=library)
                self.speak(client)
                self.assertEqual(client.commit_and_get_text(2), 'Bonjour.')

    def test_abnormal_close_while_ending_fails_with_its_code(self):
        for library in ('1.9.0', '1.9.2'):
            with self.subTest(library=library):
                client = self.client(script=(results('Bonjour.'), 1011), library=library)
                self.speak(client)
                with self.assertRaisesRegex(RuntimeError, r'\(1011\): bye'):
                    client.commit_and_get_text(2)

    def test_handshake_failure_is_not_mistaken_for_a_close(self):
        client = self.client()
        ws = client.ws
        ws.on_error(ws, BadStatusError(401))
        self.assertIn('401 Unauthorized', client._failure)

    def test_transport_error_while_ending_fails(self):
        client = self.client(script=(results('Bonjour.'),))
        ws = client.ws
        self.speak(client)
        ws.on_error(ws, ConnectionResetError('reset by peer'))
        with self.assertRaisesRegex(RuntimeError, 'reset by peer'):
            client.commit_and_get_text(2)

    def test_idle_stream_is_kept_alive(self):
        client = self.client(keepalive_seconds=0.02)
        ws = client.ws
        deadline = time.monotonic() + 2
        while 'KeepAlive' not in ws.control_messages() and time.monotonic() < deadline:
            time.sleep(0.01)
        self.assertIn('KeepAlive', ws.control_messages())
        self.assertTrue(client.connected)
        self.assertFalse(client.needs_fresh_session)

    def test_no_keepalive_while_audio_flows(self):
        client = self.client(keepalive_seconds=60)
        self.speak(client)
        self.assertNotIn('KeepAlive', client.ws.control_messages())

    def test_error_event_fails_the_recording(self):
        client = self.client()
        self.speak(client)
        client.ws.event({'type': 'Error', 'description': 'bad audio'})
        with self.assertRaisesRegex(RuntimeError, 'bad audio'):
            client.commit_and_get_text(2)

    def test_dropped_idle_stream_is_replaced_before_the_next_recording(self):
        for library in ('1.9.0', '1.9.2'):
            with self.subTest(library=library):
                client = self.client(library=library)
                client.ws.server_close(1011, 'NET-0001')
                self.assertFalse(client.connected)
                self.assertTrue(client.needs_fresh_session)
                self.assertFalse(client.busy)
                self.assertIn('(1011): NET-0001', client._failure)

    def test_language_change_replaces_an_unused_stream(self):
        client = self.client()
        client.update_language('fr')
        self.assertTrue(client.needs_fresh_session)
        client.reset()
        self.assertTrue(client.connect(URL, 'secret', 'nova-3'))
        self.assertEqual(self.query(client)['language'], 'fr')
        self.assertFalse(client.needs_fresh_session)

    def test_same_language_keeps_the_stream(self):
        client = self.client(language='fr')
        ws = client.ws
        client.update_language('fr')
        self.assertFalse(client.needs_fresh_session)
        self.assertIs(client.ws, ws)

    def test_language_change_mid_recording_leaves_the_stream_alone(self):
        client = self.client(script=(results('Bonjour.'), {'type': 'Metadata'}))
        self.speak(client)
        client.update_language('fr')
        self.assertTrue(client.connected)
        self.assertEqual(client.commit_and_get_text(2), 'Bonjour.')

    def test_used_stream_needs_a_fresh_connection(self):
        client = self.client()
        self.assertFalse(client.needs_fresh_session)
        self.speak(client)
        self.assertTrue(client.needs_fresh_session)
        with self.assertRaisesRegex(RuntimeError, 'Deepgram requires a fresh connection'):
            client.clear_audio_buffer()


class DeepgramRegistryTests(unittest.TestCase):
    def test_provider_speaks_the_deepgram_protocol(self):
        self.assertEqual(protocol_for_provider('deepgram').id, 'deepgram')
        self.assertEqual(resolve_protocol('custom', 'deepgram').id, 'deepgram')
        self.assertTrue(protocol_for_provider('deepgram').per_recording_session)

    def test_nova_3_is_offered_for_realtime_only(self):
        self.assertEqual(list(get_models_for_backend('deepgram', 'realtime-ws')), ['nova-3'])
        self.assertEqual(list(get_models_for_backend('deepgram', 'rest-api')), [])

    def test_interim_text_may_be_revised(self):
        self.assertEqual(live_text_mode('deepgram', 'nova-3'), 'revisable')


if __name__ == '__main__':
    unittest.main()
