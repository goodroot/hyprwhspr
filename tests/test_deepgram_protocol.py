"""Deepgram protocol client without hardware or network."""

import json
import sys
import threading
import types
import unittest
from pathlib import Path
from urllib.parse import parse_qs, urlsplit

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


class FakeSocket:
    """Answers Finalize and CloseStream the way Deepgram does, unless scripted otherwise."""

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
                self.on_close(self, step, 'closed')
            else:
                self.event(step)

    def control_messages(self):
        return [json.loads(payload)['type'] for opcode, payload in self.sent if opcode == 1]

    def close(self):
        self.closed.set()
        self.on_close(self, 1000, '')


class DeepgramClientTests(unittest.TestCase):
    def client(self, script=(), language=None, url=URL, key='secret', finalize_script=None):
        attributes = {'script': script}
        if finalize_script is not None:
            attributes['finalize_script'] = finalize_script
        fake = type('Fake', (FakeSocket,), attributes)
        client = DeepgramRealtimeClient()
        client._websocket_transport = types.SimpleNamespace(WebSocketApp=fake)
        client.update_language(language)
        self.addCleanup(client.close)
        self.assertTrue(client.connect(url, key, 'nova-3'))
        return client

    def query(self, client):
        return {key: values[0] for key, values in parse_qs(urlsplit(client.ws.url).query).items()}

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
        self.assertEqual((query['smart_format'], query['punctuate']), ('true', 'true'))
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

    def test_closestream_waits_for_the_finalized_tail(self):
        client = self.client(script=({'type': 'Metadata'},), finalize_script=())
        client.FINALIZE_WAIT_SECONDS = 5
        ws = client.ws
        self.speak(client)
        text = []
        commit = threading.Thread(target=lambda: text.append(client.commit_and_get_text(3)))
        commit.start()
        self.assertTrue(ws.finalize_sent.wait(2))
        self.assertEqual(ws.control_messages(), ['Finalize'])
        ws.event(results('la fin.', from_finalize=True))
        commit.join(2)
        self.assertEqual(text, ['la fin.'])
        self.assertEqual(ws.control_messages(), ['Finalize', 'CloseStream'])

    def test_unanswered_finalize_still_closes_the_stream(self):
        client = self.client(script=(results('Bonjour.'), {'type': 'Metadata'}), finalize_script=())
        client.FINALIZE_WAIT_SECONDS = 0.01
        ws = client.ws
        self.speak(client)
        self.assertEqual(client.commit_and_get_text(2), 'Bonjour.')
        self.assertEqual(ws.control_messages(), ['Finalize', 'CloseStream'])

    def test_stream_lost_while_finalizing_sends_no_closestream(self):
        client = self.client(finalize_script=(1011,))
        client.FINALIZE_WAIT_SECONDS = 5
        ws = client.ws
        self.speak(client)
        sender = client._sender_thread
        with self.assertRaisesRegex(RuntimeError, '1011'):
            client.commit_and_get_text(2)
        sender.join(2)
        self.assertFalse(sender.is_alive())
        self.assertEqual(ws.control_messages(), ['Finalize'])

    def test_error_event_fails_the_recording(self):
        client = self.client()
        self.speak(client)
        client.ws.event({'type': 'Error', 'description': 'bad audio'})
        with self.assertRaisesRegex(RuntimeError, 'bad audio'):
            client.commit_and_get_text(2)

    def test_idle_timeout_is_not_a_failure(self):
        for error_first in (True, False):
            client = self.client()
            ws = client.ws
            if error_first:
                ws.on_error(ws, 'code 1011')
            ws.on_close(ws, 1011, 'NET-0001')
            self.assertFalse(client.connected)
            self.assertIsNone(client._failure)
            self.assertFalse(client.busy)

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
