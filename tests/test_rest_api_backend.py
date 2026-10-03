"""REST backend ordered endpoint failover: retry policy and request reuse.

These tests are fully isolated: ``requests`` is imported only for its real
exception classes and every HTTP call is replaced with a mock. Nothing here
touches the network, the user's config, or the credential store.
"""
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends.rest_api_backend import RestApiBackend  # noqa: E402

PRIMARY = "https://primary.test/v1/audio/transcriptions"
FALLBACK = "https://fallback.test/v1/audio/transcriptions"


class Config:
    def __init__(self, **settings):
        self.settings = {
            "rest_endpoint_url": PRIMARY,
            "rest_fallback_endpoint_urls": [FALLBACK],
            "rest_timeout": 30,
            "rest_headers": {},
            "rest_body": {},
            "rest_api_provider": None,
            "rest_api_key": None,
            "language": None,
            "whisper_prompt": None,
            **settings,
        }

    def get_setting(self, key, default=None):
        return self.settings.get(key, default)

    def migrate_api_key_to_credential_manager(self):
        return False


class Manager:
    def __init__(self, config):
        self.config = config
        self.ready = False
        self.current_model = None
        self._last_use_time = 0


def response(status=200, payload=None, text="error"):
    result = mock.Mock(status_code=status, text=text, headers={})
    result.json.return_value = payload if payload is not None else {"text": "hello"}
    return result


def attempted_urls(post):
    return [call.args[0] for call in post.call_args_list]


class RestApiFailoverTests(unittest.TestCase):
    def backend(self, **settings):
        backend = RestApiBackend(Manager(Config(**settings)))
        backend._requests = requests
        backend._numpy_to_wav_bytes = mock.Mock(return_value=b"wav-bytes")
        return backend

    def transcribe(self, backend):
        # 10 ms of silence; the WAV conversion itself is mocked per test.
        return backend.transcribe(np.zeros(160, dtype=np.float32))

    # -- primary path --------------------------------------------------

    def test_primary_success_does_not_call_fallback(self):
        backend = self.backend()
        with mock.patch.object(requests, "post", return_value=response()) as post:
            self.assertEqual(self.transcribe(backend), "hello")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(attempted_urls(post), [PRIMARY])

    # -- transport fallback --------------------------------------------

    def test_connection_error_and_timeout_use_fallback(self):
        for failure in (requests.ConnectionError("down"), requests.Timeout("slow")):
            with self.subTest(failure=type(failure).__name__):
                backend = self.backend()
                with mock.patch.object(requests, "post", side_effect=[failure, response()]) as post:
                    self.assertEqual(self.transcribe(backend), "hello")
                self.assertEqual(attempted_urls(post), [PRIMARY, FALLBACK])

    def test_other_request_exception_is_not_retried(self):
        # A malformed URL / TLS config error is a RequestException but is not a
        # transient transport failure and must not be masked by a fallback.
        for failure in (requests.exceptions.HTTPError("bad"), requests.exceptions.InvalidURL("bad")):
            with self.subTest(failure=type(failure).__name__):
                backend = self.backend()
                with mock.patch.object(requests, "post", side_effect=failure) as post:
                    self.assertEqual(self.transcribe(backend), "")
                self.assertEqual(attempted_urls(post), [PRIMARY])

    # -- HTTP status fallback ------------------------------------------

    def test_retryable_http_status_uses_fallback(self):
        for status in (429, 500, 503):
            with self.subTest(status=status):
                backend = self.backend()
                with mock.patch.object(requests, "post", side_effect=[response(status), response()]) as post:
                    self.assertEqual(self.transcribe(backend), "hello")
                self.assertEqual(attempted_urls(post), [PRIMARY, FALLBACK])

    def test_non_retryable_http_status_stops(self):
        for status in (400, 401, 403, 404):
            with self.subTest(status=status):
                backend = self.backend()
                with mock.patch.object(requests, "post", return_value=response(status)) as post:
                    self.assertEqual(self.transcribe(backend), "")
                self.assertEqual(attempted_urls(post), [PRIMARY])

    # -- success payload handling --------------------------------------

    def test_invalid_json_does_not_fail_over(self):
        bad = response()
        bad.json.side_effect = ValueError("not json")
        backend = self.backend()
        with mock.patch.object(requests, "post", return_value=bad) as post:
            self.assertEqual(self.transcribe(backend), "")
        self.assertEqual(attempted_urls(post), [PRIMARY])

    def test_unsupported_success_payload_does_not_fail_over(self):
        backend = self.backend()
        with mock.patch.object(requests, "post", return_value=response(payload={"unexpected": 1})) as post:
            self.assertEqual(self.transcribe(backend), "")
        self.assertEqual(attempted_urls(post), [PRIMARY])

    def test_fallback_parses_common_success_payloads(self):
        # Every format the backend accepts must parse from a fallback response,
        # not just from the primary.
        for key in ("text", "transcription", "result"):
            with self.subTest(key=key):
                backend = self.backend()
                with mock.patch.object(
                        requests, "post",
                        side_effect=[requests.Timeout("slow"), response(payload={key: "hi"})]) as post:
                    self.assertEqual(self.transcribe(backend), "hi")
                self.assertEqual(attempted_urls(post), [PRIMARY, FALLBACK])

    # -- fallback configuration ----------------------------------------

    def test_malformed_fallback_setting_is_ignored_with_warning(self):
        backend = self.backend(rest_fallback_endpoint_urls="not-a-list")
        with mock.patch.object(requests, "post", side_effect=requests.ConnectionError("down")) as post:
            with mock.patch("backends.rest_api_backend.log") as logger:
                self.assertEqual(self.transcribe(backend), "")
        self.assertEqual(attempted_urls(post), [PRIMARY])
        messages = [str(call.args[0]) for call in logger.call_args_list if call.args]
        self.assertTrue(any("rest_fallback_endpoint_urls" in m and "ignoring" in m for m in messages),
                        messages)

    def test_padded_primary_is_normalized_and_matching_fallback_deduplicated(self):
        padded_primary = "  " + PRIMARY + "  "
        backend = self.backend(
            rest_endpoint_url=padded_primary,
            rest_fallback_endpoint_urls=[PRIMARY, FALLBACK],
        )
        with mock.patch.object(requests, "post",
                               side_effect=[requests.Timeout("slow"), response()]) as post:
            self.assertEqual(self.transcribe(backend), "hello")
        # The padded primary is attempted stripped, and the clean fallback that
        # matches it is dropped instead of being tried twice.
        self.assertEqual(attempted_urls(post), [PRIMARY, FALLBACK])

    def test_fallback_list_skips_invalid_and_normalizes_valid_urls(self):
        other = "https://other.test/v1/audio/transcriptions"
        padded = "  https://padded.test/v1/audio/transcriptions  "
        backend = self.backend(
            rest_api_key="super-secret-key",
            rest_fallback_endpoint_urls=[
                "  " + PRIMARY + "  ",   # duplicate of primary after strip -> skipped
                "",                      # empty -> skipped
                "   ",                   # whitespace only -> skipped
                None,                    # non-string -> skipped
                123,                     # non-string -> skipped
                "ftp://files.test/x",    # non-HTTP scheme -> skipped
                "primary.test/x",        # missing scheme -> skipped
                FALLBACK,                # kept
                FALLBACK,                # duplicate -> skipped
                padded,                  # stripped and kept
                other,                   # kept
            ],
        )
        with mock.patch("backends.rest_api_backend.log") as logger:
            with mock.patch.object(requests, "post",
                                   side_effect=[requests.ConnectionError("down"), response(429),
                                                response(429), response()]) as post:
                self.assertEqual(self.transcribe(backend), "hello")
        self.assertEqual(attempted_urls(post), [PRIMARY, FALLBACK, padded.strip(), other])
        warnings = [str(c.args[0]) for c in logger.call_args_list
                    if c.args and "WARNING" in str(c.args[0])]
        joined = "\n".join(str(c.args[0]) for c in logger.call_args_list if c.args)
        self.assertTrue(any("non-string" in w for w in warnings), warnings)
        self.assertTrue(any("empty" in w for w in warnings), warnings)
        self.assertTrue(any("non-HTTP" in w for w in warnings), warnings)
        self.assertNotIn("super-secret-key", joined)

    def test_duplicate_fallback_entry_is_ignored_with_indexed_warning(self):
        backend = self.backend(rest_fallback_endpoint_urls=[FALLBACK, FALLBACK])
        with mock.patch("backends.rest_api_backend.log") as logger:
            with mock.patch.object(requests, "post",
                                   side_effect=[requests.ConnectionError("down"), response()]) as post:
                self.assertEqual(self.transcribe(backend), "hello")
        self.assertEqual(attempted_urls(post), [PRIMARY, FALLBACK])
        messages = [str(c.args[0]) for c in logger.call_args_list if c.args]
        duplicate = [m for m in messages if "duplicate" in m]
        self.assertTrue(duplicate, messages)
        self.assertTrue(any("index 1" in m for m in duplicate), duplicate)
        # Credential-safe: the ignored URL value is never echoed.
        self.assertFalse(any(FALLBACK in m for m in duplicate), duplicate)

    def test_retryable_http_status_closes_response_before_failover(self):
        rejected = response(503)
        backend = self.backend()
        with mock.patch.object(requests, "post", side_effect=[rejected, response()]) as post:
            self.assertEqual(self.transcribe(backend), "hello")
        rejected.close.assert_called_once()
        self.assertEqual(attempted_urls(post), [PRIMARY, FALLBACK])

    # -- exhaustion ----------------------------------------------------

    def test_exhausted_endpoints_return_empty_transcript(self):
        backend = self.backend()
        with mock.patch.object(requests, "post", side_effect=[requests.Timeout("slow"), response(503)]) as post:
            self.assertEqual(self.transcribe(backend), "")
        self.assertEqual(attempted_urls(post), [PRIMARY, FALLBACK])

    def test_exhausted_transport_errors_return_empty_transcript(self):
        backend = self.backend()
        with mock.patch.object(requests, "post",
                               side_effect=[requests.ConnectionError("a"), requests.ConnectionError("b")]) as post:
            self.assertEqual(self.transcribe(backend), "")
        self.assertEqual(attempted_urls(post), [PRIMARY, FALLBACK])

    # -- preparation and content reuse ---------------------------------

    def test_request_is_prepared_once_and_reused_across_attempts(self):
        backend = self.backend()
        with mock.patch.object(requests, "post", side_effect=[requests.Timeout("slow"), response()]) as post:
            self.transcribe(backend)
        backend._numpy_to_wav_bytes.assert_called_once()
        first, second = post.call_args_list
        self.assertIs(first.kwargs["files"], second.kwargs["files"])
        self.assertIs(first.kwargs["data"], second.kwargs["data"])
        self.assertIs(first.kwargs["headers"], second.kwargs["headers"])

    def test_auth_and_body_are_reused_across_attempts(self):
        backend = self.backend(
            rest_api_key="super-secret-key",
            rest_headers={"X-Custom": "yes"},
            rest_body={"model": "whisper-1"},
            language="en",
            whisper_prompt="preserve this prompt",
        )
        with mock.patch.object(requests, "post", side_effect=[requests.Timeout("slow"), response()]) as post:
            self.assertEqual(self.transcribe(backend), "hello")
        first, second = post.call_args_list
        for call in (first, second):
            self.assertEqual(call.kwargs["headers"]["Authorization"], "Bearer super-secret-key")
            self.assertEqual(call.kwargs["headers"]["X-Custom"], "yes")
            self.assertEqual(call.kwargs["data"]["model"], "whisper-1")
            self.assertEqual(call.kwargs["data"]["language"], "en")
            self.assertEqual(call.kwargs["data"]["prompt"], "preserve this prompt")

    # -- logging -------------------------------------------------------

    def test_failover_logs_reason_and_never_credentials(self):
        # The reason must be explicit for both a transport exception type and an
        # HTTP status, and no attempt may leak the credential.
        scenarios = {
            "exception": ([requests.Timeout("slow"), response()], "Timeout"),
            "http-status": ([response(503), response()], "503"),
        }
        for name, (side_effect, reason) in scenarios.items():
            with self.subTest(scenario=name):
                backend = self.backend(rest_api_key="super-secret-key")
                with mock.patch.object(requests, "post", side_effect=side_effect):
                    with mock.patch("backends.rest_api_backend.log") as logger:
                        self.transcribe(backend)
                joined = "\n".join(str(call.args[0]) for call in logger.call_args_list if call.args)
                self.assertIn(PRIMARY, joined)
                self.assertIn(FALLBACK, joined)
                self.assertIn(reason, joined)
                self.assertIn("trying fallback endpoint", joined)
                self.assertNotIn("super-secret-key", joined)


    # -- endpoint redaction --------------------------------------------

    def test_redact_endpoint_strips_userinfo_query_and_fragment(self):
        from backends.rest_api_backend import _redact_endpoint
        raw = ("https://alice:s3cr3t-pw@api.example.com:8443"
               "/v1/audio/transcriptions?api_key=query-secret#frag")
        redacted = _redact_endpoint(raw)
        self.assertIn("api.example.com:8443", redacted)
        self.assertIn("/v1/audio/transcriptions", redacted)
        for secret in ("alice", "s3cr3t-pw", "query-secret", "api_key", "frag"):
            self.assertNotIn(secret, redacted)

    def test_redact_endpoint_placeholders_urls_without_authority(self):
        # Anything without a valid non-empty host has no safe authority to
        # name: the "path" may actually be userinfo/query, so refuse to echo.
        from backends.rest_api_backend import _redact_endpoint
        cases = (
            "https:/alice:pw@host/path",
            "https:///alice:pw@host/path",
            "/alice:pw@host/path",
            "user:pw@host/path?token=query-secret",
            "https://alice:pw@/path?token=query-secret",
            "no-scheme-host/path?token=query-secret",
            "env:MY_TOKEN",
            "not a url",
        )
        for raw in cases:
            with self.subTest(raw=raw):
                self.assertEqual(_redact_endpoint(raw), "<redacted-endpoint>", raw)

    def test_redact_endpoint_keeps_well_formed_authority_without_secrets(self):
        from backends.rest_api_backend import _redact_endpoint
        cases = {
            "userinfo-with-at": (
                "https://alice:p@ss@host.test:9443/v1/audio/transcriptions?token=q#frag-secret",
                ("host.test:9443", "/v1/audio/transcriptions"),
                ("alice", "p@ss", "token=", "frag-secret"),
            ),
            "fragment-only": (
                "https://host.test/path#frag-secret",
                ("host.test", "/path"),
                ("frag-secret",),
            ),
            "plain-userinfo": (
                "https://user:pass@host.test/path?x=secret",
                ("host.test", "/path"),
                ("user", "pass", "secret"),
            ),
        }
        for name, (raw, present, absent) in cases.items():
            with self.subTest(name=name):
                redacted = _redact_endpoint(raw)
                for token in present:
                    self.assertIn(token, redacted)
                for token in absent:
                    self.assertNotIn(token, redacted)

    def test_endpoint_logs_redact_userinfo_and_query_secrets(self):
        primary = ("https://alice:s3cr3t-pw@primary.test"
                   "/v1/audio/transcriptions?api_key=query-secret#frag")
        fallback = ("https://bob:hunter2@fallback.test"
                    "/v1/audio/transcriptions?token=fb-secret")
        scenarios = {
            "transport": [requests.Timeout("slow"), response()],
            "http-status": [response(503), response()],
        }
        for name, side_effect in scenarios.items():
            with self.subTest(scenario=name):
                backend = self.backend(rest_endpoint_url=primary,
                                       rest_fallback_endpoint_urls=[fallback])
                with mock.patch.object(requests, "post", side_effect=side_effect) as post:
                    with mock.patch("backends.rest_api_backend.log") as logger:
                        self.assertEqual(self.transcribe(backend), "hello")
                # The request still uses the exact configured URLs.
                self.assertEqual(attempted_urls(post), [primary, fallback])
                joined = "\n".join(str(c.args[0]) for c in logger.call_args_list if c.args)
                self.assertIn("primary.test", joined)
                self.assertIn("fallback.test", joined)
                self.assertIn("/v1/audio/transcriptions", joined)
                for secret in ("alice", "s3cr3t-pw", "query-secret", "frag",
                               "bob", "hunter2", "fb-secret", "token="):
                    self.assertNotIn(secret, joined, secret)


    # -- initialization and exhausted-handler redaction -----------------

    def test_initialize_logs_redact_userinfo_and_query(self):
        primary = ("https://alice:s3cr3t-pw@primary.test"
                   "/v1/audio/transcriptions?api_key=query-secret#frag")
        backend = self.backend(rest_endpoint_url=primary)
        with mock.patch("backends.rest_api_backend.log") as logger:
            self.assertTrue(backend.initialize())
        joined = "\n".join(str(c.args[0]) for c in logger.call_args_list if c.args)
        self.assertIn("primary.test", joined)
        self.assertIn("/v1/audio/transcriptions", joined)
        for secret in ("alice", "s3cr3t-pw", "query-secret", "api_key", "frag"):
            self.assertNotIn(secret, joined, secret)

    def test_initialize_warning_placeholders_authority_less_endpoint(self):
        # The warning fires exactly for non-HTTP endpoints. Without a real
        # authority the path itself may be userinfo/query, so nothing but the
        # placeholder is safe to echo.
        primary = ("user:s3cr3t-pw@primary.test"
                   "/v1/audio/transcriptions?api_key=query-secret")
        backend = self.backend(rest_endpoint_url=primary)
        with mock.patch("backends.rest_api_backend.log") as logger:
            backend.initialize()
        joined = "\n".join(str(c.args[0]) for c in logger.call_args_list if c.args)
        self.assertIn("<redacted-endpoint>", joined)
        self.assertNotIn("primary.test", joined)
        for secret in ("s3cr3t-pw", "query-secret", "api_key"):
            self.assertNotIn(secret, joined, secret)

    def _exhausted_log(self, error):
        primary = "https://primary.test/v1/audio/transcriptions"
        backend = self.backend(rest_endpoint_url=primary,
                               rest_fallback_endpoint_urls=[])
        with mock.patch.object(requests, "post", side_effect=error):
            with mock.patch("backends.rest_api_backend.log") as logger:
                self.assertEqual(self.transcribe(backend), "")
        return "\n".join(str(c.args[0]) for c in logger.call_args_list if c.args)

    def test_exhausted_connection_error_logs_type_and_safe_endpoint_only(self):
        secret_url = ("https://leakuser:leak-pw@secret.test"
                      "/leak/path?token=leak-query#leak-frag")
        joined = self._exhausted_log(
            requests.ConnectionError(f"Max retries exceeded with url: {secret_url}"))
        self.assertIn("ConnectionError", joined)
        self.assertIn("primary.test", joined)
        self.assertIn("/v1/audio/transcriptions", joined)
        for secret in ("leakuser", "leak-pw", "secret.test", "leak-query",
                       "leak-frag", "token="):
            self.assertNotIn(secret, joined, secret)

    def test_exhausted_request_exception_logs_type_and_safe_endpoint_only(self):
        secret_url = ("https://leakuser:leak-pw@secret.test"
                      "/leak/path?token=leak-query#leak-frag")
        joined = self._exhausted_log(
            requests.exceptions.InvalidURL(f"bad endpoint {secret_url}"))
        self.assertIn("InvalidURL", joined)
        self.assertIn("primary.test", joined)
        self.assertIn("/v1/audio/transcriptions", joined)
        for secret in ("leakuser", "leak-pw", "secret.test", "leak-query",
                       "leak-frag", "token="):
            self.assertNotIn(secret, joined, secret)

    def test_exhausted_unexpected_exception_logs_type_and_safe_endpoint_only(self):
        secret_url = ("https://leakuser:leak-pw@secret.test"
                      "/leak/path?token=leak-query#leak-frag")
        joined = self._exhausted_log(ValueError(f"boom {secret_url}"))
        self.assertIn("ValueError", joined)
        self.assertIn("primary.test", joined)
        for secret in ("leakuser", "leak-pw", "secret.test", "leak-query",
                       "leak-frag", "token="):
            self.assertNotIn(secret, joined, secret)


if __name__ == "__main__":
    unittest.main()
