# REST Endpoint Failover Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add ordered failover to the REST transcription backend and configure this machine to prefer the private speech service while falling back to the public service.

**Architecture:** Preserve `rest_endpoint_url` as the primary URL and add an ordered `rest_fallback_endpoint_urls` list. The REST backend prepares audio, headers, and form data once, then retries only transport failures, HTTP 429, and HTTP 5xx responses against subsequent URLs; credentials and request content remain identical across attempts.

**Tech Stack:** Python 3.14, `requests`, stdlib `unittest`/`unittest.mock`, JSON Schema, systemd user services, Hyprland.

---

## File Map

- `lib/src/backends/rest_api_backend.py`: validate the fallback list and perform ordered request attempts.
- `tests/test_rest_api_backend.py`: isolated retry-policy and request-order regression tests.
- `lib/src/config_manager.py`: register the new default configuration key.
- `share/config.schema.json`: define the new array setting and URL item type.
- `docs/CONFIGURATION.md`: document configuration and retry semantics.
- `~/.config/hyprwhspr/config.json`: select this machine's primary URL, fallback URL, and `custom` credential provider.

### Task 1: REST Failover Behavior

**Files:**
- Create: `tests/test_rest_api_backend.py`
- Modify: `lib/src/backends/rest_api_backend.py:45-87,116-246`

- [x] **Step 1: Write failing backend tests**

Create an isolated test fixture with a dictionary-backed config, a minimal manager, real `requests` exception classes, and mocked responses. Cover primary success, connection failure, timeout, HTTP 429, HTTP 5xx, non-retryable HTTP 401, and exhaustion:

```python
import sys
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import requests

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "lib" / "src"))

from backends.rest_api_backend import RestApiBackend


class Config:
    def __init__(self, **settings):
        self.settings = {
            "rest_endpoint_url": "https://primary.test/v1/audio/transcriptions",
            "rest_fallback_endpoint_urls": ["https://fallback.test/v1/audio/transcriptions"],
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


def response(status=200, payload=None):
    result = mock.Mock(status_code=status, text="error", headers={})
    result.json.return_value = payload if payload is not None else {"text": "hello"}
    return result


class RestApiFailoverTests(unittest.TestCase):
    def backend(self, **settings):
        backend = RestApiBackend(Manager(Config(**settings)))
        backend._requests = requests
        backend._numpy_to_wav_bytes = mock.Mock(return_value=b"wav")
        return backend

    def transcribe(self, backend):
        return backend.transcribe(np.zeros(160, dtype=np.float32))

    def test_primary_success_does_not_call_fallback(self):
        backend = self.backend()
        with mock.patch.object(requests, "post", return_value=response()) as post:
            self.assertEqual(self.transcribe(backend), "hello")
        self.assertEqual(post.call_count, 1)
        self.assertEqual(post.call_args.args[0], "https://primary.test/v1/audio/transcriptions")

    def test_connection_error_and_timeout_use_fallback(self):
        for failure in (requests.ConnectionError("down"), requests.Timeout("slow")):
            with self.subTest(type=type(failure).__name__):
                backend = self.backend()
                with mock.patch.object(requests, "post", side_effect=[failure, response()]) as post:
                    self.assertEqual(self.transcribe(backend), "hello")
                self.assertEqual([call.args[0] for call in post.call_args_list], [
                    "https://primary.test/v1/audio/transcriptions",
                    "https://fallback.test/v1/audio/transcriptions",
                ])

    def test_retryable_http_status_uses_fallback(self):
        for status in (429, 500, 503):
            with self.subTest(status=status):
                backend = self.backend()
                with mock.patch.object(requests, "post", side_effect=[response(status), response()]) as post:
                    self.assertEqual(self.transcribe(backend), "hello")
                self.assertEqual(post.call_count, 2)

    def test_non_retryable_http_status_stops(self):
        backend = self.backend()
        with mock.patch.object(requests, "post", return_value=response(401)) as post:
            self.assertEqual(self.transcribe(backend), "")
        self.assertEqual(post.call_count, 1)

    def test_exhausted_endpoints_return_empty_transcript(self):
        backend = self.backend()
        with mock.patch.object(requests, "post", side_effect=[requests.Timeout("slow"), response(503)]) as post:
            self.assertEqual(self.transcribe(backend), "")
        self.assertEqual(post.call_count, 2)
```

- [x] **Step 2: Run tests and verify RED**

Run:

```bash
.venv/bin/python -m unittest -v tests.test_rest_api_backend
```

Expected: failover tests fail because only `rest_endpoint_url` is attempted.

- [x] **Step 3: Implement ordered failover**

In `RestApiBackend`, read and validate `rest_fallback_endpoint_urls`, discard non-string/empty/duplicate URLs, and build `[primary, *fallbacks]`. Replace the single `requests.post` call with an ordered loop. Continue only for `requests.exceptions.Timeout`, `requests.exceptions.ConnectionError`, HTTP 429, or status `>= 500`; preserve existing parsing and immediate failure behavior for all other responses and exceptions. Log the endpoint before each attempt and a credential-free failover reason before advancing.

The request loop should follow this structure:

```python
endpoint_urls = [endpoint_url]
fallback_urls = self.config.get_setting('rest_fallback_endpoint_urls', [])
if not isinstance(fallback_urls, list):
    log('WARNING: rest_fallback_endpoint_urls must be an array; ignoring invalid value')
    fallback_urls = []
for fallback_url in fallback_urls:
    if isinstance(fallback_url, str) and fallback_url and fallback_url not in endpoint_urls:
        endpoint_urls.append(fallback_url)

for index, request_url in enumerate(endpoint_urls):
    has_fallback = index + 1 < len(endpoint_urls)
    try:
        log(f'[REST] Sending request to {request_url}...')
        start_time = time.time()
        response = requests.post(request_url, files=files, data=data, headers=headers, timeout=timeout)
        response_time = time.time() - start_time
        log(f'[REST] Response received in {response_time:.2f}s (status: {response.status_code})')
    except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
        if has_fallback:
            log(f'WARNING: [REST] {type(exc).__name__}; trying fallback endpoint')
            continue
        log(f'ERROR: REST API request failed after all endpoints: {exc}')
        return ''

    if response.status_code != 200:
        retryable = response.status_code == 429 or response.status_code >= 500
        if retryable and has_fallback:
            log(f'WARNING: [REST] Status {response.status_code}; trying fallback endpoint')
            continue
        # Preserve the existing sanitized HTTP error construction and return ''.

    # Preserve the existing JSON parsing and transcript extraction, returning
    # immediately on the first HTTP 200 response.
```

- [x] **Step 4: Run focused tests and verify GREEN**

Run:

```bash
.venv/bin/python -m unittest -v tests.test_rest_api_backend
```

Expected: all REST failover tests pass.

### Task 2: Configuration Contract and Documentation

**Files:**
- Modify: `lib/src/config_manager.py:143-150`
- Modify: `share/config.schema.json:345-383`
- Modify: `docs/CONFIGURATION.md:708-726`
- Test: `tests/test_config_schema_sync.py`

- [x] **Step 1: Add the default before changing the schema**

Add beside `rest_endpoint_url`:

```python
'rest_fallback_endpoint_urls': [],  # Ordered fallback URLs for retryable REST failures
```

- [x] **Step 2: Verify the schema-sync test fails**

Run:

```bash
.venv/bin/python -m unittest -v tests.test_config_schema_sync
```

Expected: `test_every_default_has_a_schema_entry` fails and names `rest_fallback_endpoint_urls`.

- [x] **Step 3: Add the JSON Schema property**

Add immediately after `rest_endpoint_url`:

```json
"rest_fallback_endpoint_urls": {
  "type": "array",
  "items": { "type": "string", "format": "uri" },
  "default": [],
  "description": "Ordered fallback URLs used after transport errors, HTTP 429, or HTTP 5xx responses"
},
```

- [x] **Step 4: Document usage and retry behavior**

Extend the custom backend JSON example with:

```jsonc
"rest_fallback_endpoint_urls": [
    "https://fallback.example.com/v1/audio/transcriptions"
],
```

After the example, state that fallback preserves headers, body fields, audio, and the selected credential; retries occur only for connection errors, timeouts, HTTP 429, and HTTP 5xx. State that other HTTP 4xx responses and invalid successful payloads stop immediately.

- [x] **Step 5: Verify configuration and focused backend tests**

Run:

```bash
.venv/bin/python -m unittest -v tests.test_config_schema_sync tests.test_rest_api_backend
```

Expected: all tests pass.

### Task 3: Full Verification and Machine Deployment

**Files:**
- Modify: `~/.config/hyprwhspr/config.json`
- Apply tested source changes to: `/home/cd-slash/devel/hyprwhspr/`

- [x] **Step 1: Run the complete isolated suite**

Run with desktop variables removed so mocked injection tests cannot inherit the live Hyprland path:

```bash
env -u HYPRLAND_INSTANCE_SIGNATURE -u XDG_CURRENT_DESKTOP -u XDG_SESSION_DESKTOP -u DESKTOP_SESSION \
  .venv/bin/python -m unittest discover -s tests -v
```

Expected: 1,539 existing tests plus the new REST tests pass.

- [x] **Step 2: Apply the reviewed worktree diff to the installed checkout**

Apply only the intended source, test, schema, documentation, spec, and plan changes to `/home/cd-slash/devel/hyprwhspr`. Do not alter or delete the rollback branch `main`.

- [x] **Step 3: Configure this machine**

Set these values in `~/.config/hyprwhspr/config.json` while retaining all unrelated settings:

```json
"transcription_backend": "rest-api",
"rest_endpoint_url": "https://speech.banjo-capella.ts.net/v1/audio/transcriptions",
"rest_fallback_endpoint_urls": [
  "https://speech.cdslash.com/v1/audio/transcriptions"
],
"rest_api_provider": "custom"
```

Do not place the API key in this file. The credential manager reads the shared key from `~/.local/share/hyprwhspr/credentials`, whose JSON shape is:

```json
{
  "custom": "YOUR_API_KEY"
}
```

The credential file must have mode `0600`. If no key exists yet, leave credential creation to the user rather than inventing or exposing a secret.

- [x] **Step 4: Restart and verify the service**

Run:

```bash
systemctl --user restart hyprwhspr.service
systemctl --user show hyprwhspr.service -p ActiveState -p SubState -p ExecMainStatus
journalctl --user -u hyprwhspr.service --since "2 minutes ago" --no-pager
```

Expected: `ActiveState=active`, `SubState=running`, `ExecMainStatus=0`; logs show the private REST endpoint and no startup traceback.

- [x] **Step 5: Validate desktop integration**

Run:

```bash
hyprctl reload
hyprctl configerrors
omarchy menu keybindings --print
```

Expected: no Hyprland configuration errors and `F14` remains listed as `Speech-to-text`.

- [ ] **Step 6: Report credential and live-request status** — BLOCKED for authenticated validation: no `custom` key exists in `~/.local/share/hyprwhspr/credentials`, so no authenticated live request was made. Non-authenticated status reporting (installed version, service state, endpoint order, hotkey) is complete.

Report the installed version, service status, endpoint order, and hotkey. If `custom` is absent from `~/.local/share/hyprwhspr/credentials`, explicitly state that live authenticated transcription remains pending until the user adds the key; do not print any existing credential value.

No git commit or push is included because the user did not request either.
