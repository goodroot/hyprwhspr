"""
REST API transcription backend.

Posts recorded audio as WAV to a user-configured HTTP endpoint (any
OpenAI-compatible transcription API) and returns the transcript.
Stateless: configuration is re-read on every request.
"""

import time
from typing import Optional
from urllib.parse import urlsplit, urlunsplit

try:
    from ..service_log import log
except ImportError:
    from service_log import log

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')

try:
    from ..credential_manager import get_credential
except ImportError:
    from credential_manager import get_credential

from .base import TranscriptionBackend


def _redact_endpoint(url):
    """Return a log-safe endpoint URL.

    Keeps the scheme, host, port, and path so the target stays identifiable
    while dropping embedded userinfo (which may carry credentials) and the
    query/fragment (which may carry tokens). The request itself always uses
    the original, unmodified URL.
    """
    if not isinstance(url, str) or not url:
        return url
    try:
        parts = urlsplit(url)
        hostname = parts.hostname
    except ValueError:
        # Unparseable (e.g. a malformed IPv6 literal): keep nothing that
        # could carry a secret.
        return '<redacted-endpoint>'
    # A log-safe endpoint needs a real, non-empty authority. Without one the
    # "path" may actually be userinfo/query data (e.g. "https:/user:pw@host/x"
    # or a scheme-less "user:pw@host/x"), so echo nothing at all.
    if not parts.netloc or not hostname:
        return '<redacted-endpoint>'
    netloc = parts.netloc
    if '@' in netloc:
        # rsplit keeps the real authority even if userinfo contains '@'.
        netloc = netloc.rsplit('@', 1)[1]
    return urlunsplit((parts.scheme, netloc, parts.path, '', ''))


class RestApiBackend(TranscriptionBackend):
    """Remote REST endpoint backend (no local model, no model lock)."""

    name = 'rest-api'
    is_local = False

    def __init__(self, manager):
        super().__init__(manager)
        self._requests = None

    def _requests_client(self):
        """Load the REST-only dependency when this backend is actually used."""
        if self._requests is None:
            self._requests = require_package('requests')
        return self._requests

    def initialize(self) -> bool:
        """Configure REST API backend"""
        self._requests_client()

        # Attempt migration of API key if needed (backup in case config was loaded before migration)
        self.config.migrate_api_key_to_credential_manager()

        # Validate REST configuration
        endpoint_url = self.config.get_setting('rest_endpoint_url')

        if not endpoint_url:
            log('ERROR: REST backend selected but rest_endpoint_url not configured')
            return False

        if not endpoint_url.startswith('https://') and not endpoint_url.startswith('http://'):
            log(f'WARNING: REST endpoint URL should start with https:// or http://: {_redact_endpoint(endpoint_url)}')

        # Validate timeout is reasonable
        timeout = self.config.get_setting('rest_timeout', 30)
        if timeout < 1 or timeout > 300:
            log(f'WARNING: REST timeout should be between 1-300 seconds, got {timeout}')

        log(f'[BACKEND] Using REST API: {_redact_endpoint(endpoint_url)}')
        log(f'[REST] Timeout configured: {timeout}s')

        # Log user-defined config objects (sanitized)
        rest_headers = self.config.get_setting('rest_headers', {})
        rest_body = self.config.get_setting('rest_body', {})

        # Retrieve API key: prefer credential manager, fall back to config for backward compatibility
        api_key = None
        provider_id = self.config.get_setting('rest_api_provider')
        if provider_id:
            api_key = get_credential(provider_id)
            if api_key:
                log(f'[REST] API key configured (via credential manager, provider: {provider_id})')
            else:
                log(f'WARNING: [REST] Provider {provider_id} configured but API key not found in credential store')
        else:
            # Backward compatibility: check for old rest_api_key in config
            api_key = self.config.get_setting('rest_api_key')
            if api_key:
                log('[REST] API key configured (via rest_api_key - deprecated, consider migrating)')

        if rest_headers and isinstance(rest_headers, dict):
            header_count = len([k for k in rest_headers.keys() if rest_headers.get(k) is not None])
            if header_count > 0:
                log(f'[REST] Custom headers configured ({header_count} keys)')

        if rest_body and isinstance(rest_body, dict):
            body_count = len([k for k in rest_body.keys() if rest_body.get(k) is not None])
            if body_count > 0:
                log(f'[REST] Custom body fields configured ({body_count} fields)')

        language = self.config.get_setting('language', None)
        if language:
            log(f'[REST] Language hint: {language}')

        # Explicitly set to None to avoid confusion with top-level model setting
        self.current_model = None
        self.ready = True
        return True

    def transcribe(self, audio_data: np.ndarray, sample_rate: int = 16000, language_override: Optional[str] = None) -> str:
        """
        Transcribe audio using remote REST API endpoint

        Args:
            audio_data: NumPy array of audio samples (float32)
            sample_rate: Sample rate of the audio data
            language_override: Optional language code to override config language

        Returns:
            Transcribed text string
        """
        requests = self._requests_client()
        try:
            # Tracks the endpoint being attempted so outer failure handlers can
            # name it safely; stays None if an error precedes the attempt loop.
            request_url = None

            # Get REST endpoint configuration
            endpoint_url = self.config.get_setting('rest_endpoint_url')
            
            # Retrieve API key: prefer credential manager, fall back to config for backward compatibility
            api_key = None
            provider_id = self.config.get_setting('rest_api_provider')
            if provider_id:
                api_key = get_credential(provider_id)
                if not api_key:
                    log(f'WARNING: [REST] Provider {provider_id} configured but API key not found in credential store')
            else:
                # Backward compatibility: check for old rest_api_key in config
                api_key = self.config.get_setting('rest_api_key')
            
            timeout = self.config.get_setting('rest_timeout', 30)
            rest_headers = self.config.get_setting('rest_headers', {})
            rest_body = self.config.get_setting('rest_body', {})

            if not isinstance(rest_headers, dict):
                log('WARNING: rest_headers must be an object/dict; ignoring invalid value')
                rest_headers = {}

            if not isinstance(rest_body, dict):
                log('WARNING: rest_body must be an object/dict; ignoring invalid value')
                rest_body = {}

            extra_headers = {}
            for key, value in rest_headers.items():
                if value is None:
                    continue
                try:
                    extra_headers[str(key)] = str(value)
                except Exception:
                    log(f'WARNING: Skipping non-serializable rest_headers entry: {key}')

            extra_body = {}
            for key, value in rest_body.items():
                if value is None:
                    continue
                try:
                    key_str = str(key)
                except Exception:
                    log(f'WARNING: Skipping rest_body entry with non-stringable key: {key}')
                    continue

                if isinstance(value, (dict, list, tuple, set)):
                    log(f'WARNING: rest_body values must be scalar (key: {key_str}); skipping entry')
                    continue

                extra_body[key_str] = value

            if not endpoint_url:
                raise ValueError('REST endpoint URL not configured')

            # Build the ordered attempt list: normalized primary first, then
            # configured fallbacks. Only valid HTTP(S) URLs are kept; non-string,
            # empty, non-HTTP(S), and duplicate values are dropped after
            # normalization so the same endpoint is never tried twice. Warnings
            # never include the value, which could embed credentials.
            primary_endpoint_url = endpoint_url.strip() if isinstance(endpoint_url, str) else endpoint_url
            endpoint_urls = [primary_endpoint_url]
            fallback_urls = self.config.get_setting('rest_fallback_endpoint_urls', [])
            if not isinstance(fallback_urls, list):
                log('WARNING: rest_fallback_endpoint_urls must be an array; ignoring invalid value')
                fallback_urls = []
            for index, fallback_url in enumerate(fallback_urls):
                if not isinstance(fallback_url, str):
                    log(f'WARNING: [REST] Ignoring non-string rest_fallback_endpoint_urls entry at index {index}')
                    continue
                normalized = fallback_url.strip()
                if not normalized:
                    log(f'WARNING: [REST] Ignoring empty rest_fallback_endpoint_urls entry at index {index}')
                    continue
                if not (normalized.startswith('http://') or normalized.startswith('https://')):
                    log(f'WARNING: [REST] Ignoring non-HTTP(S) rest_fallback_endpoint_urls entry at index {index}')
                    continue
                if normalized in endpoint_urls:
                    log(f'WARNING: [REST] Ignoring duplicate rest_fallback_endpoint_urls entry at index {index}')
                    continue
                endpoint_urls.append(normalized)

            # Extract model information from rest_body if available (before processing)
            model_info = rest_body.get('model') if isinstance(rest_body, dict) else None

            if model_info:
                log_msg = f'[REST API] {_redact_endpoint(endpoint_url)} - model: {model_info}'
            else:
                log_msg = f'[REST API] {_redact_endpoint(endpoint_url)}'

            log(log_msg)

            # Convert audio to WAV format
            wav_bytes = self._numpy_to_wav_bytes(audio_data, sample_rate)
            audio_duration = len(audio_data) / sample_rate
            log(f'[REST] Audio: {audio_duration:.2f}s @ {sample_rate}Hz, {len(wav_bytes)} bytes')

            # Prepare the request
            files = {'file': ('audio.wav', wav_bytes, 'audio/wav')}

            headers = {'Accept': 'application/json'}
            headers.update(extra_headers)
            if api_key:
                header_names = {key.lower() for key in headers.keys()}
                if 'authorization' not in header_names:
                    headers['Authorization'] = f'Bearer {api_key}'

            # Add language parameter if configured
            data = extra_body.copy()
            # Use language_override if provided, otherwise get from config
            language = language_override if language_override is not None else self.config.get_setting('language', None)
            if language and 'language' not in data:
                data['language'] = language

            # Fill prompt from config - use language-specific prompt if available
            if 'prompt' not in data:
                whisper_prompt, _ = self.resolve_whisper_prompt(language)
                if whisper_prompt:
                    data['prompt'] = whisper_prompt

            # Log request parameters for debugging
            if data:
                # Sanitize - don't log full prompt, just keys
                param_summary = ', '.join(f'{k}={v[:20] + "..." if isinstance(v, str) and len(v) > 20 else v}' for k, v in data.items())
                log(f'[REST] Request params: {param_summary}')

            # Send the request, advancing to the next endpoint only for
            # transient transport failures, HTTP 429, or HTTP 5xx. Audio,
            # headers, and form data are prepared once and reused verbatim.
            for index, request_url in enumerate(endpoint_urls):
                has_fallback = index + 1 < len(endpoint_urls)
                try:
                    log(f'[REST] Sending request to {_redact_endpoint(request_url)}...')
                    start_time = time.time()
                    response = requests.post(request_url, files=files, data=data, headers=headers, timeout=timeout)
                    response_time = time.time() - start_time
                    log(f'[REST] Response received in {response_time:.2f}s (status: {response.status_code})')
                except (requests.exceptions.Timeout, requests.exceptions.ConnectionError) as exc:
                    if has_fallback:
                        # Reason only, never the exception text or credentials.
                        log(f'WARNING: [REST] {type(exc).__name__} contacting {_redact_endpoint(request_url)}; trying fallback endpoint')
                        continue
                    # Exhausted: reuse the existing transport failure logging.
                    raise

                # Check for HTTP errors
                if response.status_code != 200:
                    retryable = response.status_code == 429 or response.status_code >= 500
                    if retryable and has_fallback:
                        log(f'WARNING: [REST] Status {response.status_code} from {_redact_endpoint(request_url)}; trying fallback endpoint')
                        # Release the discarded response body before the retry.
                        response.close()
                        continue
                    error_msg = f'REST API returned status {response.status_code}'
                    try:
                        error_detail = response.json()
                        error_msg += f': {error_detail}'
                    except Exception:
                        error_msg += f': {response.text[:200]}'
                    log(f'ERROR: {error_msg}')
                    return ''

                # Parse the response
                try:
                    result = response.json()
                except Exception as json_err:
                    # Show raw response for debugging
                    raw_body = response.text[:500] if response.text else '(empty)'
                    log(f'ERROR: Failed to parse JSON response: {json_err}')
                    log(f'[REST] Raw response body: {raw_body}')
                    log(f'[REST] Content-Type: {response.headers.get("Content-Type", "not set")}')
                    return ''

                # Try common response formats
                transcription = ''
                if 'text' in result:
                    transcription = result['text']
                elif 'transcription' in result:
                    transcription = result['transcription']
                elif 'result' in result:
                    transcription = result['result']
                else:
                    log(f'ERROR: Unexpected response format: {result}')
                    return ''

                log(f'[REST] Transcription received ({len(transcription)} chars)')
                return transcription.strip()

        # Failure logs below are deliberately type-only by privacy decision:
        # arbitrary exception text can carry the request URL, query string,
        # userinfo, or response/payload details, so only the exception class
        # and a redacted endpoint (or "<unknown>") are emitted.
        except requests.exceptions.Timeout:
            # Keep the configured duration; the exception text is never used.
            log(f'ERROR: REST API request timed out after {timeout}s')
            return ''
        except requests.exceptions.ConnectionError as exc:
            # Exception text can embed the full request URL (query/userinfo);
            # log only the type and the redacted endpoint.
            log(f'ERROR: Failed to connect to REST API ({type(exc).__name__}) at '
                f'{_redact_endpoint(request_url) or "<unknown>"}')
            return ''
        except requests.exceptions.RequestException as exc:
            log(f'ERROR: REST API request failed ({type(exc).__name__}) at '
                f'{_redact_endpoint(request_url) or "<unknown>"}')
            return ''
        except Exception as exc:
            log(f'ERROR: REST transcription failed ({type(exc).__name__}) at '
                f'{_redact_endpoint(request_url) or "<unknown>"}')
            return ''
