"""
REST API transcription backend.

Posts recorded audio as WAV to a user-configured HTTP endpoint (a local
parakeet server or any OpenAI-compatible transcription API) and returns
the transcript. Stateless: configuration is re-read on every request.
"""

import time
from typing import Optional

try:
    from ..dependencies import require_package
except ImportError:
    from dependencies import require_package

np = require_package('numpy')
requests = require_package('requests')

try:
    from ..credential_manager import get_credential
except ImportError:
    from credential_manager import get_credential

from .base import TranscriptionBackend


class RestApiBackend(TranscriptionBackend):
    """Remote REST endpoint backend (no local model, no model lock)."""

    name = 'rest-api'
    is_local = False

    def initialize(self) -> bool:
        """Configure REST API backend"""
        # Attempt migration of API key if needed (backup in case config was loaded before migration)
        self.config.migrate_api_key_to_credential_manager()

        # Validate REST configuration
        endpoint_url = self.config.get_setting('rest_endpoint_url')

        if not endpoint_url:
            print('ERROR: REST backend selected but rest_endpoint_url not configured')
            return False

        if not endpoint_url.startswith('https://') and not endpoint_url.startswith('http://'):
            print(f'WARNING: REST endpoint URL should start with https:// or http://: {endpoint_url}')

        # Validate timeout is reasonable
        timeout = self.config.get_setting('rest_timeout', 30)
        if timeout < 1 or timeout > 300:
            print(f'WARNING: REST timeout should be between 1-300 seconds, got {timeout}')

        print(f'[BACKEND] Using REST API: {endpoint_url}')
        print(f'[REST] Timeout configured: {timeout}s')

        # Log user-defined config objects (sanitized)
        rest_headers = self.config.get_setting('rest_headers', {})
        rest_body = self.config.get_setting('rest_body', {})

        # Retrieve API key: prefer credential manager, fall back to config for backward compatibility
        api_key = None
        provider_id = self.config.get_setting('rest_api_provider')
        if provider_id:
            api_key = get_credential(provider_id)
            if api_key:
                print(f'[REST] API key configured (via credential manager, provider: {provider_id})')
            else:
                print(f'WARNING: [REST] Provider {provider_id} configured but API key not found in credential store')
        else:
            # Backward compatibility: check for old rest_api_key in config
            api_key = self.config.get_setting('rest_api_key')
            if api_key:
                print('[REST] API key configured (via rest_api_key - deprecated, consider migrating)')

        if rest_headers and isinstance(rest_headers, dict):
            header_count = len([k for k in rest_headers.keys() if rest_headers.get(k) is not None])
            if header_count > 0:
                print(f'[REST] Custom headers configured ({header_count} keys)')

        if rest_body and isinstance(rest_body, dict):
            body_count = len([k for k in rest_body.keys() if rest_body.get(k) is not None])
            if body_count > 0:
                print(f'[REST] Custom body fields configured ({body_count} fields)')

        language = self.config.get_setting('language', None)
        if language:
            print(f'[REST] Language hint: {language}')

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
        try:

            # Get REST endpoint configuration
            endpoint_url = self.config.get_setting('rest_endpoint_url')
            
            # Retrieve API key: prefer credential manager, fall back to config for backward compatibility
            api_key = None
            provider_id = self.config.get_setting('rest_api_provider')
            if provider_id:
                api_key = get_credential(provider_id)
                if not api_key:
                    print(f'WARNING: [REST] Provider {provider_id} configured but API key not found in credential store')
            else:
                # Backward compatibility: check for old rest_api_key in config
                api_key = self.config.get_setting('rest_api_key')
            
            timeout = self.config.get_setting('rest_timeout', 30)
            rest_headers = self.config.get_setting('rest_headers', {})
            rest_body = self.config.get_setting('rest_body', {})

            if not isinstance(rest_headers, dict):
                print('WARNING: rest_headers must be an object/dict; ignoring invalid value')
                rest_headers = {}

            if not isinstance(rest_body, dict):
                print('WARNING: rest_body must be an object/dict; ignoring invalid value')
                rest_body = {}

            extra_headers = {}
            for key, value in rest_headers.items():
                if value is None:
                    continue
                try:
                    extra_headers[str(key)] = str(value)
                except Exception:
                    print(f'WARNING: Skipping non-serializable rest_headers entry: {key}')

            extra_body = {}
            for key, value in rest_body.items():
                if value is None:
                    continue
                try:
                    key_str = str(key)
                except Exception:
                    print(f'WARNING: Skipping rest_body entry with non-stringable key: {key}')
                    continue

                if isinstance(value, (dict, list, tuple, set)):
                    print(f'WARNING: rest_body values must be scalar (key: {key_str}); skipping entry')
                    continue

                extra_body[key_str] = value

            if not endpoint_url:
                raise ValueError('REST endpoint URL not configured')

            # Detect backend type and extract model info
            # Note: We need to check rest_body before it's processed into extra_body
            # to get the model info early for logging
            backend_name = None
            model_info = None
            
            # Check if this is parakeet backend
            if endpoint_url in ('http://127.0.0.1:8080/transcribe', 'http://localhost:8080/transcribe'):
                backend_name = 'parakeet-tdt-0.6b-v3'
            else:
                # Generic REST API - use endpoint URL
                backend_name = endpoint_url
            
            # Extract model information from rest_body if available (before processing)
            if isinstance(rest_body, dict):
                model_info = rest_body.get('model')
            
            # Format the log message
            if backend_name == 'parakeet-tdt-0.6b-v3':
                log_msg = f'[REST API] {backend_name}'
            elif model_info:
                log_msg = f'[REST API] {backend_name} - model: {model_info}'
            else:
                log_msg = f'[REST API] {backend_name}'
            
            print(log_msg, flush=True)

            # Convert audio to WAV format
            wav_bytes = self._numpy_to_wav_bytes(audio_data, sample_rate)
            audio_duration = len(audio_data) / sample_rate
            print(
                f'[REST] Audio: {audio_duration:.2f}s @ {sample_rate}Hz, {len(wav_bytes)} bytes',
                flush=True,
            )

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
                whisper_prompt = None
                if language:
                    whisper_prompt = self.config.get_setting(f'whisper_prompt_{language}', None)
                if not whisper_prompt:
                    whisper_prompt = self.config.get_setting('whisper_prompt', None)
                if whisper_prompt:
                    data['prompt'] = whisper_prompt

            # Log request parameters for debugging
            if data:
                # Sanitize - don't log full prompt, just keys
                param_summary = ', '.join(f'{k}={v[:20] + "..." if isinstance(v, str) and len(v) > 20 else v}' for k, v in data.items())
                print(f'[REST] Request params: {param_summary}', flush=True)

            # Send the request
            print(f'[REST] Sending request to {endpoint_url}...', flush=True)
            start_time = time.time()
            response = requests.post(endpoint_url, files=files, data=data, headers=headers, timeout=timeout)
            response_time = time.time() - start_time
            print(f'[REST] Response received in {response_time:.2f}s (status: {response.status_code})', flush=True)

            # Check for HTTP errors
            if response.status_code != 200:
                error_msg = f'REST API returned status {response.status_code}'
                try:
                    error_detail = response.json()
                    error_msg += f': {error_detail}'
                except Exception:
                    error_msg += f': {response.text[:200]}'
                print(f'ERROR: {error_msg}')
                return ''

            # Parse the response
            try:
                result = response.json()
            except Exception as json_err:
                # Show raw response for debugging
                raw_body = response.text[:500] if response.text else '(empty)'
                print(f'ERROR: Failed to parse JSON response: {json_err}')
                print(f'[REST] Raw response body: {raw_body}')
                print(f'[REST] Content-Type: {response.headers.get("Content-Type", "not set")}')
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
                print(f'ERROR: Unexpected response format: {result}')
                return ''

            print(
                f'[REST] Transcription received ({len(transcription)} chars)',
                flush=True,
            )
            return transcription.strip()

        except requests.exceptions.Timeout:
            print(f'ERROR: REST API request timed out after {timeout}s')
            return ''
        except requests.exceptions.ConnectionError as e:
            print(f'ERROR: Failed to connect to REST API: {e}')
            return ''
        except requests.exceptions.RequestException as e:
            print(f'ERROR: REST API request failed: {e}')
            return ''
        except Exception as e:
            print(f'ERROR: REST transcription failed: {e}')
            return ''
