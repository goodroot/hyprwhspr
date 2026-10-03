# REST Endpoint Failover Design

## Goal

Allow the REST transcription backend to try a private primary endpoint and a
public fallback endpoint without adding another daemon or exposing credentials
in the main configuration file.

## Configuration

- Keep `rest_endpoint_url` as the primary endpoint for compatibility.
- Add `rest_fallback_endpoint_urls`, an ordered array of additional HTTP(S)
  transcription endpoints. Its default is an empty array.
- Continue using `rest_api_provider` and the credential manager. One Bearer
  credential applies to every endpoint in the ordered attempt list.
- Configure this machine with:
  - Primary: `https://speech.banjo-capella.ts.net/v1/audio/transcriptions`
  - Fallback: `https://speech.cdslash.com/v1/audio/transcriptions`
  - Provider: `custom`

## Request Flow

The backend converts captured audio to WAV and prepares headers and multipart
form data once. It then attempts the primary URL followed by configured fallback
URLs in order.

An attempt advances to the next endpoint for:

- connection errors;
- request timeouts;
- HTTP 429 responses; and
- HTTP 5xx responses.

HTTP 4xx responses other than 429 stop immediately because fallback should not
mask invalid authentication or malformed requests. A successful HTTP 200 response
is parsed exactly as it is today. Invalid JSON or an unsupported success payload
also stops immediately because another endpoint should not hide an API contract
error.

Logs identify each attempted endpoint and explain when failover occurs, without
logging credentials.

## Validation

- Unit tests cover primary success, connection failure fallback, timeout
  fallback, 429 fallback, 5xx fallback, non-retryable 4xx behavior, and complete
  exhaustion.
- Configuration defaults and JSON schema remain synchronized.
- Documentation describes the new setting and retry policy.
- The focused REST tests and complete upstream test suite pass.
- On this machine, the user service starts successfully with the new primary and
  fallback configuration. Endpoint authentication is validated after the user
  adds the shared API key to the owner-only credential store.

## Desktop Integration

No binding changes are required. The existing Logitech speech key remains bound
through Hyprland as `F14`, `XF86Launch5`, and `XF86ScreenSaver`; pressing it once
starts recording and pressing it again stops and transcribes.
