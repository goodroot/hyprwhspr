"""Read-only, allowlisted configuration and installation diagnostics."""
import json
import math
import os
from pathlib import Path
import re
import shutil
import socket
import subprocess
import sys
import time

from config_manager import build_default_config, normalize_legacy_config, expand_env, _ENV_PATTERN
from paths import CONFIG_FILE, DATA_DIR, SOCKET_FILE
from backend_utils import normalize_backend
from dependency_plan import PLAN_SPECS, plan_key
from provider_registry import get_models_for_backend, get_realtime_capabilities, get_realtime_mode
from session_environment import classify_display_environment

ROOT = Path(os.environ.get('HYPRWHSPR_ROOT', Path(__file__).resolve().parents[2]))
SCHEMA_FILE = ROOT / 'share' / 'config.schema.json'


def finding(check_id, severity, explanation, next_step=None):
    result = dict(check_id=check_id, severity=severity, explanation=explanation)
    if next_step:
        result['next_step'] = next_step
    return result


def display_kind():
    return classify_display_environment('\n'.join(
        key + '=' + os.environ.get(key, '')
        for key in ('XDG_SESSION_TYPE', 'WAYLAND_DISPLAY', 'DISPLAY')))


def validate_config(path=None):
    findings = []
    config = build_default_config()
    try:
        raw = json.loads(Path(path or CONFIG_FILE).read_text(encoding='utf-8'))
    except FileNotFoundError:
        raw = {}
        findings.append(finding('config.missing', 'info', 'No configuration file; using defaults.'))
    except json.JSONDecodeError as exc:
        return config, [finding('config.json', 'error',
            f'Malformed JSON at line {exc.lineno}, column {exc.colno}.', 'Correct the JSON syntax in config.json.')], True
    except (OSError, UnicodeError):
        return config, [finding('config.read', 'error', 'Configuration could not be read.',
            'Check config.json permissions and UTF-8 encoding.')], True
    if not isinstance(raw, dict):
        return config, [finding('config.structure', 'error', 'Configuration must be a JSON object.')], False
    try:
        from jsonschema import Draft202012Validator
        schema = json.loads(SCHEMA_FILE.read_text(encoding='utf-8'))
    except (ImportError, OSError, ValueError):
        return config, findings + [finding('config.validator', 'error', 'Bundled schema or validator unavailable.',
            'Install jsonschema for the CLI system Python and restore the bundled schema.')], True
    known = schema['properties']
    aliases = {'push_to_talk', 'audio_device', 'shift_paste', 'rest_api_key'}
    for key in raw:
        if key in aliases:
            findings.append(finding('config.deprecated', 'warning', 'A deprecated configuration alias is present.',
                'Review legacy settings in docs/CONFIGURATION.md.'))
        elif key not in known and not re.fullmatch(r'whisper_prompt_[a-z]{2}', key):
            findings.append(finding('config.unknown', 'warning', 'An unknown configuration key is present (name redacted).'))
    # Validate legacy values before normalization so truthiness cannot hide bad types.
    if 'push_to_talk' in raw and type(raw['push_to_talk']) is not bool:
        findings.append(finding('config.alias_type', 'error', 'push_to_talk must be a boolean.'))
    if 'audio_device' in raw and not (raw['audio_device'] is None or type(raw['audio_device']) in (str, int)):
        findings.append(finding('config.alias_type', 'error', 'audio_device must be a string, integer or null.'))
    normalized, migrations = normalize_legacy_config(raw)
    if "'whisper_prompt' -> 'whisper_prompt_en'" in migrations:
        findings.append(finding('config.deprecated_prompt', 'warning',
            'An inherited whisper_prompt default is normalized to whisper_prompt_en.'))
    config.update(expand_env(normalized))
    if '$schema' in raw and not isinstance(raw['$schema'], str):
        findings.append(finding('config.schema', 'error', '$schema must be a string.'))

    def walk(value):
        if isinstance(value, dict):
            for child in value.values():
                yield from walk(child)
        elif isinstance(value, list):
            for child in value:
                yield from walk(child)
        else:
            yield value
    if any(isinstance(v, str) and any(m.group(1) not in os.environ for m in _ENV_PATTERN.finditer(v)) for v in walk(normalized)):
        findings.append(finding('config.environment', 'warning', 'Unresolved environment references are present (names and values redacted).',
            'Provide the referenced variables in the daemon environment.'))
    if any(isinstance(v, float) and not math.isfinite(v) for v in walk(config)):
        findings.append(finding('config.finite', 'error', 'Numeric settings must be finite.'))
    if isinstance(config['transcription_backend'], str):
        config['transcription_backend'] = normalize_backend(config['transcription_backend'])
    for error in Draft202012Validator(schema).iter_errors(config):
        path_parts = list(error.absolute_path)
        key = path_parts[0] if path_parts and path_parts[0] in known else '[redacted]'
        findings.append(finding('config.schema', 'error', f'Invalid setting: {key} ({error.validator}).',
            'Check the type and allowed values in docs/CONFIGURATION.md.'))
    if raw.get('transcription_backend') in ('local', 'remote', 'amd'):
        findings.append(finding('config.deprecated_backend', 'warning', 'A legacy backend name is present.'))
    if any(f['severity'] == 'error' for f in findings):
        return config, findings, False
    if config['secondary_language'] and not config['secondary_shortcut']:
        findings.append(finding('config.secondary_shortcut', 'warning', 'Secondary language has no shortcut.'))
    if config['use_hypr_bindings'] and config['grab_keys']:
        findings.append(finding('config.keyboard', 'warning', 'Compositor bindings bypass keyboard grabbing.'))
    if normalize_backend(config['transcription_backend']) == 'realtime-ws':
        provider, model = config['websocket_provider'], config['websocket_model']
        models = get_models_for_backend(provider, 'realtime-ws')
        if provider != 'custom' and model not in models:
            findings.append(finding('config.realtime_model', 'error', 'Unsupported realtime provider/model combination.'))
        elif provider != 'custom':
            caps = get_realtime_capabilities(provider, model)
            if config['realtime_mode'] != get_realtime_mode(provider, model):
                findings.append(finding('config.realtime_mode', 'error', 'Realtime mode does not match the model.'))
            if config['recording_mode'] == 'continuous' and not caps.get('continuous', False):
                findings.append(finding('config.realtime_continuous', 'error', 'This model does not support continuous recording.'))
        elif not config['websocket_url']:
            findings.append(finding('config.realtime_url', 'error', 'Custom realtime provider requires a WebSocket URL.'))
    if not findings:
        findings.append(finding('config.valid', 'info', 'Configuration checks passed.'))
    return config, findings, False


def sanitize_snapshot(value):
    """Ignore all unrecognized keys and values, including older daemon errors."""
    if not isinstance(value, dict) or value.get('format_version') != 1:
        return None
    result = {'format_version': 1, 'best_effort': True}
    allowed = {
        'recording': bool, 'processing': bool, 'backend_ready': bool,
        'model_loaded': bool, 'selection_failed': bool, 'recovering': bool,
        'backend_initializing': bool, 'backend_failed': bool,
        'microphone_index': int, 'sample_rate': (int, float),
    }
    for key, typ in allowed.items():
        item = value.get(key)
        types = typ if isinstance(typ, tuple) else (typ,)
        if type(item) in types and (type(item) not in (int, float) or math.isfinite(item)):
            result[key] = item
    for key in ('recording_state', 'backend_state', 'audio_state'):
        if value.get(key) in ('available', 'busy', 'unavailable'):
            result[key] = value[key]
    if value.get('display') in ('wayland', 'x11', 'unknown'):
        result['display'] = value['display']
    if value.get('model_state') in ('loaded', 'unloaded', 'not_applicable', 'unavailable'):
        result['model_state'] = value['model_state']
    if value.get('fallback') in ('unknown', 'active', 'inactive'):
        result['fallback'] = value['fallback']
    routes = value.get('delivery_routes')
    if isinstance(routes, list):
        result['delivery_routes'] = [r for r in ('wtype', 'ydotool', 'xdotool', 'hyprctl') if r in routes]
    return result


def daemon_snapshot(app):
    result = {'format_version': 1, 'best_effort': True, 'display': display_kind() or 'unknown', 'fallback': 'unknown'}
    def section(name, lock, read):
        if lock is None:
            result[name] = 'unavailable'
        elif not lock.acquire(blocking=False):
            result[name] = 'busy'
        else:
            try:
                result.update(read())
                result[name] = 'available'
            except Exception:
                # Partially initialized or concurrently torn-down sections must
                # not discard other live state or expose exception contents.
                result[name] = 'unavailable'
            finally:
                lock.release()
    section('recording_state', getattr(app, '_recording_lock', None), lambda: {
        'recording': app.is_recording, 'processing': app.is_processing,
        'backend_initializing': getattr(app, '_model_initializing', False),
        'backend_failed': getattr(app, '_backend_init_failed', False)})
    manager = getattr(app, 'whisper_manager', None)
    def backend_state():
        backend = manager._backend
        state = {'backend_ready': manager.ready, 'model_state': 'unavailable'}
        if backend is not None:
            if getattr(backend, 'is_local', True):
                state['model_loaded'] = bool(backend.is_loaded)
                state['model_state'] = 'loaded' if state['model_loaded'] else 'unloaded'
            else:
                state['model_state'] = 'not_applicable'
        return state
    section('backend_state', getattr(manager, '_model_lock', None), backend_state)
    audio = getattr(app, 'audio_capture', None)
    section('audio_state', getattr(audio, 'recovery_lock', None), lambda: {
        'microphone_index': getattr(audio, 'device_id', None),
        'sample_rate': audio.sample_rate,
        'recovering': audio.recovery_in_progress,
        'selection_failed': bool(audio._input_selection_error)})
    injector = getattr(app, 'text_injector', None)
    result['delivery_routes'] = [name for name in ('wtype', 'ydotool', 'xdotool')
        if getattr(injector, name + '_available', False) is True]
    if display_kind() == 'wayland' and os.environ.get('HYPRLAND_INSTANCE_SIGNATURE'):
        result['delivery_routes'].append('hyprctl')
    return sanitize_snapshot(result)


class Probes:
    def __init__(self):
        self.deadline = time.monotonic() + 10

    def timeout(self):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError()
        return min(2, remaining)

    def run(self, args):
        return subprocess.run(args, capture_output=True, text=True, timeout=self.timeout(), check=False)

    def live(self):
        deadline = time.monotonic() + self.timeout()
        with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as client:
            client.settimeout(self.timeout())
            client.connect(str(SOCKET_FILE))
            remaining = min(deadline, self.deadline) - time.monotonic()
            if remaining <= 0:
                raise TimeoutError()
            client.settimeout(remaining)
            client.sendall(b'{"verb":"diagnostics"}\n')
            data = b''
            while b'\n' not in data and len(data) <= 65536:
                remaining = min(deadline, self.deadline) - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError()
                client.settimeout(remaining)
                chunk = client.recv(4096)
                if not chunk:
                    break
                data += chunk
            response = json.loads(data)
            return sanitize_snapshot(response.get('snapshot')) if response.get('ok') is True else None


def build_report():
    probes = Probes()
    config, findings, unusable = validate_config()
    report = {'format_version': 1, 'findings': findings, 'live': None}
    def add(check, severity, message, step=None):
        findings.append(finding(check, severity, message, step))
    # No dependency imports: only top-level find_spec and distribution metadata.
    modules = ()
    if not any(f['severity'] == 'error' for f in findings):
        backend = normalize_backend(config['transcription_backend'])
        report['configured'] = {'backend': backend, 'recording_mode': config['recording_mode']}
        modules = PLAN_SPECS[plan_key(backend, config['websocket_provider'], None, ValueError)][1]
    environment = Path(os.environ['HYPRWHSPR_BACKEND_ENV']) if os.environ.get('HYPRWHSPR_BACKEND_ENV') else DATA_DIR / 'venv'
    interpreter = environment / 'bin' / 'python'
    script = "import importlib.util as u, importlib.metadata as m, json, sys; distributions = m.packages_distributions(); print(json.dumps({'python': list(sys.version_info[:3]), 'modules': {n: u.find_spec(n) is not None for n in sys.argv[1:]}, 'metadata': {n: bool(distributions.get(n)) for n in sys.argv[1:]}}))"
    try:
        output = probes.run([str(interpreter), '-c', script, *modules])
        if output.returncode != 0:
            raise ValueError('interpreter probe failed')
        data = json.loads(output.stdout)
        version = data['python']
        if not (isinstance(version, list) and len(version) == 3
                and all(type(n) is int and n >= 0 for n in version)):
            raise ValueError('invalid interpreter version')
        if not all(isinstance(data.get(key), dict) and all(
                type(data[key].get(module)) is bool for module in modules)
                for key in ('modules', 'metadata')):
            raise ValueError('invalid dependency probe response')
        report['python_version'] = '.'.join(map(str, version))
        if version < [3, 10, 0]:
            add('python.version', 'error', 'Selected interpreter is older than the required Python 3.10.',
                'Recreate the application environment with Python 3.10 or newer.')
        else:
            add('python.version', 'info', 'Selected interpreter version checked.')
        for module in modules:
            available = data['modules'].get(module) is True
            metadata = data['metadata'].get(module) is True
            add('dependency.' + module, 'info' if available else 'error',
                module + (': module discoverable' if available else ': module unavailable') +
                ('; package metadata present.' if metadata else '; package metadata absent.'),
                None if available else 'Install the selected backend dependencies.')
    except FileNotFoundError:
        add('dependencies.interpreter', 'error', 'Application virtual environment interpreter is missing.', 'Install the application environment.')
    except (subprocess.TimeoutExpired, TimeoutError):
        add('dependencies.probe', 'warning', 'Selected interpreter probe timed out.', 'Retry the report when the system is idle.')
    except (OSError, ValueError, KeyError, TypeError):
        add('dependencies.probe', 'error', 'Selected interpreter could not complete the Python and dependency checks.',
            'Repair or recreate the application virtual environment.')
    try:
        metadata = ROOT / 'release.json'
        if metadata.is_file():
            version = json.loads(metadata.read_text(encoding="utf-8"))['version']
            valid = isinstance(version, str)
        else:
            output = probes.run(['git', '-C', str(ROOT), 'describe', '--tags', '--abbrev=7'])
            version = output.stdout.strip()
            valid = output.returncode == 0
        if valid and re.fullmatch(r'v?\d+\.\d+\.\d+(?:-\d+-g[0-9a-f]+)?', version):
            report['application_version'] = version
            add('application.version', 'info', 'Application version identified.')
        else:
            add('application.version', 'warning', 'Application version unavailable.')
    except (OSError, ValueError, KeyError, TypeError, subprocess.TimeoutExpired, TimeoutError):
        add('application.version', 'warning', 'Application version probe unavailable or timed out.')
    try:
        output = probes.run(['systemctl', '--user', 'is-active', 'hyprwhspr.service'])
        state = output.stdout.strip()
        if state not in ('active', 'inactive', 'failed', 'activating', 'deactivating'):
            state = 'unavailable'
        add('service.state', 'error' if state == 'failed' else ('info' if state == 'active' else 'warning'), 'User service state: ' + state + '.')
    except (OSError, subprocess.TimeoutExpired, TimeoutError):
        add('service.state', 'warning', 'User service probe unavailable or timed out.')
    display = display_kind()
    add('desktop.session', 'info' if display else 'warning', 'Display kind: ' + (display or 'unknown') + '.')
    names = ('xclip', 'xsel', 'xdotool') if display == 'x11' else ('wl-copy', 'wtype', 'ydotool')
    if display == 'wayland' and os.environ.get('HYPRLAND_INSTANCE_SIGNATURE'):
        names += ('hyprctl',)
    available = {name: bool(shutil.which(name)) for name in names}
    for name, present in available.items():
        add('tool.' + name, 'info' if present else 'warning', name + (': executable available.' if present else ': executable absent.'))
    if not config.get('use_hypr_bindings'):
        devices = list(Path('/dev/input').glob('event*'))
        add('permissions.keyboard', 'info' if any(os.access(p, os.R_OK) for p in devices) else 'warning',
            'Keyboard input access available.' if any(os.access(p, os.R_OK) for p in devices) else 'Keyboard input access unavailable; compositor shortcuts may still work.')
    if available.get('ydotool'):
        add('permissions.uinput', 'info' if os.access('/dev/uinput', os.W_OK) else 'warning',
            'Virtual input access available.' if os.access('/dev/uinput', os.W_OK) else 'Direct virtual input access unavailable; a delivery service may provide access.')
    try:
        report['live'] = probes.live()
    except (OSError, ValueError, TypeError, AttributeError, TimeoutError):
        pass
    if report['live']:
        live = report['live']
        if live.get('selection_failed') is True:
            add('daemon.microphone', 'error', 'The daemon could not select a usable microphone.', 'Review the configured microphone selection.')
        if live.get('backend_failed') is True:
            add('daemon.backend_failed', 'error', 'The live backend failed to initialize.', 'Review the selected backend installation and configuration.')
        if live.get('backend_ready') is False:
            add('daemon.backend', 'warning', 'The live backend is not ready; it may be loading or unloaded.')
        if any(live.get(key) == 'busy' for key in ('recording_state', 'backend_state', 'audio_state')):
            add('daemon.busy', 'warning', 'Some live sections are busy; retry the report when idle.')
    add('daemon.snapshot', 'info' if report['live'] else 'warning',
        'Live snapshot is best-effort.' if report['live'] else 'Live information unavailable; daemon may be absent, older, busy or shutting down.')
    return report, unusable


def run_diagnostics(report=False, json_output=False):
    try:
        if report:
            document, unusable = build_report()
        else:
            _, findings, unusable = validate_config()
            document = {'format_version': 1, 'findings': findings}
    except Exception:
        document = {'format_version': 1, 'findings': [finding('diagnostics.unavailable', 'error', 'Diagnostics could not produce a usable result.')]}
        unusable = True
    if json_output:
        print(json.dumps(document, allow_nan=False))
    else:
        for item in document['findings']:
            print(f"{item['severity'].upper()} [{item['check_id']}] {item['explanation']}")
            if item.get('next_step'):
                print('  ' + item['next_step'])
        for key in ('application_version', 'python_version', 'configured', 'live'):
            if document.get(key) is not None:
                print(key + ': ' + json.dumps(document[key]))
    return 2 if unusable else int(any(f['severity'] == 'error' for f in document['findings']))
