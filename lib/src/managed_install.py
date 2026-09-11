"""Release lifecycle. This module deliberately depends only on the standard library.

Immutable payloads and virtualenvs live at their final paths. The only activation
operation is replacing current.json; the write-ahead journal owns staged garbage.
"""
import argparse
from contextlib import contextmanager
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import time
import urllib.request
import uuid

# -I ignores PYTHONDONTWRITEBYTECODE; lifecycle imports must never mutate payloads.
sys.dont_write_bytecode = True

REPOSITORY = 'goodroot/hyprwhspr'
VERSION = re.compile(r'^v?\d+\.\d+\.\d+$')
CLI_IMPORTS = ('rich', 'requests', 'jsonschema', 'evdev', 'pulsectl', 'pyudev')


def clean_env():
    env = os.environ.copy()
    for key in list(env):
        if key.startswith(('PYTHON', 'PIP_')) or key == 'VIRTUAL_ENV':
            env.pop(key, None)
    env.update(PYTHONDONTWRITEBYTECODE='1', PYTHONNOUSERSITE='1', PIP_CONFIG_FILE=os.devnull, PIP_USER='0')
    env.pop('HYPRWHSPR_INSTALL_LOCK_FD', None)
    return env


def legacy_build_env():
    """Avoid mise command shims without changing user pip sources or manager files."""
    env = os.environ.copy()
    roots = [Path.home() / '.local/share/mise',
             Path(env.get('MISE_DATA_DIR', Path.home() / '.local/share/mise'))]
    paths = [part for part in env.get('PATH', '').split(os.pathsep)
             if part and not any(Path(part).is_relative_to(root) for root in roots)]
    env['PATH'] = os.pathsep.join(paths) if paths else os.defpath
    for key in ('MISE_SHELL', '__MISE_ACTIVATE', 'MISE_DATA_DIR'):
        env.pop(key, None)
    return env


def run(command, **kwargs):
    kwargs.setdefault('env', clean_env())
    return subprocess.run([str(x) for x in command], check=True, **kwargs)


def digest(path):
    with Path(path).open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def read_json(path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding='utf-8'))
    except FileNotFoundError:
        return default


def read_state(path, default, *, on_error=None, validate=None):
    """Tolerant recovery-state read; payload/ownership verification uses read_json."""
    try:
        value = read_json(path, default)
        if not isinstance(value, type(default)):
            raise ValueError(f'expected {type(default).__name__}')
        if isinstance(default, list) and any(not isinstance(item, str) for item in value):
            raise ValueError('expected a list of paths')
        if validate is not None and not validate(value):
            raise ValueError('invalid state structure')
        return value
    except (OSError, ValueError, TypeError) as exc:
        print(f'Unusable recovery state {path}: {exc}; preserving evidence and continuing conservatively.', file=sys.stderr)
        if on_error is not None:
            on_error(Path(path))
        return default.copy()


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name('.' + path.name + '.tmp')
    with temporary.open('w', encoding='utf-8') as stream:
        json.dump(value, stream, indent=2)
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)
    fd = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def atomic_text(path, content, mode=0o644):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(fd, 'w', encoding='utf-8') as stream:
            stream.write(content)
            stream.flush()
            os.fchmod(stream.fileno(), mode)
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        Path(temporary).unlink(missing_ok=True)


class EnvironmentVerificationError(RuntimeError):
    pass


class HTTPSRedirectHandler(urllib.request.HTTPRedirectHandler):
    """Release metadata carries the archive checksum, so every hop must stay HTTPS."""
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        if not newurl.startswith('https://'):
            raise ValueError(f'Release downloads require HTTPS; refused redirect to {newurl}')
        return super().redirect_request(req, fp, code, msg, headers, newurl)


def download(url, destination, maximum=128 * 1024 * 1024):
    if not url.startswith('https://'):
        raise ValueError('Release downloads require HTTPS')
    request = urllib.request.Request(url, headers={'User-Agent': 'hyprwhspr-installer'})
    opener = urllib.request.build_opener(HTTPSRedirectHandler)
    with opener.open(request, timeout=60) as response, Path(destination).open('wb') as output:
        total = 0
        while block := response.read(65536):
            total += len(block)
            if total > maximum:
                raise ValueError('Release download exceeds size limit')
            output.write(block)


def extract(archive, destination, expected):
    if digest(archive) != expected:
        raise ValueError('Application archive checksum mismatch')
    destination = Path(destination)
    with tarfile.open(archive, 'r:gz') as bundle:
        members = bundle.getmembers()
        names = set()
        total = 0
        for member in members:
            path = PurePosixPath(member.name)
            if (path.is_absolute() or '..' in path.parts or '\\' in member.name
                    or not path.parts or path.as_posix() in names
                    or not (member.isfile() or member.isdir())):
                raise ValueError(f'Unsafe archive member: {member.name}')
            names.add(path.as_posix())
            total += member.size
            if total > 512 * 1024 * 1024 or len(members) > 20000:
                raise ValueError('Application archive exceeds extraction limit')
        destination.mkdir(parents=True)
        for member in members:
            target = destination / member.name
            if member.isdir():
                target.mkdir(parents=True, exist_ok=True)
            else:
                target.parent.mkdir(parents=True, exist_ok=True)
                with bundle.extractfile(member) as source, target.open('wb') as output:
                    shutil.copyfileobj(source, output)
                target.chmod(0o755 if member.mode & 0o111 else 0o644)
    verify_payload(destination)


def verify_payload(root):
    root = Path(root)
    metadata = read_json(root / 'release.json')
    if not metadata or metadata.get('format') != 1 or not VERSION.fullmatch(metadata.get('version', '')):
        raise ValueError('Unsupported application release metadata')
    files = metadata.get('files', {})
    required = {'bin/hyprwhspr', 'lib/cli.py', 'lib/main.py', 'lib/src/managed_install.py',
                'requirements-cli.txt', 'requirements.txt', 'share/config.schema.json'}
    if not required <= files.keys():
        raise ValueError('Incomplete application release')
    for name, checksum in files.items():
        path = PurePosixPath(name)
        if path.is_absolute() or '..' in path.parts or (root / name).is_symlink():
            raise ValueError('Unsafe payload manifest')
        if digest(root / name) != checksum:
            raise ValueError(f'Payload integrity failure: {name}')
    actual = {str(p.relative_to(root)) for p in root.rglob('*') if p.is_file()}
    if actual != set(files) | {'release.json'}:
        extra = actual - set(files) - {'release.json'}
        hint = ' Generated Python bytecode found; run install repair to replace the payload.' if any('__pycache__' in PurePosixPath(name).parts for name in extra) else ''
        raise ValueError('Unrecorded payload files.' + hint)
    return metadata


def interpreter(path=None, fallback=False):
    candidates = ([path] if path else [])
    if not path or fallback:
        candidates.extend(p for p in ('/usr/bin/python3', '/usr/local/bin/python3') if p not in candidates)
    for candidate in candidates:
        try:
            result = run([candidate, '-I', '-c',
                          'import json,sys; print(json.dumps({"path":sys._base_executable,'
                          '"version":list(sys.version_info[:3]),"identity":sys.version}))'],
                         capture_output=True, text=True, timeout=10)
            identity = json.loads(result.stdout)
            if (3, 11) <= tuple(identity['version'][:2]) <= (3, 14):
                return identity
        except (OSError, subprocess.SubprocessError, ValueError):
            pass
    raise RuntimeError('No compatible Python (3.11–3.14). Use --python PATH; keep that interpreter installed.')


def group_membership_present(user, group):
    """Check the supplementary membership that installation could have added."""
    import grp
    import pwd
    try:
        account = pwd.getpwnam(user)
        membership = grp.getgrnam(group)
    except KeyError:
        return False
    return user in membership.gr_mem or (membership.gr_gid != account.pw_gid and
        membership.gr_gid in os.getgrouplist(user, account.pw_gid))


def remove_added_group(user, group, remove):
    if not group_membership_present(user, group):
        print(f'Group membership already absent: {user} / {group}')
        return
    try:
        remove()
    except (OSError, subprocess.SubprocessError):
        if group_membership_present(user, group):
            raise
        print(f'Group membership already absent: {user} / {group}')


class CleanupError(RuntimeError):
    """Removal failed after the remaining paths were durably queued for retry."""


class CommittedCleanupError(RuntimeError):
    """Activation succeeded; only post-commit cleanup remains incomplete."""
    def __init__(self, generation, error):
        self.generation = generation
        super().__init__(f"Generation {generation['version']} is active; cleanup incomplete: {error}")


class Installation:
    def __init__(self, data=None, state=None):
        self.data = Path(data or Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share')) / 'hyprwhspr')
        self.state = Path(state or Path(os.environ.get('XDG_STATE_HOME', Path.home() / '.local/state')) / 'hyprwhspr')
        self.current = self.data / 'current.json'
        self.journal = self.state / 'transaction.json'
        self.receipt = self.state / 'ownership.json'
        self.lock_fd = None

    @contextmanager
    def lock(self):
        inherited = os.environ.get('HYPRWHSPR_INSTALL_LOCK_FD')
        if inherited:
            try:
                fd = int(inherited)
                actual = os.fstat(fd)
                expected = (self.state / 'installation.lock').stat()
                valid = (actual.st_dev, actual.st_ino) == (expected.st_dev, expected.st_ino)
            except (OSError, ValueError):
                valid = False
            if valid:
                self.lock_fd = fd
                try:
                    yield
                finally:
                    self.lock_fd = None
                return
            # Closed/stale descriptors are not evidence of owning a lock.
            # Acquire normally; a live parent owner will produce the busy error.
        self.state.mkdir(parents=True, exist_ok=True)
        with (self.state / 'installation.lock').open('a+') as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                raise RuntimeError('Another installation operation is running') from None
            self.lock_fd = stream.fileno()
            try:
                yield
            finally:
                self.lock_fd = None

    @staticmethod
    def valid_receipt(receipt):
        return (isinstance(receipt, dict) and isinstance(receipt.get('files', {}), dict)
                and isinstance(receipt.get('permissions', []), list)
                and all(isinstance(entry, dict) for entry in receipt.get('files', {}).values())
                and all(isinstance(entry, dict) for entry in receipt.get('permissions', [])))

    def recover_core_receipts(self, receipt):
        """Reconstruct only exact generated core files from a verified active payload."""
        try:
            generation = self.generation_state()
            root = Path(generation['root'])
            if root.parent != self.data / 'releases' or root.is_symlink():
                return
            if verify_payload(root)['version'] != generation['version']:
                return
            launcher = self.data / 'launcher'
            expected = {launcher: (root / 'scripts/managed-launcher.sh').read_text(encoding='utf-8'),
                        self.data / 'interpreter': generation['python']['path'] + '\n'}
            config = self.config_home()
            unit = config / 'systemd/user/hyprwhspr.service'
            # Adoption must match the generated unit exactly; merging the installed
            # values in would adopt a unit whose directories the user re-pointed.
            expected[unit] = service_content(root, launcher)
            files = receipt.setdefault('files', {})
            for path, content in expected.items():
                try:
                    if str(path) not in files and not path.is_symlink() and path.is_file() and path.read_text(encoding='utf-8') == content:
                        files[str(path)] = file_receipt(path, 'integration')
                        print(f'Reconstructed ownership of verified generated file: {path}')
                except (OSError, UnicodeError) as exc:
                    print(f'Preserved unverified file during ownership recovery: {path}: {exc}', file=sys.stderr)
            command = self.command_path()
            if (str(command) not in files and str(launcher) in files and command.is_symlink()
                    and (command.parent / os.readlink(command)).resolve() == launcher.resolve()):
                files[str(command)] = file_receipt(command, 'integration')
        except (OSError, ValueError, TypeError, KeyError) as exc:
            print(f'Could not reconstruct core ownership; existing files remain unowned: {exc}', file=sys.stderr)

    def read_receipt(self):
        # Readers and writers share the same short lock, including corruption
        # archival and reconstruction; no read-modify-write race on repair.
        with self.receipts() as receipt:
            return receipt

    @contextmanager
    def receipts(self):
        """Serialize short receipt edits independently of long lifecycle operations."""
        self.state.mkdir(parents=True, exist_ok=True)
        with (self.state / 'ownership.lock').open('a+') as stream:
            fcntl.flock(stream, fcntl.LOCK_EX)
            existed = self.receipt.exists() or self.receipt.is_symlink()
            receipt = self.read_state(self.receipt, {}, validate=self.valid_receipt, protect_runtime=False)
            if any(self.state.glob('ownership.json.unreadable-*')):
                self.recover_core_receipts(receipt)
            yield receipt
            if receipt or existed:
                atomic_json(self.receipt, receipt)

    def generation_state(self):
        """The active generation, or {} if the pointer is unusable.

        current.json is the one state file that used to abort every command when
        corrupt. It is quarantined like the journal and the receipt so update,
        repair and uninstall stay reachable without hand-editing anything.
        """
        return self.read_state(self.current, {}, protect_runtime=False,
                               validate=lambda value: not value or isinstance(value.get('root', ''), str))

    def xdg_roots(self, old=None, repair=False):
        """The XDG config root this installation is pinned to.

        Only the config root is pinned. The data and state roots are how the
        launcher and CLI *find* an installation at all (managed-launcher.sh reads
        $XDG_DATA_HOME/hyprwhspr/current.json), so they cannot be dictated by the
        generation they locate - recording them would claim an authority this code
        does not have. Config has no such circularity: nothing needs it to find the
        install, so it is fixed at first install and carried forward, which stops an
        update run from a bare TTY silently repointing the daemon's configuration.
        """
        ambient = str(Path(os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config')))
        recorded = ((old or {}).get('xdg') or {}).get('config')
        if not recorded or repair:
            return {'config': ambient}
        if recorded != ambient:
            print(f'Keeping the pinned configuration directory {recorded} rather than the '
                  f'current environment\'s {ambient}. Run hyprwhspr install repair from the '
                  'session you want to pin to change it.', file=sys.stderr)
        return {'config': recorded}

    def config_home(self):
        """Config directory of the active generation, falling back to the environment."""
        recorded = self.generation_state().get('xdg') or {}
        return Path(recorded.get('config')
                    or os.environ.get('XDG_CONFIG_HOME', Path.home() / '.config'))

    def read_transaction(self):
        try:
            tx = read_json(self.journal)
            if tx is None:
                return None
            if not isinstance(tx, dict) or tx.get('phase') not in ('staging', 'activating', 'activated', 'committed'):
                raise ValueError('invalid transaction phase or object')
            names = ('garbage',) if tx['phase'] == 'committed' else ('created',)
            for key in names:
                if not isinstance(tx.get(key), list) or any(not isinstance(p, str) for p in tx[key]):
                    raise ValueError(f'invalid {key} paths')
            if tx['phase'] in ('activating', 'activated'):
                if 'old' not in tx or (tx['old'] is not None and not isinstance(tx['old'], dict)):
                    raise ValueError('missing or invalid previous generation')
                if not isinstance(tx.get('integrations', {}), dict):
                    raise ValueError('invalid integration snapshots')
            return tx
        except (OSError, ValueError, TypeError) as exc:
            raise ValueError(f'Cannot read transaction journal {self.journal}: {exc}') from exc

    def quarantine_transaction(self, error):
        # An invalid journal cannot tell us which generation was committed.
        # Preserve all existing runtimes and the original evidence, not guesses.
        marker = self.state / 'unverified-generations.json'
        protected = self.read_state(marker, [])
        for directory in ('releases', 'environments', 'transactions'):
            base = self.data / directory
            if base.is_dir():
                protected.extend(str(path) for path in base.iterdir())
        atomic_json(marker, list(dict.fromkeys(protected)))
        destination = self.journal.with_name('transaction-unreadable-' + uuid.uuid4().hex + '.json')
        os.replace(self.journal, destination)
        print(f'{error}. Preserved journal at {destination}; current pointer and existing runtimes were left unchanged.', file=sys.stderr)

    def status(self):
        def inspect(path, reader=None, validate=None):
            try:
                value = reader() if reader else read_json(path)
                if validate is not None and value is not None and not validate(value):
                    raise ValueError('Invalid ownership receipt')
                return value
            except (OSError, ValueError, TypeError) as exc:
                return {'error': f'{path}: {exc}'}
        return {'generation': inspect(self.current), 'transaction': inspect(self.journal, self.read_transaction),
                'ownership': inspect(self.receipt, validate=self.valid_receipt),
                'deferred_cleanup': inspect(self.state / 'deferred-cleanup.json'),
                'deferred_restoration': inspect(self.state / 'deferred-restoration.json'),
                'unverified_generations': inspect(self.state / 'unverified-generations.json'),
                'restoration_reviews': sorted(str(path) for path in self.state.glob('restoration-review-*.json')),
                'preserved_ownership_receipts': sorted(str(path) for path in self.state.glob('ownership.json.unreadable-*'))}

    def read_state(self, path, default, *, validate=None, protect_runtime=True):
        def archive(candidate):
            destination = candidate.with_name(candidate.name + '.unreadable-' + uuid.uuid4().hex)
            try:
                os.replace(candidate, destination)
                print(f'Preserved unusable recovery state at {destination}', file=sys.stderr)
            except OSError as exc:
                print(f'Could not archive {candidate}: {exc}; leave it intact for inspection', file=sys.stderr)

        def preserve(candidate):
            archive(candidate)
            if not protect_runtime:
                return
            protected = set(getattr(self, '_uncertain_paths', set()))
            for name in ('releases', 'environments', 'transactions'):
                try:
                    protected.update(str(p) for p in (self.data / name).iterdir())
                except OSError:
                    pass
            marker = self.state / 'unverified-generations.json'
            protected.update(read_state(marker, [], on_error=archive))
            self._uncertain_paths = protected
            try:
                atomic_json(marker, sorted(protected))
            except OSError as exc:
                print(f'Could not persist conservative runtime protection: {marker}: {exc}', file=sys.stderr)
        return read_state(path, default, on_error=preserve, validate=validate)

    def owned_path(self, path):
        path = Path(path)
        legacy_owned = [*self.read_state(self.journal, {}, validate=lambda tx: isinstance(tx.get('legacy_owned', []), list) and all(isinstance(p, str) for p in tx.get('legacy_owned', []))).get('legacy_owned', []),
                        *self.read_state(self.state / 'deferred-legacy-owned.json', [])]
        legacy = path in (self.data / 'src', self.data / 'venv') and str(path) in legacy_owned
        if not legacy and path.parent not in (self.data / 'releases', self.data / 'environments', self.data / 'transactions'):
            raise ValueError(f'Refusing to remove unowned path: {path}')
        if path.is_symlink():
            raise ValueError(f'Refusing to follow symlink: {path}')
        return path

    def remove(self, paths, preserve=(), final_uninstall=False):
        deferred_file = self.state / 'deferred-cleanup.json'
        pending = self.read_state(deferred_file, [])
        deferred = []
        failures = []
        daemon_paths = self.daemon_paths()
        restoration = self.recovery_step(self.restoration_state)
        # Unknown rollback references must never authorize deleting a runtime.
        preserve = {*preserve, *self.read_state(self.state / 'unverified-generations.json', []),
                    *([*pending, *paths] if restoration is None else restoration.get('protected', []))}
        preserve.update(getattr(self, '_uncertain_paths', set()))
        for name in dict.fromkeys([*pending, *paths]):
            try:
                path = self.owned_path(name)
                # In-process backend setup can replace its own CLI environment.
                # Keep its libraries and source available until a later process cleans up.
                if not final_uninstall and any(Path(active).is_relative_to(path) for active in
                       (sys.prefix, sys.executable, __file__)):
                    deferred.append(name)
                    print(f'Deferred cleanup of running application: {path}')
                    continue
                if name in preserve or name in getattr(self, '_uncertain_paths', set()) or path in daemon_paths:
                    deferred.append(name)
                    print(f'Deferred cleanup while daemon may still use runtime: {path}')
                    continue
                if path.exists():
                    shutil.rmtree(path)
            except (OSError, ValueError) as exc:
                deferred.append(name)
                if name in pending:
                    print(f'Previously deferred cleanup still needs attention: {name}: {exc}', file=sys.stderr)
                else:
                    failures.append(f'{name}: {exc}')
        legacy = [*self.read_state(self.journal, {}, validate=lambda tx: isinstance(tx.get('legacy_owned', []), list) and all(isinstance(p, str) for p in tx.get('legacy_owned', []))).get('legacy_owned', []),
                  *self.read_state(self.state / 'deferred-legacy-owned.json', [])]
        atomic_json(self.state / 'deferred-legacy-owned.json',
                    [name for name in deferred if name in legacy and
                     Path(name) in (self.data / 'src', self.data / 'venv')])
        atomic_json(deferred_file, deferred)
        if failures:
            raise CleanupError('Cleanup incomplete; failed paths are recorded for retry (correct ownership/permissions if needed): ' + '; '.join(failures))

    def service(self, action):
        return run(['systemctl', '--user', action, 'hyprwhspr.service'],
                   capture_output=True, text=True, timeout=30)

    def service_state(self):
        try:
            result = run(['systemctl', '--user', 'show', 'hyprwhspr.service',
                          '--property=ActiveState,SubState,NRestarts,MainPID,FragmentPath'],
                         capture_output=True, text=True, timeout=10)
            return dict(line.split('=', 1) for line in result.stdout.splitlines() if '=' in line)
        except (OSError, subprocess.SubprocessError):
            return {}

    def daemon_paths(self):
        """Protect the daemon's generation, not every previously deferred path."""
        protected = self.state / 'daemon-protected-cleanup.json'
        if not self.daemon_running():
            try:
                protected.unlink(missing_ok=True)
            except OSError:
                pass
            return set()
        runtime = Path(os.environ.get('XDG_RUNTIME_DIR', tempfile.gettempdir()))
        runtime /= 'hyprwhspr' if os.environ.get('XDG_RUNTIME_DIR') else f'hyprwhspr-{os.getuid()}'
        try:
            pid = int((runtime / 'hyprwhspr.lock').read_text(encoding='utf-8').strip())
            args = Path(f'/proc/{pid}/cmdline').read_bytes().split(b'\0')
            roots = [Path(os.fsdecode(arg)).parents[1] for arg in args
                     if os.fsdecode(arg).endswith('/lib/main.py')]
            if roots:
                return {*roots, Path(os.fsdecode(args[0])).parent.parent}
        except (OSError, ValueError):
            pass
        # A failed rollback stop may leave the replacement running while the
        # pointer names the restored generation. Preserve those specific paths.
        generation = self.read_state(self.current, {}, validate=lambda value: all(
            value.get(key) is None or isinstance(value[key], dict) for key in ('cli', 'backend')))
        paths = [generation.get('root'), (generation.get('backend') or {}).get('path'),
                 (generation.get('cli') or {}).get('path'), *self.read_state(protected, [])]
        return {Path(path) for path in paths if path}

    def daemon_running(self):
        runtime = Path(os.environ.get('XDG_RUNTIME_DIR', tempfile.gettempdir()))
        runtime = runtime / ('hyprwhspr' if os.environ.get('XDG_RUNTIME_DIR') else f'hyprwhspr-{os.getuid()}')
        path = runtime / 'hyprwhspr.lock'
        try:
            with path.open('r', encoding='utf-8') as stream:
                try:
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    return False
                except BlockingIOError:
                    return True
        except FileNotFoundError:
            return False
        except OSError as exc:
            print(f'Could not inspect daemon lock {path}: {exc}; treating daemon as running', file=sys.stderr)
            return True

    def stable(self):
        deadline = time.monotonic() + 60
        since = None
        identity = None
        while time.monotonic() < deadline:
            state = self.service_state()
            current = (state.get('MainPID'), state.get('NRestarts'))
            if state.get('ActiveState') == 'active' and state.get('SubState') == 'running':
                if current != identity:
                    since, identity = time.monotonic(), current
                if time.monotonic() - since >= 15:
                    return
            else:
                since, identity = None, None
            time.sleep(1)
        raise RuntimeError('Service did not remain stable for 15 seconds within 60 seconds')

    def recovery_command(self, action, *args, **kwargs):
        try:
            action(*args, **kwargs)
            return True
        except Exception as exc:
            detail = getattr(exc, 'stderr', None) or getattr(exc, 'stdout', None) or str(exc)
            print(f'Recovery warning: {detail}. Check the desktop/user systemd session; repair remains available.', file=sys.stderr)
            return False

    def recovery_step(self, action, *args, default=None):
        """Optional recovery work cannot prevent mandatory routing restoration."""
        try:
            return action(*args)
        except Exception as exc:
            print(f'Recovery deferred: {exc}. Recovery state is retained for retry.', file=sys.stderr)
            return default

    def restoration_state(self):
        return self.read_state(self.state / 'deferred-restoration.json', {}, validate=lambda pending:
            isinstance(pending.get('files', {}), dict) and
            isinstance(pending.get('protected', []), list) and
            all(isinstance(path, str) for path in pending.get('protected', [])))

    @contextmanager
    def restore_pointer(self, tx):
        """All pre-restore work belongs inside this mandatory unwind boundary."""
        try:
            yield
        finally:
            if tx['old'] is not None:
                atomic_json(self.current, tx['old'])
            else:
                self.current.unlink(missing_ok=True)

    def integration_fingerprint(self, path):
        path = Path(path)
        if path.is_symlink():
            return {'target': os.readlink(path)}
        if not path.exists():
            return None
        return {'sha256': digest(path)}

    def stop_for_recovery(self):
        if self.service_state().get('ActiveState') in ('active', 'activating'):
            return self.recovery_command(self.service, 'stop')
        return True

    def restore_integrations(self):
        # Include queue reads and final persistence in the guarded boundary.
        # True means work remains, including an unreadable/malformed queue.
        return self.recovery_step(self._restore_integrations, default=True)

    def _restore_integrations(self):
        """Retry independent rollback steps; retain failures and their runtime paths."""
        queue_path = self.state / 'deferred-restoration.json'
        pending = self.restoration_state()
        jobs = pending.get('files', {})
        for name, job in list(jobs.items()):
            path = Path(name)
            temporary = None
            try:
                original = job['snapshot']
                desired = (None if original is None else {'target': original['target']} if 'target' in original else
                           {'sha256': hashlib.sha256(original['content'].encode('utf-8')).hexdigest()})
                current = self.integration_fingerprint(path)
                if 'expected' not in job:
                    review = queue_path.with_name('restoration-review-' + uuid.uuid4().hex + '.json')
                    atomic_json(review, {'path': name, 'job': job, 'observed': current})
                    content = os.readlink(path) if path.is_symlink() else (path.read_text(encoding='utf-8', errors='replace') if path.exists() else '')
                    referenced = [p for p in pending.get('protected', []) if p in content]
                    if referenced:
                        marker = self.state / 'unverified-generations.json'
                        atomic_json(marker, list(dict.fromkeys([*self.read_state(marker, []), *referenced])))
                    print(f'Preserved integration without rollback precondition: {path}; snapshot retained at {review}. '
                          'Unsafe automatic restore retired; only referenced runtimes remain protected.', file=sys.stderr)
                    del jobs[name]
                    continue
                if current != desired and current != job['expected']:
                    print(f'Preserved integration changed since rollback was queued: {path}', file=sys.stderr)
                    continue
                if current != desired:
                    if original is None:
                        path.unlink(missing_ok=True)
                    elif 'target' in original:
                        path.parent.mkdir(parents=True, exist_ok=True)
                        temporary = path.with_name('.' + path.name + '-rollback-' + uuid.uuid4().hex)
                        temporary.symlink_to(original['target'])
                        os.replace(temporary, path)
                    else:
                        atomic_text(path, original['content'], original['mode'])
                if 'receipt' in job:
                    with self.receipts() as receipt:
                        files = receipt.setdefault('files', {})
                        if job['receipt'] is None:
                            files.pop(name, None)
                        else:
                            files[name] = job['receipt']
                del jobs[name]
            except (OSError, ValueError, KeyError, TypeError) as exc:
                print(f'Recovery warning: integration restoration deferred: {path}: {exc}. '
                      'Correct access and retry repair; the generation pointer was restored.', file=sys.stderr)
            finally:
                if temporary is not None:
                    try:
                        temporary.unlink(missing_ok=True)
                    except OSError:
                        pass
        if pending:
            if not jobs:
                if pending.get('legacy_unit'):
                    self.recovery_command(run, ['systemctl', '--user', 'daemon-reload'], capture_output=True, timeout=30)
                if pending.get('reload_hyprland'):
                    self.recovery_command(run, ['hyprctl', 'reload'], capture_output=True, timeout=10)
                if pending.get('running') and self.service_state().get('ActiveState') not in ('active', 'activating'):
                    self.recovery_command(self.service, 'start')
                queue_path.unlink(missing_ok=True)
            else:
                atomic_json(queue_path, pending)
        return bool(jobs)

    def recover(self, strict=False, final_uninstall=False):
        """Restore routing first; isolate integration failures in a durable retry queue.

        Operation completion can request a nonzero cleanup result with strict=True.
        The journal retires only after failed paths have been persisted separately.
        """
        def cleanup(paths, preserve=()):
            try:
                self.remove(paths, preserve=preserve, final_uninstall=final_uninstall)
            except CleanupError as exc:
                print(str(exc), file=sys.stderr)
                return exc
            return None

        try:
            tx = self.read_transaction()
        except ValueError as exc:
            self.quarantine_transaction(exc)
            return
        if tx is None:
            self.restore_integrations()
            error = cleanup([])
            if error and strict:
                raise error
            return
        if tx['phase'] != 'committed':
            if tx['phase'] in ('activating', 'activated'):
                with self.restore_pointer(tx):
                    stopped = self.recovery_step(self.stop_for_recovery, default=False)
                if not stopped:
                    atomic_json(self.state / 'daemon-protected-cleanup.json', tx['created'])
                    deferred = self.state / 'deferred-cleanup.json'
                    atomic_json(deferred, list(dict.fromkeys([*self.read_state(deferred, []), *tx['created']])))
                if tx.get('integrations'):
                    queue_path = self.state / 'deferred-restoration.json'
                    pending = self.restoration_state()
                    jobs = pending.setdefault('files', {})
                    for name, snapshot in tx['integrations'].items():
                        job = {'snapshot': snapshot}
                        try:
                            job['expected'] = self.integration_fingerprint(name)
                        except OSError as exc:
                            print(f'Cannot establish rollback precondition for {name}: {exc}; preserving file', file=sys.stderr)
                        if 'receipt' in tx:
                            job['receipt'] = tx['receipt'].get('files', {}).get(name)
                        jobs.setdefault(name, job)
                    pending['protected'] = list(dict.fromkeys([*pending.get('protected', []), *tx['created']]))
                    for flag in ('legacy_unit', 'reload_hyprland', 'running'):
                        pending[flag] = pending.get(flag) or tx.get(flag)
                    # Do not retire the transaction unless failures can be retried.
                    atomic_json(queue_path, pending)
                self.restore_integrations()
                if tx.get('legacy_unit'):
                    self.recovery_command(run, ['systemctl', '--user', 'daemon-reload'], capture_output=True, timeout=30)
                if tx.get('reload_hyprland'):
                    self.recovery_command(run, ['hyprctl', 'reload'], capture_output=True, timeout=10)
                if tx.get('running'):
                    self.recovery_command(self.service, 'start')
            error = cleanup(tx['created'], preserve=tx['created'] if tx['phase'] in ('activating', 'activated') and not stopped else ())
        else:
            error = cleanup(tx['garbage'])
        self.journal.unlink()
        if error and strict:
            raise error

    def release(self, version, work):
        # Match application semver tags only, never wheels or GUI runtime tags.
        if version and not VERSION.fullmatch(version):
            raise ValueError('Version must be a stable application tag, e.g. v1.2.3')
        if version:
            version = 'v' + version.lstrip('v')
        endpoint = f'/tags/{version}' if version else '?per_page=100'
        download(f'https://api.github.com/repos/{REPOSITORY}/releases{endpoint}', work / 'releases.json', 8 * 1024 * 1024)
        result = read_json(work / 'releases.json')
        releases = [result] if version else result
        selected = [r for r in releases if not r['draft'] and not r['prerelease']
                    and VERSION.fullmatch(r['tag_name'])
                    and (not version or r['tag_name'].lstrip('v') == version.lstrip('v'))
                    and any(a['name'] == 'hyprwhspr-release.json' for a in r['assets'])]
        if not selected:
            raise RuntimeError('No compatible published application release found')
        release = max(selected, key=lambda r: tuple(map(int, r['tag_name'].lstrip('v').split('.'))))
        assets = {a['name']: a['browser_download_url'] for a in release['assets']}
        download(assets['hyprwhspr-release.json'], work / 'metadata.json', 1024 * 1024)
        metadata = read_json(work / 'metadata.json')
        if metadata.get('format') != 1 or metadata['version'] != release['tag_name']:
            raise ValueError('Release metadata does not match selected tag')
        if not re.fullmatch('[0-9a-f]{64}', metadata['sha256']):
            raise ValueError('Invalid release checksum')
        download(assets[metadata['archive']], work / 'application.tar.gz')
        return metadata

    def healthy(self, environment, imports):
        try:
            python = Path(environment) / 'bin/python'
            run([python, '-s', '-c', ';'.join('import ' + name for name in imports)],
                capture_output=True, timeout=max(300, 60 * len(imports)))
            # Distro distributions are visible for desktop bindings. Their unrelated
            # dependency conflicts must not invalidate the application's import probe.
            return True
        except (OSError, subprocess.SubprocessError):
            return False

    def build(self, root, identity, kind, selection, old, tx, force):
        sys.path.insert(0, str(root / 'lib/src'))
        from dependency_manifest import parse_graph, fingerprint
        from dependency_plan import resolve
        plan = resolve(root, selection['backend'], selection.get('provider'), selection.get('variant'), ValueError) if kind == 'backend' else None
        manifest = plan.manifest if plan else root / 'requirements-cli.txt'
        imports = plan.required_imports if plan else CLI_IMPORTS
        key = {'python': identity, 'dependencies': fingerprint(parse_graph(manifest, ValueError).manifests),
               'selection': selection if plan else None}
        if plan and plan.family == 'pywhispercpp':
            key['builder'] = digest(root / 'lib/src/backend_installer.py')
        previous = (old or {}).get(kind)
        if not force and previous and previous['key'] == key and self.healthy(previous['path'], imports):
            return previous
        target = self.data / 'environments' / (kind + '-' + uuid.uuid4().hex)
        tx['created'].append(str(target))
        atomic_json(self.journal, tx)
        target.parent.mkdir(parents=True, exist_ok=True)
        run([identity['path'], '-I', '-m', 'venv', '--system-site-packages', target])
        env = clean_env()
        env.update(PIP_CACHE_DIR=str(Path(tx['work']) / 'pip-cache'), TMPDIR=tx['work'])
        python = target / 'bin/python'
        # --ignore-installed makes the entire declared closure local even with
        # system-site access retained for distro desktop bindings.
        subprocess.run([str(python), '-s', '-m', 'pip', 'install', '--ignore-installed',
                        '-r', str(manifest)], check=True, env=env)
        if plan and plan.family == 'pywhispercpp' and selection.get('variant') in ('nvidia', 'amd', 'vulkan'):
            env['HYPRWHSPR_ROOT'] = str(root)
            env['HYPRWHSPR_BUILD_ENV'] = str(target)
            env['HYPRWHSPR_BUILD_WORK'] = tx['work']
            script = root / 'lib/src/managed_backend.py'
            subprocess.run([str(python), '-s', str(script), selection['variant']], check=True, env=env)
        if not self.healthy(target, imports):
            raise EnvironmentVerificationError(f'{kind} environment verification failed')
        return {'path': str(target), 'key': key}

    def build_backend(self, root, identity, selection, old, tx, force):
        if not selection:
            return None, selection
        try:
            return self.build(root, identity, 'backend', selection, old, tx, force), selection
        except (subprocess.CalledProcessError, EnvironmentVerificationError) as exc:
            if selection.get('variant') not in ('nvidia', 'amd', 'vulkan', 'cuda', 'gpu'):
                raise
            # Only failed environment builds can fall back. Activation, disk,
            # payload and configuration errors must retain the previous runtime.
            fallback = dict(selection, variant=None)
            if selection['backend'] in ('nvidia', 'amd', 'vulkan', 'pywhispercpp'):
                fallback.update(backend='cpu', variant='cpu')
            print(f'Accelerated backend build failed ({exc}); rebuilding in a fresh CPU environment.')
            return self.build(root, identity, 'backend', fallback, old, tx, True), fallback

    def validate_config(self, root, cli, config_home=None):
        config = (Path(config_home) if config_home is not None else self.config_home()) / 'hyprwhspr/config.json'
        if config.exists():
            try:
                run([Path(cli['path']) / 'bin/python', '-s', '-c',
                     'import json,jsonschema,sys; sys.path.insert(0,sys.argv[3]); '
                     'from config_manager import normalize_legacy_config; '
                     'config,_=normalize_legacy_config(json.load(open(sys.argv[1],encoding="utf-8"))); '
                     'jsonschema.validate(config,json.load(open(sys.argv[2],encoding="utf-8")))',
                     config, root / 'share/config.schema.json', root / 'lib/src'], capture_output=True, timeout=30)
            except subprocess.CalledProcessError as exc:
                detail = exc.stderr or exc.stdout or 'validator exited without a diagnostic'
                if isinstance(detail, bytes):
                    detail = detail.decode(errors='replace')
                raise RuntimeError(f'Configuration validation failed for {config}. '
                                   f'Fix the configuration and retry; the active installation is unchanged.\n{detail.strip()}') from None


    def package_installed(self):
        return Path('/usr/lib/hyprwhspr').exists()

    def update(self, version=None, python=None, repair=False, selection=None, force_backend=False, local_payload=False):
        with self.lock():
            self.recover()
            old = self.generation_state() or None
            if not old and self.package_installed():
                raise RuntimeError('Package installation detected. Update hyprwhspr with your package manager.')
            identity = interpreter(python or (old or {}).get('python', {}).get('path'), fallback=not python)
            work = self.data / 'transactions' / uuid.uuid4().hex
            tx = {'phase': 'staging', 'old': old, 'created': [str(work)], 'garbage': [], 'work': str(work)}
            atomic_json(self.journal, tx)
            work.mkdir(parents=True)
            try:
                if local_payload:
                    if not old or (version and version.lstrip('v') != old['version'].lstrip('v')):
                        raise RuntimeError('Local backend rebuild requires the installed application version')
                    root = Path(old['root'])
                    metadata = verify_payload(root)
                    if metadata['version'] != old['version']:
                        raise ValueError('Local payload version does not match the active generation')
                else:
                    metadata = self.release(version or (old['version'] if repair and old else None), work)
                    root = self.data / 'releases' / uuid.uuid4().hex
                    tx['created'].append(str(root))
                    atomic_json(self.journal, tx)
                    extract(work / 'application.tar.gz', root, metadata['sha256'])
                    if read_json(root / 'release.json')['version'] != metadata['version']:
                        raise ValueError('Archive version does not match release metadata')
                if selection is None:
                    selection = (old or {}).get('selection') or self.legacy_selection()
                available = shutil.disk_usage(self.data).free
                print(f'Staging space available: {available // 1048576} MiB. '
                      'Replacements temporarily coexist with active environments; '
                      'package downloads and GPU builds require additional space. '
                      'The 256 MiB minimum check does not guarantee sufficient build space.')
                if available < 256 * 1048576:
                    raise RuntimeError('Insufficient staging space; active installation is unchanged')
                if old and old['version'] == metadata['version'] and not repair and not local_payload:
                    try:
                        verify_payload(old['root'])
                    except (OSError, ValueError):
                        pass
                    else:
                        tx['garbage'].append(str(root))
                        root = Path(old['root'])
                if not old and not (self.data / 'src').exists():
                    print('Installing host prerequisites; these changes are outside application rollback.')
                    # interpreter() already validated this one. Hand it over, or the
                    # script judges the distro's default python3 and refuses a
                    # bootstrap that was explicitly given a supported --python.
                    prerequisites = clean_env()
                    prerequisites['INSTALL_DEPS_PYTHON'] = identity['path']
                    prerequisites['HYPRWHSPR_MANAGED_PREREQUISITES'] = '1'
                    run(['bash', root / 'scripts/install-deps.sh'], env=prerequisites)
                cli = self.build(root, identity, 'cli', selection, old, tx, repair)
                backend, selection = self.build_backend(root, identity, selection, old, tx, repair or force_backend)
                xdg = self.xdg_roots(old, repair)
                self.validate_config(root, cli, xdg['config'])
                generation = {'format': 1, 'id': uuid.uuid4().hex, 'version': metadata['version'],
                              'root': str(root), 'python': identity, 'cli': cli, 'backend': backend,
                              'selection': selection, 'xdg': xdg}
                if old and all(generation[key] == old.get(key) for key in ('root', 'python', 'cli', 'backend', 'selection', 'xdg')):
                    tx.update(phase='committed', garbage=tx['created'])
                    atomic_json(self.journal, tx)
                    try:
                        self.recover(strict=True)
                    except Exception as exc:
                        raise CommittedCleanupError(old, exc) from exc
                    print(f"Already installed and healthy: {old['version']}")
                    return old
                legacy_unit = self.check_integrations(root)
                service = self.service_state()
                running = service.get('ActiveState') in ('active', 'activating')
                if self.daemon_running() and not running:
                    raise RuntimeError('Stop the manually launched daemon before activation')
                bindings = self.binding_migrations()
                bar_changes = self.bar_migrations(root)
                reload_hyprland = bool(bindings and os.environ.get('HYPRLAND_INSTANCE_SIGNATURE') and shutil.which('hyprctl'))
                targets = [self.command_path(), self.data / 'launcher', self.data / 'interpreter', *bindings, *bar_changes]
                if legacy_unit:
                    targets.append(Path(legacy_unit))
                snapshots = {}
                for path in targets:
                    snapshots[str(path)] = ({'target': os.readlink(path)} if path.is_symlink() else
                        {'content': path.read_text(encoding='utf-8'), 'mode': path.stat().st_mode & 0o777} if path.exists() else None)
                tx.update(phase='activating', running=running, integrations=snapshots,
                          receipt=self.read_receipt(), legacy_unit=legacy_unit, reload_hyprland=reload_hyprland)
                atomic_json(self.journal, tx)
                if running:
                    self.service('stop')
                    if self.daemon_running():
                        raise RuntimeError('Daemon did not stop gracefully')
                atomic_json(self.current, generation)
                tx['phase'] = 'activated'
                atomic_json(self.journal, tx)
                self.install_launcher(root)
                changes = {**bindings, **bar_changes}
                if changes:
                    try:
                        from .managed_integrations import edit_files
                    except ImportError:
                        from managed_integrations import edit_files
                    with edit_files(changes, changes) as proceed:
                        if not proceed:
                            raise RuntimeError('Binding migration was skipped; preserving the previous generation')
                        for path, content in changes.items():
                            path.parent.mkdir(parents=True, exist_ok=True)
                            snapshot = snapshots[str(path)]
                            mode = snapshot['mode'] if snapshot else (0o755 if path.suffix == '.sh' else 0o644)
                            atomic_text(path, content, mode)
                    if reload_hyprland:
                        self.recovery_command(run, ['hyprctl', 'reload'], capture_output=True, timeout=10)
                if legacy_unit:
                    atomic_text(Path(legacy_unit),
                                service_content(root, self.data / 'launcher', generation['xdg']),
                                Path(legacy_unit).stat().st_mode & 0o777)
                    record_file(legacy_unit)
                    run(['systemctl', '--user', 'daemon-reload'], capture_output=True, timeout=30)
                if running:
                    self.service('start')
                    self.stable()
                references = {str(root), cli['path']}
                if backend:
                    references.add(backend['path'])
                tx['garbage'].append(str(work))
                if old:
                    tx['garbage'] += [p for p in [old['root'], old['cli']['path'],
                                      (old.get('backend') or {}).get('path')] if p and p not in references]
                tx['garbage'].extend(p for p in tx['created'] if p not in references and p not in tx['garbage'])
                tx['phase'] = 'committed'
                atomic_json(self.journal, tx)
                try:
                    self.recover(strict=True)
                    self.clean_legacy()
                except Exception as exc:
                    raise CommittedCleanupError(generation, exc) from exc
                return generation
            except BaseException as original:
                if tx['phase'] != 'committed':
                    try:
                        self.recover()
                    except Exception as recovery_error:
                        original.add_note(f'Recovery also incomplete: {recovery_error}')
                raise

    def legacy_selection(self):
        if not (self.data / 'src/.git').exists():
            return None
        try:
            upstream = run(['git', '-C', self.data / 'src', 'remote', 'get-url', 'origin'], capture_output=True, text=True, timeout=30).stdout.strip()
        except (OSError, subprocess.SubprocessError) as exc:
            print(f'Could not identify legacy checkout upstream: {exc}. Preserving it and continuing without adopting its backend selection.')
            return None
        if upstream not in (f'https://github.com/{REPOSITORY}.git', f'https://github.com/{REPOSITORY}'):
            raise RuntimeError('Legacy location is an unrelated checkout; refusing to adopt it')
        state = read_json(self.state / 'install-state.json', {})
        backend = state.get('installed_backend') or state.get('backend_type') or state.get('backend')
        if not backend:
            config = self.config_home() / 'hyprwhspr/config.json'
            backend = read_json(config, {}).get('transcription_backend')
        return {'backend': backend, 'variant': state.get('accelerated_variant') or (backend if backend in ('cpu', 'nvidia', 'amd', 'vulkan') else None)} if backend else None

    def legacy_bar_paths(self):
        config = self.config_home()
        data = Path(os.environ.get('XDG_DATA_HOME', Path.home() / '.local/share'))
        return [config / 'waybar/hyprwhspr-module.jsonc', config / 'waybar/style.css',
                data / 'noctalia/plugins/noctwhspr/widget.luau',
                data / 'noctalia/plugins/hyprwhspr/widget.luau']

    def bar_processes(self):
        """Identify current user's bars, including start time to avoid PID reuse."""
        result = {}
        for process in Path('/proc').iterdir():
            if not process.name.isdigit():
                continue
            try:
                if process.stat().st_uid == os.getuid() and process.joinpath('comm').read_text(encoding='utf-8').strip() in ('waybar', 'noctalia'):
                    result[process.name] = process.joinpath('stat').read_text(encoding='utf-8').rsplit(')', 1)[1].split()[19]
            except (OSError, IndexError):
                continue
        return result

    def bar_migrations(self, root=None):
        """Rewrite exact legacy references; preserve uncertain plugins and checkouts."""
        import shlex
        try:
            from .managed_integrations import managed_widget_content
        except ImportError:
            from managed_integrations import managed_widget_content
        legacy = self.data / 'src'
        if not legacy.is_dir():
            return {}
        shim = self.data / 'integrations/legacy-tray.sh'
        shim_content = '#!/bin/sh\nexec ' + shlex.quote(str(self.data / 'launcher')) + ' --managed-tray "$@"\n'
        try:
            conflicting_shim = shim.is_symlink() or (shim.exists() and shim.read_text(encoding='utf-8') != shim_content)
        except (OSError, UnicodeError) as exc:
            print(f'Preserving unreadable migration shim: {shim}: {exc}')
            return {}
        if conflicting_shim:
            print(f'Preserved conflicting migration shim: {shim}; legacy bars retain their checkout')
            return {}
        changes = {}
        tray = str(legacy / 'config/hyprland/hyprwhspr-tray.sh')
        css = legacy / 'config/waybar/hyprwhspr-style.css'
        css_source = css if css.is_file() else Path(root or Path(__file__).resolve().parents[2]) / 'config/waybar/hyprwhspr-style.css'
        for path in self.legacy_bar_paths():
            if not path.is_file() or (path.is_symlink() and path.name != 'style.css'):
                continue
            try:
                content = path.read_text(encoding='utf-8')
            except (OSError, UnicodeError) as exc:
                print(f'Preserving unreadable legacy bar integration: {path}: {exc}')
                continue
            updated = content
            if path.name == 'hyprwhspr-module.jsonc':
                updated = content.replace(tray, str(shim))
            elif path.name == 'style.css' and str(css) in content and css_source.is_file():
                target = path.parent / 'hyprwhspr-style.css'
                try:
                    css_content = css_source.read_text(encoding='utf-8')
                    conflicting_css = target.is_symlink() or (target.exists() and target.read_text(encoding='utf-8') != css_content)
                except (OSError, UnicodeError) as exc:
                    print(f'Preserving unreadable stylesheet migration from {css_source} to {target}: {exc}')
                    continue
                if conflicting_css:
                    print(f'Preserving conflicting stylesheet: {target}')
                    continue
                updated = content.replace(str(css), str(target))
                changes[target] = css_content
            elif path.name == 'widget.luau' and self.unmanaged_widget(content):
                try:
                    updated = managed_widget_content(content, shim)
                except RuntimeError as exc:
                    print(f'Preserving legacy plugin: {path}: {exc}')
            if updated != content:
                changes[path.resolve() if path.name == 'style.css' else path] = updated
                if path.name != 'style.css':
                    changes[shim] = shim_content
        if changes:
            processes = self.bar_processes()
            if processes:
                changes[self.state / 'legacy-bar-processes.json'] = json.dumps(processes)
        return changes

    def legacy_bar_references(self):
        references = []
        for path in self.legacy_bar_paths():
            if not path.exists():
                continue
            try:
                content = path.read_text(encoding='utf-8')
            except (OSError, UnicodeError) as exc:
                print(f'Preserving unreadable legacy bar integration: {path}: {exc}')
                references.append(path)
                continue
            # Only an actual reference to the checkout blocks reclaiming it. A widget
            # that resolves through its own root - including one installed from
            # Noctalia's registry - is migrated when possible but never pins cleanup.
            if str(self.data / 'src') in content:
                references.append(path)
        return references

    def unmanaged_widget(self, content):
        """A widget still resolving its tray through an unmigrated install root."""
        try:
            from .managed_integrations import MANAGED_WIDGET_MARKER
        except ImportError:
            from managed_integrations import MANAGED_WIDGET_MARKER
        return ('/config/hyprland/hyprwhspr-tray.sh' in content
                and MANAGED_WIDGET_MARKER not in content)

    def hyprland_files(self):
        """Discover up to 256 configs, following source directives resolvable here."""
        import glob
        config = self.config_home() / 'hypr'
        pending = list(config.rglob('*.conf')) if config.exists() else []
        seen = set()
        while pending and len(seen) < 256:
            path = pending.pop()
            if path in seen:
                continue
            seen.add(path)
            try:
                content = path.read_text(encoding='utf-8')
            except (OSError, UnicodeError):
                continue
            for match in re.finditer(r'(?m)^\s*source\s*=\s*([^\n#]+)', content):
                source = os.path.expandvars(os.path.expanduser(match.group(1).strip().strip('"\'')))
                if '$' in source:
                    continue
                pattern = Path(source) if Path(source).is_absolute() else path.parent / source
                matches = [Path(name) for name in glob.glob(str(pattern)) if Path(name).is_file()]
                pending.extend(matches)
        return seen

    def legacy_binding_references(self):
        files = self.hyprland_files()
        references = []
        for path in files:
            try:
                content = path.read_text(encoding='utf-8', errors='replace')
            except (OSError, UnicodeError) as exc:
                print(f'Could not inspect Hyprland config: {path}: {exc}')
                continue
            for line in content.splitlines():
                line = line.strip()
                if line.startswith('#'):
                    continue
                if str(self.data / 'src') in line or ('hyprwhspr-tray.sh' in line and ('~' in line or '$' in line)):
                    references.append(path)
                    break
        return references

    def clean_legacy(self):
        legacy = self.data / 'src'
        if not legacy.exists():
            return
        process_marker = self.state / 'legacy-bar-processes.json'
        # An advisory bar-PID hint. Losing it costs a restart prompt, never a
        # generation, so a corrupt marker must not escalate to runtime protection.
        previous_bars = self.read_state(process_marker, {}, protect_runtime=False, validate=lambda bars: all(
            isinstance(pid, str) and isinstance(start, str) for pid, start in bars.items()))
        current_bars = self.bar_processes() if previous_bars else {}
        if any(current_bars.get(pid) == start for pid, start in previous_bars.items()):
            print('Preserved legacy checkout/runtime for bars still using the old configuration. Restart Waybar/Noctalia; the next lifecycle operation can finish cleanup.')
            return
        process_marker.unlink(missing_ok=True)
        referenced = self.legacy_bar_references() + self.legacy_binding_references()
        if referenced:
            print('Preserved legacy checkout and runtime still referenced by desktop integrations: ' + ', '.join(map(str, referenced)) + '. Replace the listed legacy tray/CSS references with the managed integration, or remove the obsolete integration, then retry update.')
            return
        if any(Path(active).is_relative_to(path) for active in (sys.prefix, sys.executable, __file__)
               for path in (legacy, self.data / 'venv')):
            print(f'Preserved legacy installation still used by this process: {legacy}')
            return
        try:
            upstream = run(['git', '-C', legacy, 'remote', 'get-url', 'origin'], capture_output=True, text=True, timeout=30).stdout.strip()
            if upstream not in (f'https://github.com/{REPOSITORY}.git', f'https://github.com/{REPOSITORY}'):
                raise ValueError('unexpected upstream')
            status = run(['git', '-C', legacy, 'status', '--porcelain', '-z', '--untracked-files=all', '--ignored'], capture_output=True, text=True, timeout=30).stdout
            entries = iter(status.split('\0'))
            for entry in entries:
                if not entry:
                    continue
                name = entry[3:]
                status_code = entry[:2]
                if 'R' in status_code or 'C' in status_code:
                    source = next(entries, '(missing source path)')
                    name = f'{source} -> {name}'
                if status_code != '!!':
                    category = 'untracked file' if status_code == '??' else f'tracked change ({status_code})'
                    raise ValueError(f'{category}: {name}; inspect and move or resolve it before retrying cleanup')
                artifact = legacy / name
                if entry.startswith('!! ') and not artifact.is_symlink():
                    if artifact.is_file() and artifact.suffix in ('.pyc', '.pyo'):
                        continue
                    if '__pycache__' in artifact.parts and artifact.is_dir() and all(
                            not child.is_symlink() and (child.is_dir() or child.suffix in ('.pyc', '.pyo'))
                            for child in artifact.rglob('*')):
                        continue
                raise ValueError(f'unrecognized ignored content: {name}; inspect and move or remove it before retrying cleanup')
            if run(['git', '-C', legacy, 'rev-list', 'HEAD', '--not', '--remotes=origin'], capture_output=True, text=True, timeout=30).stdout:
                raise ValueError('local-only commits')
        except (OSError, ValueError, subprocess.SubprocessError) as exc:
            print(f'Preserved legacy checkout and environment at {legacy.parent}: {exc}')
            return
        garbage = [str(legacy)]
        venv = self.data / 'venv'
        if (self.state / 'install-state.json').is_file() and (venv / 'pyvenv.cfg').is_file() and not venv.is_symlink():
            garbage.append(str(venv))
        elif venv.exists():
            print(f'Preserved legacy environment without ownership evidence: {venv}')
        atomic_json(self.journal, {'phase': 'committed', 'garbage': garbage, 'legacy_owned': garbage})
        self.recover(strict=True)
        for name in ('pywhispercpp-src', 'wheel-cache', 'runtime'):
            if (self.data / name).exists():
                print(f'Preserved legacy component without ownership evidence: {self.data / name}')

    def command_path(self):
        return Path.home() / '.local/bin/hyprwhspr'

    def check_integrations(self, root=None):
        command = self.command_path()
        if command.exists() or command.is_symlink():
            allowed = [self.data / 'launcher', self.data / 'src/bin/hyprwhspr']
            if not command.is_symlink() or (command.parent / os.readlink(command)).resolve() not in [p.resolve() for p in allowed]:
                raise RuntimeError(f'Conflicting command preserved: {command}')
        state = self.service_state()
        unit = state.get('FragmentPath')
        if unit and not Path(unit).exists() and not Path(unit).is_symlink():
            print(f'Ignoring stale systemd FragmentPath for missing unit: {unit}')
            return None
        if unit:
            if not Path(unit).is_file() or Path(unit).is_symlink():
                raise RuntimeError(f'Customized or unowned service preserved: {unit}')
            receipt = self.read_receipt().get('files', {})
            if str(unit) in receipt and digest(unit) == receipt[str(unit)].get('sha256'):
                # An owned unit still has to track the release template, which has
                # changed repeatedly across versions. Rewrite it whenever the
                # generation would render different content.
                if root is None:
                    return None
                try:
                    if Path(unit).read_text(encoding='utf-8') != service_content(
                            root, self.data / 'launcher'):
                        return unit
                except (OSError, UnicodeError) as exc:
                    print(f'Could not compare the installed service unit; leaving it unchanged: {exc}')
                return None
            if str(unit) not in receipt or digest(unit) != receipt[str(unit)].get('sha256'):
                legacy = self.data / 'src'
                if root is not None and (legacy / '.git').exists():
                    try:
                        from .legacy_units import generated_unit
                    except ImportError:
                        from legacy_units import generated_unit
                    if generated_unit(unit, legacy):
                        return unit
                raise RuntimeError(f'Customized or unowned service preserved: {unit}')

    def binding_migrations(self):
        try:
            from .managed_integrations import stable_hyprland_content
        except ImportError:
            from managed_integrations import stable_hyprland_content
        changes = {}
        candidates = {name: entry for name, entry in self.read_receipt().get('files', {}).items()
                      if entry.get('kind') in (None, 'integration') and Path(name).suffix == '.conf'}
        config = self.config_home() / 'hypr'
        for filename in ('bindings.conf', 'hyprland.conf'):
            candidates.setdefault(str(config / filename), None)
        for path in self.hyprland_files():
            candidates.setdefault(str(path), None)
        for name, entry in candidates.items():
            path = Path(name)
            if not path.is_file():
                continue
            try:
                content = path.read_text(encoding='utf-8')
            except (OSError, UnicodeError) as exc:
                print(f'Skipping unreadable binding file: {path}: {exc}')
                continue
            if entry and digest(path) != entry.get('sha256') and str(self.data / 'src/config/hyprland/hyprwhspr-tray.sh') not in content:
                continue
            updated = stable_hyprland_content(content, self.data, [self.data / 'src'])
            if updated != content:
                changes[path.resolve()] = updated
        return changes

    def install_launcher(self, root):
        # A fixed shell resolver survives missing CLI/backend environments.
        source = root / 'scripts/managed-launcher.sh'
        target = self.data / 'launcher'
        content = source.read_bytes()
        if target.exists() or target.is_symlink():
            previous = self.read_receipt().get('files', {}).get(str(target), {})
            if target.is_symlink() or previous.get('sha256') != digest(target):
                raise RuntimeError(f'Modified or unowned launcher preserved: {target}')
        temporary = target.with_name('.launcher.tmp')
        temporary.write_bytes(content)
        temporary.chmod(0o755)
        os.replace(temporary, target)
        generation = self.generation_state() or None
        atomic_text(self.data / 'interpreter', generation['python']['path'] + '\n')
        command = self.command_path()
        command.parent.mkdir(parents=True, exist_ok=True)
        if not command.is_symlink():
            command.symlink_to(target)
        elif Path(os.readlink(command)) != target:
            command.unlink()
            command.symlink_to(target)
        with self.receipts() as receipt:
            receipt.setdefault('files', {})[str(command)] = {'target': str(target)}
            receipt['files'][str(target)] = {'sha256': digest(target)}
            receipt['files'][str(self.data / 'interpreter')] = {'sha256': digest(self.data / 'interpreter')}


def launch(args):
    install = Installation()
    resolved = os.environ.pop('HYPRWHSPR_RESOLVED_GENERATION', None)
    if args and (args[0] == 'update' or args[:2] in (['install', 'repair'], ['install', 'status'])):
        return main(args)
    if args and args[0] == 'uninstall':
        return uninstall(args[1:])
    if args[:2] in (['backend', 'repair'], ['backend', 'reset']):
        install.update(force_backend=True, local_payload=True)
        return 0
    if args and args[0] == 'state' and any(x in args for x in ('reset', '--all')):
        raise RuntimeError('Managed dependency state belongs to a generation. Use install repair.')
    lifecycle_mutation = args and args[0] in ('setup', 'backend', 'state', 'install', 'systemd', 'waybar', 'noctalia', 'model', 'config')
    if lifecycle_mutation:
        with install.lock():
            install.recover()
            # Recovery may restore a different pointer and remove the generation
            # the shell resolver observed. Resolve all child paths under the lock.
            return _launch_generation(args, install.generation_state(), install.lock_fd)
    generation = json.loads(resolved) if resolved else install.generation_state()
    return _launch_generation(args, generation)


def _launch_generation(args, generation, lock_fd=None):
    if not generation:
        raise RuntimeError('No managed generation. Run the bootstrap installer.')
    env = clean_env()
    pinned = (generation.get('xdg') or {}).get('config')
    if pinned:
        env['XDG_CONFIG_HOME'] = pinned
    if args[:1] == ['--managed-tray']:
        env.update(HYPRWHSPR_ROOT=generation['root'],
                   HYPRWHSPR_BACKEND_ENV=(generation.get('backend') or generation['cli'])['path'])
        script = Path(generation['root']) / 'config/hyprland/hyprwhspr-tray.sh'
        os.execve('/bin/bash', ['/bin/bash', str(script), *args[1:]], env)
    root = Path(generation['root'])
    kind = 'backend' if not args or args[0] in ('test', 'transcribe') else 'cli'
    env_spec = generation.get(kind)
    if not env_spec:
        raise RuntimeError('No transcription environment. Run hyprwhspr setup.')
    env.update(HYPRWHSPR_ROOT=str(root), HYPRWHSPR_GENERATION=json.dumps(generation),
               HYPRWHSPR_BACKEND_ENV=(generation.get('backend') or generation['cli'])['path'],
               PYTHONDONTWRITEBYTECODE='1')
    command = [str(Path(env_spec['path']) / 'bin/python'), '-s', str(root / 'lib/cli.py' if args else root / 'lib/main.py'), *args]
    if lock_fd is not None:
        env['HYPRWHSPR_INSTALL_LOCK_FD'] = str(lock_fd)
        return subprocess.run(command, env=env, pass_fds=(lock_fd,)).returncode
    os.execve(command[0], command, env)


def service_content(root, launcher, xdg=None):
    """Render the unit. XDG values come from the generation, never the caller's shell.

    They are pinned at first install (Installation.xdg_roots) so re-rendering from
    a different session cannot repoint the daemon's directories; passing xdg=None
    falls back to the active generation.
    """
    escaped = str(launcher).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%')
    content = (Path(root) / 'config/systemd/hyprwhspr.service').read_text(encoding='utf-8')
    content = content.replace('ExecStart=/usr/lib/hyprwhspr/bin/hyprwhspr', f'ExecStart="{escaped}"')
    content = content.replace('Environment=HYPRWHSPR_ROOT=/usr/lib/hyprwhspr\n', '')
    install = Installation()
    if xdg is None:
        xdg = install.generation_state().get('xdg') or install.xdg_roots()
    # Data and state are wherever this installation actually lives, which is what
    # the launcher uses to find it; only the config root is pinned.
    homes = {'XDG_DATA_HOME': str(Path(launcher).parent.parent),
             'XDG_STATE_HOME': str(install.state.parent),
             'XDG_CONFIG_HOME': xdg['config']}
    for key, value in homes.items():
        value = str(value).replace('\\', '\\\\').replace('"', '\\"').replace('%', '%%')
        content = content.replace('[Service]\n', '[Service]\nEnvironment="' + key + '=' + value + '"\n')
    return content


def record_permission(entry):
    install = Installation()
    with install.receipts() as receipt:
        permissions = receipt.setdefault('permissions', [])
        if entry not in permissions:
            permissions.append(entry)


def file_receipt(path, kind):
    return ({'target': os.readlink(path), 'kind': kind} if path.is_symlink()
            else {'sha256': digest(path), 'kind': kind})


def record_file(path, kind='integration'):
    install = Installation()
    path = Path(path)
    with install.receipts() as receipt:
        receipt.setdefault('files', {})[str(path)] = file_receipt(path, kind)


def record_ownership(kind, *args, **kwargs):
    """Record a file or permission, tolerating any receipt failure.

    The ownership receipt is shared: cli/uninstall.py and managed uninstall both
    read it, so every install flavour records into it. Recording is bookkeeping,
    never the point of the operation, so a failure here must not turn a completed
    install step into a reported failure. Callers used to hand-roll this import
    and except clause, which repeatedly drifted out of sync. Returns False and
    leaves the cause on .last_error so a caller can report it in its own voice.
    """
    record_ownership.last_error = None
    try:
        (record_permission if kind == 'permission' else record_file)(*args, **kwargs)
        return True
    except (OSError, ValueError, TypeError, ImportError) as exc:
        record_ownership.last_error = exc
        return False


record_ownership.last_error = None


def forget_receipt_entry(install, name, entry):
    with install.receipts() as receipt:
        if receipt.get('files', {}).get(name) == entry:
            del receipt['files'][name]


def write_owned(path, content):
    install = Installation()
    path = Path(path)
    with install.receipts() as receipt:
        previous = receipt.get('files', {}).get(str(path))
        if path.exists() or path.is_symlink():
            if path.is_symlink() or not previous or previous.get('sha256') != digest(path):
                raise RuntimeError(f'Modified or unowned integration preserved: {path}')
        path.parent.mkdir(parents=True, exist_ok=True)
        atomic_text(path, content, path.stat().st_mode & 0o777 if path.exists() else 0o644)
        receipt.setdefault('files', {})[str(path)] = file_receipt(path, 'integration')


def uninstall(argv):
    parser = argparse.ArgumentParser(description='Remove owned application components; preserve personal data by default')
    parser.add_argument('--purge', action='store_true')
    parser.add_argument('--keep-models', action='store_true')
    parser.add_argument('--remove-permissions', action='store_true')
    parser.add_argument('--skip-permissions', action='store_true')
    parser.add_argument('--yes', action='store_true')
    args = parser.parse_args(argv)
    install = Installation()
    with install.lock():
        install.recover()
        generation = install.generation_state() or None
        receipt = install.read_receipt()
        if not args.yes and input('Remove owned hyprwhspr components? [y/N] ').lower() != 'y':
            return 0
        try:
            install.check_integrations()
        except (RuntimeError, OSError) as exc:
            print(f'Preserving conflicting integration while continuing uninstall: {exc}')
        keep_runtime = False
        state = install.service_state()
        fragment = state.get('FragmentPath')
        if fragment and (Path(fragment).exists() or Path(fragment).is_symlink()):
            try:
                keep_runtime = Path(fragment).is_symlink() or digest(fragment) != receipt.get('files', {}).get(str(fragment), {}).get('sha256')
            except OSError:
                keep_runtime = True
        running = state.get('ActiveState') in ('active', 'activating')
        if install.daemon_running() and not running:
            raise RuntimeError('Stop the manually launched daemon before uninstall')
        if running:
            install.service('stop')
            if install.daemon_running():
                raise RuntimeError('Daemon did not stop gracefully; runtime and entrypoints preserved')
        failures = []
        reload_marker = install.state / 'systemd-reload-pending.json'
        entrypoints = {str(install.command_path()), str(install.data / 'launcher'), str(install.data / 'interpreter')}
        entrypoint_files = [(name, entry) for name, entry in receipt.get('files', {}).items() if name in entrypoints]
        for name, entry in sorted(((name, entry) for name, entry in receipt.get('files', {}).items() if name not in entrypoints),
                                  key=lambda item: Path(item[0]).name.startswith('hyprwhspr-')):
            path = Path(name)
            kind = entry.get('kind', 'integration')
            if kind in ('personal', 'model') and (not args.purge or (kind == 'model' and args.keep_models)):
                print(f'Preserved personal data: {path}')
                continue
            try:
                if path.exists() or path.is_symlink():
                    if 'original' in entry:
                        try:
                            from .managed_integrations import remove_entry
                        except ImportError:
                            from managed_integrations import remove_entry
                        if remove_entry(path, entry):
                            forget_receipt_entry(install, name, entry)
                            print(f'Removed owned integration changes: {path}')
                        else:
                            print(f'Preserved modified integration: {path}')
                        continue
                    matches = (path.is_symlink() and os.readlink(path) == entry.get('target')) if 'target' in entry else (not path.is_symlink() and digest(path) == entry.get('sha256'))
                    if not matches:
                        print(f'Preserved modified file: {path}')
                        if path.name in ('hyprwhspr.service', 'hyprwhspr-resume.service'):
                            keep_runtime = True
                        continue
                    if path.name == 'hyprwhspr.service':
                        install.service('disable')
                        atomic_json(reload_marker, True)
                    path.unlink()
                    print(f'Removed: {path}')
                forget_receipt_entry(install, name, entry)
            except (OSError, subprocess.SubprocessError) as exc:
                failures.append(f'{path}: {exc}')
        if reload_marker.exists():
            try:
                run(['systemctl', '--user', 'daemon-reload'], capture_output=True, timeout=30)
                reload_marker.unlink()
            except (OSError, subprocess.SubprocessError) as exc:
                failures.append(f'systemd daemon-reload failed; retry uninstall: {exc}')
        if args.remove_permissions and not args.skip_permissions:
            for permission in receipt.get('permissions', []):
                if permission.get('adopted_legacy'):
                    print(f'Preserved pre-existing permission rule: {permission.get("path")}')
                    continue
                if not permission.get('added'):
                    continue
                try:
                    if permission['kind'] == 'group':
                        remove_added_group(permission['user'], permission['group'],
                            lambda: run(['sudo', 'gpasswd', '-d', permission['user'], permission['group']]))
                    elif permission['kind'] == 'rule':
                        path = Path(permission['path'])
                        if path.is_symlink() or (path.exists() and digest(path) != permission['sha256']):
                            print(f'Preserved modified permission rule: {path}; ownership retained for retry')
                            continue
                        if path.exists():
                            run(['sudo', 'rm', str(path)])
                        # Also retry reload after an earlier successful unlink.
                        run(['sudo', 'udevadm', 'control', '--reload-rules'])
                    else:
                        continue
                    with install.receipts() as latest:
                        for recorded in latest.get('permissions', []):
                            if recorded == permission:
                                recorded['added'] = False
                except (OSError, subprocess.SubprocessError) as exc:
                    failures.append(str(exc))
        if failures:
            raise RuntimeError('Uninstall incomplete: ' + '; '.join(failures))
        if keep_runtime:
            print('Preserved managed runtime and launcher for the customized service; reconcile the unit and retry uninstall.')
        else:
            # All payload helpers needed after this point are already loaded.
            # Save recovery sources in memory in case final filesystem cleanup fails.
            recovery_sources = {}
            if generation:
                for filename in ('managed_install.py', 'managed_integrations.py', 'legacy_units.py'):
                    helper = Path(generation['root']) / 'lib/src' / filename
                    if helper.is_file():
                        recovery_sources[helper] = helper.read_text(encoding='utf-8')
            removed = []
            try:
                for name, entry in entrypoint_files:
                    path = Path(name)
                    if not path.exists() and not path.is_symlink():
                        continue
                    matches = (path.is_symlink() and os.readlink(path) == entry.get('target')) if 'target' in entry else (not path.is_symlink() and digest(path) == entry.get('sha256'))
                    if not matches:
                        print(f'Preserved modified command/launcher: {path}')
                        continue
                    snapshot = {'target': os.readlink(path)} if path.is_symlink() else {'content': path.read_text(encoding='utf-8'), 'mode': path.stat().st_mode & 0o777}
                    path.unlink()
                    removed.append((path, snapshot))
                if generation:
                    paths = [generation['root'], generation['cli']['path']]
                    if generation.get('backend'):
                        paths.append(generation['backend']['path'])
                    atomic_json(install.journal, {'phase': 'committed', 'garbage': paths, 'created': [], 'old': generation})
                    install.recover(strict=True, final_uninstall=True)
                for name, entry in entrypoint_files:
                    if not Path(name).exists() and not Path(name).is_symlink():
                        forget_receipt_entry(install, name, entry)
                install.current.unlink(missing_ok=True)
            except (OSError, CleanupError):
                for helper, content in recovery_sources.items():
                    if not helper.is_file():
                        atomic_text(helper, content)
                for path, snapshot in removed:
                    if 'target' in snapshot:
                        path.symlink_to(snapshot['target'])
                    else:
                        atomic_text(path, snapshot['content'], snapshot['mode'])
                raise
        print('Settings, credentials, unrecorded models and shared caches are preserved. Ownership receipt retained for auditing.')
    return 0


def main(argv=None):
    parser = argparse.ArgumentParser(description='hyprwhspr release installation management')
    parser.add_argument('command', choices=['update', 'install', 'bootstrap'])
    parser.add_argument('action', nargs='?', choices=['repair', 'status'])
    parser.add_argument('--version')
    parser.add_argument('--python')
    parser.add_argument('--repair', action='store_true')
    args = parser.parse_args(argv)
    install = Installation()
    if args.command == 'install' and args.action == 'status':
        print(json.dumps(install.status(), indent=2))
        return 0
    if args.command == 'install' and args.action != 'repair':
        parser.error('install requires repair or status')
    if args.command != 'bootstrap' and not install.current.exists():
        raise RuntimeError('Not a managed release installation. Use your package manager or bootstrap installer.')
    fresh = not install.current.exists() and not (install.data / 'src').exists()
    generation = install.update(args.version, args.python, args.repair or args.action == 'repair')
    # A migration that could not determine the legacy backend leaves no transcription
    # environment. Offer setup there too, or the service starts only to fail. The
    # update itself already succeeded, so declining the wizard is not a failed update.
    offered = not fresh and not (generation or {}).get('backend')
    if offered:
        print('No transcription environment was carried over from the previous installation.')
    if fresh or offered:
        try:
            terminal = open('/dev/tty')
        except OSError:
            print('Installation completed. Run hyprwhspr setup from an interactive terminal to configure dictation.')
            return 0
        with terminal:
            result = subprocess.call([str(install.data / 'launcher'), 'setup'], stdin=terminal, env=clean_env())
        if result:
            print('Application installed; setup did not complete. Run hyprwhspr setup to resume.')
        # The install or update itself succeeded; declining the wizard is not a failure.
        return 0
    return 0


if __name__ == '__main__':
    try:
        raise SystemExit(launch(sys.argv[2:]) if sys.argv[1:2] == ['launch'] else main())
    except (Exception, KeyboardInterrupt) as exc:
        print(f'Installation incomplete: {exc}', file=sys.stderr)
        raise SystemExit(1)
