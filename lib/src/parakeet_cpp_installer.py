"""Staged, checksum-verified installation of the optional Parakeet.cpp payload."""

import os
from pathlib import Path
import shutil
import tarfile
import tempfile

try:
    from . import parakeet_cpp_runtime as runtime
    from .output_control import log_info, log_warning, log_error
except ImportError:
    import parakeet_cpp_runtime as runtime
    from output_control import log_info, log_warning, log_error


def _primitives():
    # Lazy to avoid an installer import cycle and keep the runtime stdlib-only.
    try:
        from .backend_installer import _download_bounded_file, _file_matches
    except ImportError:
        from backend_installer import _download_bounded_file, _file_matches
    return _download_bounded_file, _file_matches


def _record(path, kind):
    try:
        from .managed_install import record_ownership
    except ImportError:
        from managed_install import record_ownership
    if not record_ownership('file', path, kind):
        log_warning(f'Installed but ownership could not be recorded: {path}: {record_ownership.last_error}')


def extract_runtime(archive, destination, root):
    """Accept exactly the four flat regular files in upstream CPU/Vulkan bundles."""
    seen = set()
    with tarfile.open(archive, 'r:gz') as bundle:
        for member in bundle.getmembers():
            if member.name.rstrip('/') == root and member.isdir():
                continue
            prefix = root + '/'
            if not member.name.startswith(prefix):
                raise RuntimeError(f'Unsafe archive member: {member.name}')
            name = member.name[len(prefix):]
            if (name not in runtime.ARCHIVE_FILES or not member.isfile()
                    or name in seen or member.size > 128 * 1024 * 1024):
                raise RuntimeError(f'Unexpected archive member: {member.name}')
            seen.add(name)
            with bundle.extractfile(member) as source, (destination / name).open('xb') as target:
                shutil.copyfileobj(source, target)
    if seen != set(runtime.ARCHIVE_FILES):
        raise RuntimeError('Incomplete Parakeet.cpp runtime archive')


def install_runtime(device, force=False):
    download, matches = _primitives()
    filename, size, digest = runtime.ASSETS[(runtime.architecture(), device)]
    target = runtime.library_path(device).parent
    if not force and target.is_dir() and runtime.probe_library(target / 'libparakeet.so')[0]:
        return
    target.parent.mkdir(parents=True, exist_ok=True)
    # Same-filesystem staging; the old runtime remains usable through download,
    # checksum, extraction, and load probe. Roll back even on KeyboardInterrupt.
    with tempfile.TemporaryDirectory(prefix='.parakeet-', dir=target.parent) as temporary:
        staging = Path(temporary)
        archive = staging / filename
        download(f'{runtime.BASE_URL}/{filename}', archive, size)
        if not matches(archive, size, digest):
            raise RuntimeError(f'{filename} failed pinned size/SHA-256 validation')
        payload = staging / 'payload'
        payload.mkdir()
        extract_runtime(archive, payload, filename.removesuffix('.tar.gz'))
        compatible, detail = runtime.probe_library(payload / 'libparakeet.so')
        if not compatible:
            raise RuntimeError(f'{device} library cannot load on this system: {detail}')
        previous = staging / 'previous'
        if target.exists():
            target.replace(previous)
        try:
            payload.replace(target)
        except BaseException:
            if previous.exists():
                previous.replace(target)
            raise
    for name in runtime.ARCHIVE_FILES:
        _record(target / name, 'runtime')


def download_model(model=runtime.MODEL_ID):
    if model != runtime.MODEL_ID:
        log_error(f'Unsupported Parakeet.cpp model: {model}; use {runtime.MODEL_ID}')
        return False
    download, matches = _primitives()
    metadata = runtime.MODEL
    target = runtime.model_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        if not matches(target, metadata['size'], metadata['sha256']):
            with tempfile.TemporaryDirectory(prefix='.parakeet-', dir=target.parent) as temporary:
                partial = Path(temporary) / metadata['filename']
                url = (f"https://huggingface.co/{metadata['repo']}/resolve/"
                       f"{metadata['revision']}/{metadata['filename']}")
                log_info(f'Downloading Parakeet.cpp {model} (~941 MB)')
                download(url, partial, metadata['size'])
                if not matches(partial, metadata['size'], metadata['sha256']):
                    raise RuntimeError('Model failed pinned size/SHA-256 validation')
                partial.replace(target)
        _record(target, 'model')
        return True
    except Exception as exc:
        log_error(f'Parakeet.cpp model installation failed: {exc}')
        return False


def install_payload(config, force=False):
    try:
        device = runtime.resolve_device(config, force_probe=force)
        try:
            install_runtime(device, force)
        except Exception as exc:
            if device != 'vulkan' or config.get_setting('parakeet_cpp_device', 'auto') != 'auto':
                raise
            log_warning(f'Vulkan runtime unavailable: {exc}. Installing CPU runtime instead.')
            device = 'cpu'
            install_runtime(device, force)
        if not download_model():
            return False
        selected = runtime.selection_path()
        with tempfile.NamedTemporaryFile(mode='w', dir=selected.parent, delete=False) as output:
            temporary = Path(output.name)
            output.write(device + '\n')
        try:
            temporary.replace(selected)
        finally:
            temporary.unlink(missing_ok=True)
        _record(selected, 'runtime')
        return True
    except Exception as exc:
        log_error(f'Parakeet.cpp installation failed: {exc}. Install the required system '
                  'libraries/Vulkan driver, or set parakeet_cpp_device to cpu and rerun setup.')
        return False
