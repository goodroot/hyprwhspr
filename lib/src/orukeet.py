"""Pinned Orukeet model files for the optional onnx-asr backend."""

import hashlib
import json
import os
from pathlib import Path

REPO_ID = "oruk/orukeet"
REVISION = "eac739d754bb171287930e6e63386f5b88f8179e"
SUBFOLDER = "onnx/combined-v0.1.0-int8"
MANIFEST_SHA256 = "77f9f8e9fadeddbbf98b1dfd9e51f90a8e109c795b1679d9d85e3a5e9ef205f4"
FILES = (
    "encoder-model.int8.onnx",
    "decoder_joint-model.int8.onnx",
    "vocab.txt",
    "config.json",
    "LICENSE-WEIGHTS",
    "LICENSE-CONVERTER.txt",
    "NOTICE.md",
)
# Records size/mtime of files that passed a full hash, so warm starts skip ~672 MB of hashing.
STAMP = ".hyprwhspr-verified.json"


def sha256(path: Path) -> str:
    """Hash a model without reading all its weights into memory."""
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def hub_cache_dir():
    """Match Hugging Face cache environment defaults without importing the Hub."""
    root = os.environ.get('HF_HUB_CACHE') or os.environ.get('HUGGINGFACE_HUB_CACHE')
    if root:
        return Path(root).expanduser()
    home = os.environ.get('HF_HOME') or str(Path(os.environ.get('XDG_CACHE_HOME', Path.home() / '.cache')) / 'huggingface')
    return Path(home).expanduser() / 'hub'


class ChecksumError(ValueError):
    """A cached Orukeet file does not match the pinned release."""


def _snapshot(cache_dir=None) -> Path:
    root = cache_dir or hub_cache_dir()
    return Path(root) / 'models--oruk--orukeet' / 'snapshots' / REVISION / SUBFOLDER


def _stamp(paths) -> dict:
    return {name: [path.stat().st_size, path.stat().st_mtime_ns] for name, path in paths.items()}


def cache_state(cache_dir: str | None = None) -> str:
    """Report 'verified', 'unverified' or 'missing' without hashing or writing."""
    directory = _snapshot(cache_dir)
    paths = {name: directory / name for name in FILES}
    if not (directory / "manifest.json").is_file() or not all(path.is_file() for path in paths.values()):
        return "missing"
    try:
        verified = json.loads((directory / STAMP).read_text(encoding="utf-8")) == _stamp(paths)
    except (OSError, ValueError):
        verified = False
    return "verified" if verified else "unverified"


def download_model(cache_dir: str | None = None, *, offline: bool = False, repair: bool = False) -> Path:
    """Return a verified Hub snapshot directory, reusing cached files first.

    Service starts fail closed on a corrupt file; explicit preparation passes
    repair=True to re-download just that file once.
    """

    # Offline status also works in the lightweight CLI environment.
    if not offline:
        from huggingface_hub import hf_hub_download
        from huggingface_hub.errors import LocalEntryNotFoundError

    def fetch(filename: str, force: bool = False) -> Path:
        if offline:
            path = _snapshot(cache_dir) / filename
            if not path.is_file():
                raise FileNotFoundError(f'Orukeet cache missing: {path}')
            return path
        kwargs = {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "filename": f"{SUBFOLDER}/{filename}",
            "cache_dir": cache_dir,
        }
        if force:
            return Path(hf_hub_download(**kwargs, force_download=True))
        try:
            return Path(hf_hub_download(**kwargs, local_files_only=True))
        except LocalEntryNotFoundError:
            return Path(hf_hub_download(**kwargs))

    def verified(filename, check, path=None):
        path = path or fetch(filename)
        if check(path):
            return path
        if repair and not offline:
            path = fetch(filename, force=True)
            if check(path):
                return path
        raise ChecksumError(f"Orukeet checksum mismatch: {filename}")

    manifest_path = verified("manifest.json", lambda path: sha256(path) == MANIFEST_SHA256)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    paths = {filename: fetch(filename) for filename in FILES}
    stamp_path = manifest_path.parent / STAMP
    try:
        if json.loads(stamp_path.read_text(encoding="utf-8")) == _stamp(paths):
            return manifest_path.parent
    except (OSError, ValueError):
        pass
    for filename in FILES:
        expected = manifest["files"][filename]
        paths[filename] = verified(filename, lambda path: path.stat().st_size == expected["bytes"]
                                   and sha256(path) == expected["sha256"], paths[filename])
    try:
        stamp_path.write_text(json.dumps(_stamp(paths)), encoding="utf-8")
    except OSError:
        pass  # A read-only cache just means verifying again next time.
    return manifest_path.parent
