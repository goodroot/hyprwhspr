"""Pinned Orukeet model files for the optional onnx-asr backend."""

import hashlib
import json
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


def sha256(path: Path) -> str:
    """Hash a model without reading all its weights into memory."""
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_model(cache_dir: str | None = None, *, offline: bool = False) -> Path:
    """Return a verified Hub snapshot directory, reusing cached files first."""

    from huggingface_hub import hf_hub_download
    from huggingface_hub.errors import LocalEntryNotFoundError

    def fetch(filename: str) -> Path:
        kwargs = {
            "repo_id": REPO_ID,
            "revision": REVISION,
            "filename": f"{SUBFOLDER}/{filename}",
            "cache_dir": cache_dir,
        }
        try:
            return Path(hf_hub_download(**kwargs, local_files_only=True))
        except LocalEntryNotFoundError:
            if offline:
                raise
            return Path(hf_hub_download(**kwargs))

    manifest_path = fetch("manifest.json")
    if sha256(manifest_path) != MANIFEST_SHA256:
        msg = "Orukeet release manifest checksum mismatch"
        raise ValueError(msg)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    for filename in FILES:
        expected = manifest["files"][filename]
        path = fetch(filename)
        if path.stat().st_size != expected["bytes"] or sha256(path) != expected["sha256"]:
            msg = f"Orukeet checksum mismatch: {filename}"
            raise ValueError(msg)
    return manifest_path.parent

