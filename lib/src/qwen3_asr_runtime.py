"""Pinned runtime and model contract for the Qwen3-ASR backend."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import Optional

try:
    from .paths import DATA_DIR, RUNTIME_DIR
except ImportError:
    from paths import DATA_DIR, RUNTIME_DIR

try:
    from .backend_utils import vulkaninfo_has_hardware_gpu
except ImportError:
    from backend_utils import vulkaninfo_has_hardware_gpu


LLAMA_CPP_RELEASE = "b10516"
LLAMA_CPP_BASE_URL = (
    "https://github.com/ggml-org/llama.cpp/releases/download/" + LLAMA_CPP_RELEASE
)

QWEN3_ASR_MODELS_DIR = DATA_DIR / "qwen3-asr" / "models"
# Bundled binaries live under DATA_DIR/runtime so `hyprwhspr uninstall` removes
# them with the other optional runtimes (cli/uninstall.py clears that subtree);
# models stay outside it so --keep-models can preserve them.
QWEN3_ASR_RUNTIMES_DIR = DATA_DIR / "runtime" / "llama-cpp"
QWEN3_ASR_SOCKET = RUNTIME_DIR / "qwen3-asr.sock"
QWEN3_ASR_LOG = RUNTIME_DIR / "qwen3-asr-server.log"
QWEN3_ASR_MAX_AUDIO_SECONDS = 120

# Devices we can actually install. Upstream publishes no Linux CUDA or ROCm
# build (those assets are Windows-only), and Vulkan covers NVIDIA, AMD and
# Intel, so the accelerated tier is Vulkan or nothing.
QWEN3_ASR_DEVICES = ("cpu", "vulkan")

# Order `auto` picks between runtimes that are already installed, accelerated
# first. Kept separate from QWEN3_ASR_DEVICES on purpose: reusing that tuple as
# both the membership set and the preference order silently selected CPU on a
# machine that had both runtimes on disk.
DEVICE_PREFERENCE = ("vulkan", "cpu")

# Pinned llama.cpp release assets. Sizes and digests are immutable upstream
# identities, kept beside the filenames so the installer consumes one contract.
LLAMA_CPP_ASSETS = {
    "cpu": ("llama-b10516-bin-ubuntu-x64.tar.gz", 16667775,
            "f263a91280471b4c33c4999d7c76259c0f3a0a53a0b3e692b2c0b84380137a35"),
    "vulkan": ("llama-b10516-bin-ubuntu-vulkan-x64.tar.gz", 33289144,
               "5ce186720f43c415465869b0cd93973b828b219cbf6fbcc22aa899531973c505"),
}

# Every asset above unpacks to this single top-level directory, flat: the
# binaries sit beside their shared objects with no bin/ or lib/ split.
LLAMA_CPP_ARCHIVE_ROOT = "llama-" + LLAMA_CPP_RELEASE

# Revisions and digests are immutable upstream LFS identities.  Keeping them
# beside sizes and filenames lets setup/model/status consume one contract.
QWEN3_ASR_MODELS = {
    "1.7b-q8_0": {
        "repo": "ggml-org/Qwen3-ASR-1.7B-GGUF",
        "revision": "36a678687ba7d07a74ca70ccb0e36902e005fb80",
        "decoder": ("Qwen3-ASR-1.7B-Q8_0.gguf", 2165034944,
                    "58e22d0532d4eacaf034cfac17a6fed159f37c41390c710186783be439d1fc57"),
        "projector": ("mmproj-Qwen3-ASR-1.7B-Q8_0.gguf", 355709344,
                      "46c1d533af3f354ceb37ce855dbceff7da7fa7cf1e6a523df3b13440bd164c0d"),
    },
    "0.6b-q8_0": {
        "repo": "ggml-org/Qwen3-ASR-0.6B-GGUF",
        # Pinned snapshot advertised by llama.cpp's model resolver.
        "revision": "928ab958557df9aa2ef1c93e0e83c7ad0933fae2",
        "decoder": ("Qwen3-ASR-0.6B-Q8_0.gguf", 804749248,
                    "bca259818b50ca7c4c05e9bdb35a5dc04fa039653a6d6f3f0f331f96f6aa1971"),
        "projector": ("mmproj-Qwen3-ASR-0.6B-Q8_0.gguf", 214392480,
                      "41a342b5e4c514e968cb756de6cd1b7be39eff43c44c57a2ef5fc6522e36603d"),
    },
}

DEFAULT_MODEL = "1.7b-q8_0"


def model_paths(model_name: str) -> tuple[Path, Path]:
    metadata = QWEN3_ASR_MODELS[model_name]
    return (QWEN3_ASR_MODELS_DIR / metadata["decoder"][0],
            QWEN3_ASR_MODELS_DIR / metadata["projector"][0])


def runtime_dir(device: str) -> Path:
    return QWEN3_ASR_RUNTIMES_DIR / LLAMA_CPP_RELEASE / device


def server_path(device: str) -> Path:
    """The llama-server binary. The archive is flat, so there is no bin/."""
    return runtime_dir(device) / "llama-server"


def library_dir(device: str) -> Path:
    """Shared objects sit beside the binaries in the same flat directory."""
    return runtime_dir(device)


def runtime_installed(device: str) -> bool:
    server = server_path(device)
    return server.is_file() and os.access(server, os.X_OK)


def resolve_device(config, force_probe: bool = False) -> str:
    """Pick the llama.cpp runtime variant, preferring one already installed.

    Single source of truth: the backend and install verification must agree, or
    verification passes against a runtime the backend will never launch.

    `force_probe` re-reads the hardware instead of settling for what is on disk;
    setup passes it on an explicit reinstall so a machine that has gained a GPU
    can move from the CPU runtime to Vulkan.
    """
    requested = config.get_setting("qwen3_asr_device", "auto")
    if requested in QWEN3_ASR_DEVICES:
        return requested
    if force_probe:
        return detect_device()

    # An installed runtime wins over a fresh probe so a machine whose GPU comes
    # and goes keeps using whichever variant is on disk — accelerated first.
    for device in DEVICE_PREFERENCE:
        if runtime_installed(device):
            return device
    return detect_device()


def detect_device() -> str:
    """Probe for a usable hardware GPU, ignoring what is already installed."""
    if not shutil.which("vulkaninfo"):
        return "cpu"
    try:
        result = subprocess.run(["vulkaninfo", "--summary"], capture_output=True,
                                text=True, timeout=5, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return "cpu"
    if result.returncode == 0 and vulkaninfo_has_hardware_gpu(result.stdout):
        return "vulkan"
    return "cpu"


def model_installed(model_name: str) -> bool:
    """Whether both halves of a pinned pair are present at their pinned sizes.

    Size only: this runs on every status/detect call, and re-hashing 2.4 GB to
    answer "is it there?" would make those commands unusable. The installer
    verifies SHA-256 before the file is ever put in place.
    """
    metadata = QWEN3_ASR_MODELS.get(model_name)
    if metadata is None:
        return False
    decoder, projector = model_paths(model_name)
    return all(
        path.is_file() and path.stat().st_size == metadata[role][1]
        for role, path in (("decoder", decoder), ("projector", projector))
    )


def is_installed(model_name: Optional[str] = None, device: Optional[str] = None,
                 config=None) -> bool:
    """Whether the sidecar binary and the selected model pair are both present."""
    if config is not None:
        model_name = model_name or config.get_setting("qwen3_asr_model", DEFAULT_MODEL)
        device = device or resolve_device(config)
    model_name = model_name or DEFAULT_MODEL
    if device is None:
        device = next((d for d in QWEN3_ASR_DEVICES if runtime_installed(d)), "cpu")
    return runtime_installed(device) and model_installed(model_name)
