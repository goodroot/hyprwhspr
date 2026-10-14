"""Installation ownership shared by the launcher-facing CLI and runtime."""

import os
import shlex
from pathlib import Path

NIX_REPAIR_HINT = 'Rebuild the hyprwhspr Nix package and restart the service.'
NIX_TRAY_COMMAND = 'hyprwhspr-tray'


def is_nix_install() -> bool:
    return os.environ.get('HYPRWHSPR_INSTALL_METHOD') == 'nix'


def is_nixos() -> bool:
    return Path('/etc/NIXOS').exists()


def input_group_hint() -> str:
    """How to join the input group; NixOS discards usermod changes on rebuild."""
    if is_nixos():
        return 'add your user to services.hyprwhspr.users and rebuild NixOS'
    return 'sudo usermod -aG input $USER'


def repair_hint(default: str) -> str:
    """The repair advice for this installation; Nix rebuilds instead of running setup."""
    return NIX_REPAIR_HINT if is_nix_install() else default


def is_supported_backend(backend: str) -> bool:
    if not is_nix_install():
        return True
    # Imported here: every CLI call, including the hotkey path, imports this module.
    try:
        from .backend_utils import normalize_backend
    except ImportError:
        from backend_utils import normalize_backend
    return normalize_backend(backend) in ('cpu', 'pywhispercpp')


def nix_profile_command(name: str, root=None):
    """This package's `name` as a stable PATH entry (a profile link), or None.

    Store paths vanish after garbage collection; profile links follow updates.
    Only a link resolving to this package counts, so another install earlier on
    PATH is skipped. `root` is the package's lib/hyprwhspr directory.
    """
    root = root or os.environ.get('HYPRWHSPR_ROOT')
    if not root:
        return None
    target = os.path.realpath(Path(root).parent.parent / 'bin' / name)
    for directory in os.environ.get('PATH', '').split(os.pathsep):
        candidate = os.path.join(directory, name)
        if (os.path.isabs(candidate) and not candidate.startswith('/nix/store/')
                and os.access(candidate, os.X_OK) and os.path.realpath(candidate) == target):
            return candidate
    return None


def tray_command(root):
    """Shell command for the tray script that survives updates, or None if it has none."""
    if os.environ.get('HYPRWHSPR_GENERATION'):
        try:
            from .managed_install import Installation
        except ImportError:
            from managed_install import Installation
        return shlex.quote(str(Installation().data / 'launcher')) + ' --managed-tray'
    if is_nix_install():
        # Absolute, because bars started by systemd lack the profile on PATH.
        command = nix_profile_command(NIX_TRAY_COMMAND, root)
        if command is None:
            try:
                from .output_control import log_error
            except ImportError:
                from output_control import log_error
            log_error(f'{NIX_TRAY_COMMAND} is not on PATH through a Nix profile. '
                      'Install hyprwhspr into a profile, then rerun this command.')
            return None
        return shlex.quote(command)
    return shlex.quote(str(Path(root) / 'config/hyprland/hyprwhspr-tray.sh'))


def require_supported_backend(backend: str):
    """Reject an unavailable Nix backend before loading models or audio devices."""
    if not is_supported_backend(backend):
        raise ValueError(
            f"The Nix package supports only Whisper.cpp CPU transcription, not {backend!r}. "
            'Run hyprwhspr config edit and set transcription_backend to "cpu". '
            'Your configuration has not been changed.'
        )


def nix_command_error(args):
    """Return guidance for commands that conflict with Nix ownership."""
    if not is_nix_install():
        return None
    command = args.command
    if command == 'setup':
        return ('Nix manages installation. Run hyprwhspr config init, '
                'hyprwhspr config edit, and hyprwhspr model download base. '
                'Configure the service and input permissions through the NixOS module, '
                'or run hyprwhspr systemd install on other distributions.')
    if command in ('update', 'uninstall', 'backend') or (
            command == 'install' and getattr(args, 'install_action', None) != 'status') or (
            command == 'state' and getattr(args, 'state_action', None) == 'reset'):
        return ('Nix manages this package and its dependencies. Update or remove it through '
                'your Nix configuration or profile, then rebuild; do not use the managed installer.')
    if (command == 'systemd' and getattr(args, 'systemd_action', None) in ('install', 'enable', 'disable')
            and is_nixos()):
        return ('NixOS manages the service. Change services.hyprwhspr in your NixOS '
                'configuration and rebuild. Service status and restart remain available.')
    return None
