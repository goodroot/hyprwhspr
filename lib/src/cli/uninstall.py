"""
Uninstall command for hyprwhspr — removes services, integrations, user data
and optionally system permissions
"""

import os
import shutil
import subprocess
from pathlib import Path

from rich.prompt import Confirm

try:
    from ..backend_installer import (
        VENV_DIR, STATE_DIR, USER_BASE,
        PYWHISPERCPP_SRC_DIR, PYWHISPERCPP_MODELS_DIR
    )
except ImportError:
    from backend_installer import (
        VENV_DIR, STATE_DIR, USER_BASE,
        PYWHISPERCPP_SRC_DIR, PYWHISPERCPP_MODELS_DIR
    )

try:
    from ..credential_manager import CREDENTIALS_FILE
except ImportError:
    from credential_manager import CREDENTIALS_FILE

try:
    from ..output_control import (
        log_info, log_success, log_warning, run_command, run_sudo_command
    )
except ImportError:
    from output_control import (
        log_info, log_success, log_warning, run_command, run_sudo_command
    )

from ._shared import (SERVICE_NAME, RESUME_SERVICE_NAME, YDOTOOL_UNIT,
                      USER_HOME, USER_CONFIG_DIR, USER_SYSTEMD_DIR)
from .systemd import _is_hyprwhspr_managed_ydotool_unit



def _generated_legacy_unit(path):
    """Recognize exact generated content, not just a familiar unit filename."""
    from ._shared import HYPRWHSPR_ROOT
    try:
        from ..legacy_units import generated_unit
    except ImportError:
        from legacy_units import generated_unit
    return generated_unit(path, HYPRWHSPR_ROOT)


def uninstall_command(keep_models: bool = False, remove_permissions: bool = False,
                     skip_permissions: bool = False, yes: bool = False, purge: bool = False):
    """Remove recognized legacy components; report anything whose ownership is uncertain."""
    from ._shared import HYPRWHSPR_ROOT
    try:
        from ..managed_install import Installation, digest, forget_receipt_entry
    except ImportError:
        from managed_install import Installation, digest, forget_receipt_entry
    install = Installation(data=USER_BASE, state=STATE_DIR)
    receipt = install.read_receipt()
    units = []
    preserved_unit = False
    for name in (SERVICE_NAME, RESUME_SERVICE_NAME):
        path = USER_SYSTEMD_DIR / name
        if path.exists() or path.is_symlink():
            if path.is_symlink() and not path.exists():
                log_warning(f'Preserving dangling service link (no runtime dependency): {path}')
                continue
            entry = receipt.get('files', {}).get(str(path), {})
            try:
                recorded = not path.is_symlink() and entry.get('sha256') == digest(path)
            except OSError as exc:
                preserved_unit = True
                log_warning(f'Preserving unreadable service and its runtime: {path}: {exc}')
                continue
            if recorded or _generated_legacy_unit(path):
                units.append(path)
            else:
                preserved_unit = True
                log_warning(f'Preserving customized or unowned service: {path}')
    ydotool = USER_SYSTEMD_DIR / YDOTOOL_UNIT
    if _is_hyprwhspr_managed_ydotool_unit(ydotool):
        units.append(ydotool)

    targets = []
    if VENV_DIR.exists() and not VENV_DIR.is_symlink():
        if preserved_unit:
            log_warning(f'Preserving backend environment for the retained service: {VENV_DIR}')
        else:
            targets.append((f'Main backend venv: {VENV_DIR}', VENV_DIR))
    for path in (PYWHISPERCPP_SRC_DIR, USER_BASE / 'src'):
        if path.exists():
            log_warning(f'Preserving unrecorded source checkout: {path}')
    command = USER_HOME / '.local/bin/hyprwhspr'
    recognized = {Path(HYPRWHSPR_ROOT) / 'bin/hyprwhspr', USER_BASE / 'src/bin/hyprwhspr'}
    if command.is_symlink():
        target = (command.parent / os.readlink(command)).resolve()
        if target in {path.resolve() for path in recognized}:
            targets.append((f'Command symlink: {command}', command))
        else:
            log_warning(f'Preserving foreign command symlink: {command}')
    elif command.exists():
        log_warning(f'Preserving unowned command: {command}')
    if purge:
        for path in (USER_CONFIG_DIR / 'config.json', CREDENTIALS_FILE):
            if path.is_file() and not path.is_symlink():
                targets.append((f'Personal file: {path}', path))
    temp_dir = USER_BASE / 'temp'
    runtime_dir = USER_BASE / 'runtime'
    if not preserved_unit:
        for label, path in (('Temporary files', temp_dir), ('Optional GUI runtimes', runtime_dir)):
            if path.exists():
                targets.append((f'{label}: {path}', path))
    models = []
    if purge and not keep_models:
        for name, entry in receipt.get('files', {}).items():
            path = Path(name)
            if entry.get('kind') == 'model' and path.is_file() and not path.is_symlink():
                try:
                    matches = digest(path) == entry.get('sha256')
                except OSError as exc:
                    log_warning(f'Preserving unreadable model: {path}: {exc}')
                    continue
                if matches:
                    models.append((path, entry))
                else:
                    log_warning(f'Preserving modified model: {path}')
    if keep_models:
        log_info('Keeping models (--keep-models).')
    elif not purge:
        log_info('Keeping models and personal data (ordinary uninstall).')
    else:
        log_info('Purging recorded models only; unrecorded legacy models are preserved.')
    if skip_permissions:
        log_info('Skipping all permission removal (--skip-permissions).')
    elif remove_permissions:
        log_info('Removing only permissions recorded as added by installation; unrecorded legacy permissions are preserved.')
    else:
        log_info('Keeping system permissions; select --remove-permissions to remove recorded additions.')
    log_info('Preserving unrecorded legacy bar integrations and shared caches.')

    print('\nComponents to remove:')
    for path in units:
        print(f'  Systemd service: {path}')
    for label, _ in targets:
        print(f'  {label}')
    for path, _ in models:
        print(f'  Recorded model: {path}')
    if not yes and not Confirm.ask('Remove these hyprwhspr components?', default=False):
        return
    errors = []
    runtime_unsafe = False
    for path in units:
        try:
            for action in ('stop', 'disable'):
                run_command(['systemctl', '--user', action, path.name], check=True)
            path.unlink()
            log_success(f'Removed service: {path}')
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f'Could not remove {path.name}: {exc}; retry with a working user systemd bus')
            if path.name in (SERVICE_NAME, RESUME_SERVICE_NAME):
                runtime_unsafe = True
    if units:
        try:
            run_command(['systemctl', '--user', 'daemon-reload'], check=True)
        except (OSError, subprocess.SubprocessError) as exc:
            errors.append(f'systemd daemon-reload failed: {exc}')
            runtime_unsafe = True

    for label, path in targets:
        if runtime_unsafe and path in (VENV_DIR, temp_dir, runtime_dir):
            log_warning(f'Preserving runtime needed by service: {path}')
            continue
        try:
            if path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            log_success(f'Removed {label}')
        except OSError as exc:
            errors.append(f'{path}: {exc}')
    for path, entry in models:
        try:
            path.unlink()
            forget_receipt_entry(install, str(path), entry)
            log_success(f'Removed recorded model: {path}')
        except (OSError, ValueError, TypeError) as exc:
            errors.append(f'{path}: {exc}')
    if remove_permissions and not skip_permissions:
        for permission in receipt.get('permissions', []):
            if permission.get('adopted_legacy'):
                log_info(f'Preserving pre-existing permission rule: {permission.get("path")}')
                continue
            if not permission.get('added'):
                continue
            try:
                if permission['kind'] == 'group':
                    try:
                        from ..managed_install import remove_added_group
                    except ImportError:
                        from managed_install import remove_added_group
                    remove_added_group(permission['user'], permission['group'],
                        lambda: run_sudo_command(['gpasswd', '-d', permission['user'], permission['group']], check=True))
                elif permission['kind'] == 'rule':
                    path = Path(permission['path'])
                    if path.is_symlink() or (path.exists() and digest(path) != permission['sha256']):
                        log_warning(f'Preserving modified permission rule: {path}')
                        continue
                    if path.exists():
                        run_sudo_command(['rm', str(path)], check=True)
                    run_sudo_command(['udevadm', 'control', '--reload-rules'], check=True)
                else:
                    continue
                with install.receipts() as latest:
                    for entry in latest.get('permissions', []):
                        if entry == permission:
                            entry['added'] = False
                log_success('Removed recorded permission addition')
            except Exception as exc:
                errors.append(f'Permission removal: {exc}')
    if errors:
        for error in errors:
            log_warning(error)
        raise RuntimeError('Uninstall incomplete; runtime preserved where needed; see reported failures')
    log_success('Uninstall completed; preserved components are listed above.')
