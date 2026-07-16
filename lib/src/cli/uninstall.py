"""
Uninstall command for hyprwhspr — removes services, integrations, user data
and optionally system permissions
"""

import getpass
import os
import shutil
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
from .waybar import setup_waybar



# ==================== Uninstall Command ====================

def uninstall_command(keep_models: bool = False, remove_permissions: bool = False,
                     skip_permissions: bool = False, yes: bool = False):
    """Completely remove hyprwhspr and all user data"""
    print("\n" + "="*60)
    print("hyprwhspr Uninstall")
    print("="*60)
    
    # Build summary of what will be removed
    items_to_remove = []
    
    # Systemd services
    if (USER_SYSTEMD_DIR / SERVICE_NAME).exists():
        items_to_remove.append(f"Systemd service: {SERVICE_NAME}")
    if (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).exists():
        items_to_remove.append(f"Systemd service: {RESUME_SERVICE_NAME} (deprecated)")
    ydotool_unit_path = USER_SYSTEMD_DIR / YDOTOOL_UNIT
    if _is_hyprwhspr_managed_ydotool_unit(ydotool_unit_path):
        items_to_remove.append(f"Systemd service: {YDOTOOL_UNIT}")

    # Waybar integration
    waybar_module = USER_HOME / '.config' / 'waybar' / 'hyprwhspr-module.jsonc'
    if waybar_module.exists():
        items_to_remove.append("Waybar integration")
    
    # Plain filesystem targets: declared once here, listed in the summary and
    # removed in one pass below (dirs rmtree'd, files/symlinks unlinked)
    fs_targets = []
    if USER_CONFIG_DIR.exists():
        fs_targets.append((f"User configuration: {USER_CONFIG_DIR}", USER_CONFIG_DIR))
    if VENV_DIR.exists():
        fs_targets.append((f"Main backend venv: {VENV_DIR}", VENV_DIR))
    if PYWHISPERCPP_SRC_DIR.exists():
        fs_targets.append((f"pywhispercpp source: {PYWHISPERCPP_SRC_DIR}", PYWHISPERCPP_SRC_DIR))
    src_dir = USER_BASE / 'src'
    if (src_dir / '.git').exists():
        fs_targets.append((f"Managed clone: {src_dir}", src_dir))
    cmd_symlink = USER_HOME / '.local' / 'bin' / 'hyprwhspr'
    if cmd_symlink.is_symlink() and os.readlink(cmd_symlink).endswith('/bin/hyprwhspr'):
        fs_targets.append((f"Command symlink: {cmd_symlink}", cmd_symlink))
    if STATE_DIR.exists():
        fs_targets.append((f"State files: {STATE_DIR}", STATE_DIR))
    if CREDENTIALS_FILE.exists():
        fs_targets.append(("Stored API credentials", CREDENTIALS_FILE))
    temp_dir = USER_BASE / 'temp'
    if temp_dir.exists():
        fs_targets.append((f"Temporary files: {temp_dir}", temp_dir))
    items_to_remove.extend(label for label, _ in fs_targets)

    # Models
    if not keep_models and PYWHISPERCPP_MODELS_DIR.exists():
        models = list(PYWHISPERCPP_MODELS_DIR.glob('ggml-*.bin'))
        if models:
            items_to_remove.append(f"Whisper models: {len(models)} model(s) in {PYWHISPERCPP_MODELS_DIR}")
    
    # Permissions (if not skipped)
    if not skip_permissions:
        items_to_remove.append("System permissions (groups, udev rules) - optional")
    
    if not items_to_remove:
        log_info("Nothing to remove - hyprwhspr appears to be already uninstalled")
        return
    
    # Show summary
    print("\nThe following will be removed:")
    for item in items_to_remove:
        print(f"  • {item}")
    print()
    
    # Confirmation
    if not yes:
        log_warning("This will permanently delete all hyprwhspr data and configuration.")
        if not Confirm.ask("Are you sure you want to continue?", default=False):
            print("\nUninstall cancelled.")
            return
    
    print("\n" + "="*60)
    print("Removing Components")
    print("="*60 + "\n")
    
    errors = []
    
    # 1. Stop and remove systemd services
    log_info("Stopping and removing systemd services...")
    try:
        # Stop and disable hyprwhspr service
        if (USER_SYSTEMD_DIR / SERVICE_NAME).exists():
            run_command(['systemctl', '--user', 'stop', SERVICE_NAME], check=False)
            run_command(['systemctl', '--user', 'disable', SERVICE_NAME], check=False)
            (USER_SYSTEMD_DIR / SERVICE_NAME).unlink(missing_ok=True)
            log_success(f"Removed {SERVICE_NAME}")


        # Stop and disable deprecated resume service
        if (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).exists():
            run_command(['systemctl', '--user', 'stop', RESUME_SERVICE_NAME], check=False)
            run_command(['systemctl', '--user', 'disable', RESUME_SERVICE_NAME], check=False)
            (USER_SYSTEMD_DIR / RESUME_SERVICE_NAME).unlink(missing_ok=True)
            log_success(f"Removed {RESUME_SERVICE_NAME}")

        if _is_hyprwhspr_managed_ydotool_unit(ydotool_unit_path):
            run_command(['systemctl', '--user', 'stop', YDOTOOL_UNIT], check=False)
            run_command(['systemctl', '--user', 'disable', YDOTOOL_UNIT], check=False)
            ydotool_unit_path.unlink(missing_ok=True)
            log_success(f"Removed {YDOTOOL_UNIT}")

        # Reload systemd daemon
        run_command(['systemctl', '--user', 'daemon-reload'], check=False)
    except Exception as e:
        error_msg = f"Failed to remove systemd services: {e}"
        log_warning(error_msg)
        errors.append(error_msg)
    
    # 2. Remove Waybar integration
    log_info("Removing Waybar integration...")
    try:
        setup_waybar('remove')
    except Exception as e:
        error_msg = f"Failed to remove Waybar integration: {e}"
        log_warning(error_msg)
        errors.append(error_msg)
    
    # 3. Backend-specific cleanup (while config still exists for detection)
    log_info("Cleaning up backend...")
    try:
        # Lazy import with attribute access: keeps test patches on cli.setup
        # effective without importing the whole setup module at load time.
        try:
            from . import setup as _setup
        except ImportError:
            from cli import setup as _setup
        current_backend = _setup._detect_current_backend()
        if current_backend:
            _setup._cleanup_backend(current_backend)
    except Exception as e:
        error_msg = f"Backend cleanup failed: {e}"
        log_warning(error_msg)
        errors.append(error_msg)

    # 4. Remove filesystem targets from the summary manifest
    log_info("Removing files...")
    for label, target in fs_targets:
        try:
            if target.is_dir() and not target.is_symlink():
                shutil.rmtree(target, ignore_errors=True)
            else:
                target.unlink(missing_ok=True)
            log_success(f"Removed {label}")
        except Exception as e:
            error_msg = f"Failed to remove {label}: {e}"
            log_warning(error_msg)
            errors.append(error_msg)

    # 5. Remove models (if not keeping)
    if not keep_models:
        log_info("Removing Whisper models...")
        try:
            if PYWHISPERCPP_MODELS_DIR.exists():
                models = list(PYWHISPERCPP_MODELS_DIR.glob('ggml-*.bin'))
                if models:
                    shutil.rmtree(PYWHISPERCPP_MODELS_DIR, ignore_errors=True)
                    log_success(f"Removed {len(models)} model(s) from {PYWHISPERCPP_MODELS_DIR}")
                else:
                    # Remove empty directory
                    PYWHISPERCPP_MODELS_DIR.rmdir()
        except Exception as e:
            error_msg = f"Failed to remove models: {e}"
            log_warning(error_msg)
            errors.append(error_msg)
    else:
        log_info("Keeping Whisper models (--keep-models flag)")
    
    # 6. Remove the base directory if it's empty or only contains empty subdirs
    try:
        if USER_BASE.exists():
            has_content = False
            for item in USER_BASE.iterdir():
                if item.is_file():
                    has_content = True
                    break
                elif item.is_dir():
                    try:
                        if any(item.iterdir()):
                            has_content = True
                            break
                    except Exception:
                        pass

            if not has_content:
                shutil.rmtree(USER_BASE, ignore_errors=True)
                log_success(f"Removed {USER_BASE}")
    except Exception:
        pass  # Ignore errors when trying to remove base directory

    # 7. Remove system permissions (if requested)
    permissions_removed = False
    if not skip_permissions:
        log_info("Checking system permissions...")
        
        should_remove = remove_permissions
        if not remove_permissions and not yes:
            should_remove = Confirm.ask(
                "Remove system permissions (remove user from input/audio/tty groups and udev rules)?",
                default=False
            )
        
        if should_remove:
            permissions_removed = True
            try:
                username = os.environ.get('SUDO_USER') or os.environ.get('USER') or getpass.getuser()
                if not username:
                    log_warning("Could not determine username for permission removal")
                else:
                    # Remove from groups
                    groups_to_remove = ['input', 'audio', 'tty']
                    for group in groups_to_remove:
                        try:
                            run_sudo_command(['gpasswd', '-d', username, group], check=False)
                            log_success(f"Removed user from '{group}' group")
                        except Exception as e:
                            log_warning(f"Failed to remove user from '{group}' group: {e}")
                    
                    # Remove udev rule (only if it exists and was created by hyprwhspr)
                    udev_rule = Path('/etc/udev/rules.d/99-uinput.rules')
                    if udev_rule.exists():
                        # Check if it's our rule by reading it
                        try:
                            with open(udev_rule, 'r', encoding='utf-8') as f:
                                content = f.read()
                            if 'hyprwhspr' in content.lower() or 'input' in content.lower():
                                run_sudo_command(['rm', str(udev_rule)], check=False)
                                log_success("Removed udev rule")
                                # Reload udev
                                run_sudo_command(['udevadm', 'control', '--reload-rules'], check=False)
                                run_sudo_command(['udevadm', 'trigger', '--name-match=uinput'], check=False)
                        except Exception as e:
                            log_warning(f"Failed to remove udev rule: {e}")
            except Exception as e:
                error_msg = f"Failed to remove system permissions: {e}"
                log_warning(error_msg)
                errors.append(error_msg)
        else:
            log_info("Skipping permission removal")
    else:
        log_info("Skipping permission removal (--skip-permissions flag)")
    
    # Summary
    print("\n" + "="*60)
    if errors:
        log_warning("Uninstall completed with some errors:")
        for error in errors:
            log_warning(f"  • {error}")
        print("="*60)
    else:
        log_success("Uninstall completed successfully!")
        print("="*60)
    
    print("\nAll hyprwhspr user data has been removed.")
    hf_cache = Path(os.environ.get('XDG_CACHE_HOME', USER_HOME / '.cache')) / 'huggingface'
    if hf_cache.exists():
        print(f"Note: {hf_cache} (shared model cache) was not removed.")
        print("      Delete it manually if no other apps use Hugging Face models.")
    if not skip_permissions and not permissions_removed:
        print("Note: System permissions (group memberships, udev rules) were not removed.")
        print("      You may want to remove them manually if needed.")
    print()
