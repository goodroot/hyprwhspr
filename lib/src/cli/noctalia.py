"""
Noctalia shell integration commands for hyprwhspr
"""

import os
import json
import re
import shutil
import subprocess
from pathlib import Path

try:
    from ..output_control import log_info, log_success, log_warning, log_error
except ImportError:
    from output_control import log_info, log_success, log_warning, log_error

try:
    from ..installation import is_nix_install, tray_command as stable_tray_command
except ImportError:
    from installation import is_nix_install, tray_command as stable_tray_command

try:
    from ..paths import DATA_DIR
except ImportError:
    from paths import DATA_DIR

from ._shared import HYPRWHSPR_ROOT, USER_HOME, _validate_hyprwhspr_root


# ==================== Noctalia Commands ====================

# Plugin id must match config/noctalia/plugin/plugin.toml
NOCTALIA_PLUGIN_ID = 'goodroot/noctwhspr'
NOCTALIA_LEGACY_PLUGIN_ID = 'goodroot/hyprwhspr'
NOCTALIA_TEMPLATE_ID = 'noctwhspr_mic_osd'
NOCTALIA_LEGACY_TEMPLATE_ID = 'hyprwhspr_mic_osd'


def _noctalia_paths():
    """Resolve Noctalia's XDG locations."""
    xdg_data = Path(os.environ.get('XDG_DATA_HOME', USER_HOME / '.local' / 'share'))
    xdg_state = Path(os.environ.get('XDG_STATE_HOME', USER_HOME / '.local' / 'state'))
    xdg_config = Path(os.environ.get('XDG_CONFIG_HOME', USER_HOME / '.config'))
    return {
        'plugin_dir': xdg_data / 'noctalia' / 'plugins' / 'noctwhspr',
        'legacy_plugin_dir': xdg_data / 'noctalia' / 'plugins' / 'hyprwhspr',
        'settings': xdg_state / 'noctalia' / 'settings.toml',
        'template_input': xdg_config / 'noctalia' / 'templates' / 'noctwhspr-mic-osd.css',
        'legacy_template_input': xdg_config / 'noctalia' / 'templates' / 'hyprwhspr-mic-osd.css',
        'template_output': xdg_config / 'hyprwhspr' / 'theme' / 'mic-osd.css',
    }


def _noctalia_detected() -> bool:
    """True when the Noctalia shell appears to be installed."""
    if shutil.which('noctalia') is not None:
        return True
    return _noctalia_paths()['settings'].exists()


def _noctalia_integration_installed() -> bool:
    """True when hyprwhspr's Noctalia pieces have been installed at all.

    Distinguishes 'integration is broken' from 'user runs Noctalia but never
    asked for the integration', so status can stay quiet about the latter.
    """
    dst = _noctalia_paths()
    return ((dst['plugin_dir'] / 'plugin.toml').exists()
            or dst['template_input'].exists())


def _noctalia_msg(*args) -> bool:
    """Send a command to a running Noctalia instance (best effort)."""
    if shutil.which('noctalia') is None:
        return False
    try:
        result = subprocess.run(['noctalia', 'msg', *args],
                                capture_output=True, timeout=10, check=False)
        return result.returncode == 0
    except Exception:
        return False


def _noctalia_register_template(settings_path: Path, template_input: Path,
                                template_output: Path) -> bool:
    """Register the mic-OSD user template in Noctalia's settings.toml.

    Appends a [theme.templates.user.*] section (valid TOML regardless of
    where [theme] appears), then validates with `noctalia config validate`
    and rolls back on failure.
    """
    marker = f'theme.templates.user.{NOCTALIA_TEMPLATE_ID}'
    try:
        content = settings_path.read_text(encoding='utf-8')
    except OSError:
        return False
    if marker in content:
        # Registered — but if the section no longer references the current
        # template filename, its input_path points somewhere stale (e.g. a
        # partially-migrated path form). Surface it instead of masking it.
        if template_input.name not in content:
            log_warning(f"Noctalia template section [{marker}] exists but does not "
                        f"reference {template_input.name} - check its input_path")
            return False
        return True  # already registered

    section = (
        f'\n[{marker}]\n'
        f'input_path = "{template_input}"\n'
        f'output_path = "{template_output}"\n'
    )
    backup = content
    try:
        settings_path.write_text(content.rstrip('\n') + '\n' + section,
                                 encoding='utf-8')
    except OSError:
        return False

    if shutil.which('noctalia') is not None:
        try:
            result = subprocess.run(
                ['noctalia', 'config', 'validate', str(settings_path)],
                capture_output=True, timeout=10, check=False)
            if result.returncode != 0:
                settings_path.write_text(backup, encoding='utf-8')
                return False
        except Exception:
            pass  # validation unavailable; keep the append
    return True


def _noctalia_migrate_legacy_settings(settings_path: Path, dst: dict) -> bool:
    """Rename legacy hyprwhspr Noctalia identifiers to noctwhspr.

    Replacements are anchored: ids only match when not followed by a
    word/dash character (so 'goodroot/hyprwhspr-extras' is untouched), and the
    template path is migrated by its filename alone, which covers every
    serialization Noctalia may have stored ($XDG_CONFIG_HOME token, absolute,
    tilde) since only the filename changed in the rename.
    """
    try:
        content = settings_path.read_text(encoding='utf-8')
    except OSError:
        return False

    updated = content
    replacements = {
        re.escape(NOCTALIA_LEGACY_PLUGIN_ID) + r'(?![\w-])': NOCTALIA_PLUGIN_ID,
        re.escape(NOCTALIA_LEGACY_TEMPLATE_ID) + r'(?![\w-])': NOCTALIA_TEMPLATE_ID,
        re.escape(dst['legacy_template_input'].name): dst['template_input'].name,
    }
    for pattern, new in replacements.items():
        updated = re.sub(pattern, new, updated)

    if updated == content:
        return True

    backup = content
    try:
        settings_path.write_text(updated, encoding='utf-8')
    except OSError:
        return False

    if shutil.which('noctalia') is not None:
        try:
            result = subprocess.run(
                ['noctalia', 'config', 'validate', str(settings_path)],
                capture_output=True, timeout=10, check=False)
            if result.returncode != 0:
                settings_path.write_text(backup, encoding='utf-8')
                return False
        except Exception:
            pass
    return True


def noctalia_command(action: str):
    """Handle noctalia subcommands"""
    if action == 'install':
        if setup_noctalia('install') is False:
            raise RuntimeError('Noctalia installation incomplete; see the reported conflicts')
    elif action == 'remove':
        if setup_noctalia('remove') is False:
            raise RuntimeError('Noctalia removal incomplete; see the reported guidance')
    elif action == 'status':
        noctalia_status()
    else:
        log_error(f"Unknown noctalia action: {action}")


def _setup_noctalia(mode: str = 'install', *, widget_content=None, tray=None):
    """Install or remove the Noctalia shell integration.

    Two independent pieces:
    - bar widget: a Luau plugin shimming the existing tray script
    - mic-OSD theming: a Noctalia app-theming user template rendering the
      live palette to ~/.config/hyprwhspr/theme/mic-osd.css
    """
    if not _validate_hyprwhspr_root():
        return False

    src_plugin = Path(HYPRWHSPR_ROOT) / 'config' / 'noctalia' / 'plugin'
    src_template = Path(HYPRWHSPR_ROOT) / 'config' / 'noctalia' / 'templates' / 'mic-osd.css'
    dst = _noctalia_paths()

    if mode == 'install':
        log_info("Setting up Noctalia integration...")

        if not src_plugin.is_dir() or not src_template.exists():
            log_error(f"Noctalia assets not found under {HYPRWHSPR_ROOT}/config/noctalia")
            return False

        # Validate first: a failed reinstall must leave the working shim intact.
        # Publish the shim and patched widget before enabling the plugin.
        if tray is not None:
            command = stable_tray_command(HYPRWHSPR_ROOT)
            if command is None:
                return False
            from managed_install import atomic_text
            atomic_text(tray, f'#!/bin/sh\nexec {command} "$@"\n', 0o755)

        # 1. Bar widget plugin
        # copyfile, not copy2: read-only package modes (the Nix store) would
        # block the next reinstall.
        (dst['plugin_dir'] / 'translations').mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_plugin / 'plugin.toml', dst['plugin_dir'] / 'plugin.toml')
        if widget_content is None:
            shutil.copyfile(src_plugin / 'widget.luau', dst['plugin_dir'] / 'widget.luau')
        else:
            from managed_install import atomic_text
            atomic_text(dst['plugin_dir'] / 'widget.luau', widget_content)
        for translation in (src_plugin / 'translations').iterdir():
            shutil.copyfile(translation, dst['plugin_dir'] / 'translations' / translation.name)
        log_success(f"Plugin installed to {dst['plugin_dir']}")
        if dst['legacy_plugin_dir'].is_dir() and not os.environ.get('HYPRWHSPR_GENERATION'):
            shutil.rmtree(dst['legacy_plugin_dir'], ignore_errors=True)
            log_success("Legacy hyprwhspr plugin directory removed")

        _noctalia_msg('plugins', 'disable', NOCTALIA_LEGACY_PLUGIN_ID)
        if _noctalia_msg('plugins', 'enable', NOCTALIA_PLUGIN_ID):
            log_success(f"Plugin '{NOCTALIA_PLUGIN_ID}' enabled")
        else:
            log_warning("Could not enable plugin (is Noctalia running?)")
            log_info(f"  Enable manually: noctalia msg plugins enable {NOCTALIA_PLUGIN_ID}")

        # 2. Mic-OSD palette template
        dst['template_input'].parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(src_template, dst['template_input'])
        log_success(f"Theme template installed to {dst['template_input']}")
        if dst['legacy_template_input'].exists() and not os.environ.get('HYPRWHSPR_GENERATION'):
            dst['legacy_template_input'].unlink()
            log_success("Legacy hyprwhspr theme template removed")

        if dst['settings'].exists():
            if _noctalia_migrate_legacy_settings(dst['settings'], dst):
                log_success("Noctalia settings use noctwhspr identifiers")
            else:
                log_warning("Could not migrate old hyprwhspr Noctalia identifiers automatically")
            if _noctalia_register_template(dst['settings'], dst['template_input'],
                                           dst['template_output']):
                log_success("Theme template registered in Noctalia settings")
                _noctalia_msg('config-reload')
                if _noctalia_msg('templates-apply'):
                    log_success("Palette rendered - mic-OSD now follows Noctalia's theme")
            else:
                log_warning("Could not register theme template automatically")
                _print_noctalia_template_snippet(dst)
        else:
            log_warning(f"Noctalia settings not found at {dst['settings']}")
            _print_noctalia_template_snippet(dst)

        widget_placed = False
        if dst['settings'].exists():
            try:
                widget_placed = (f'{NOCTALIA_PLUGIN_ID}:status'
                                 in dst['settings'].read_text(encoding='utf-8'))
            except OSError:
                pass
        log_info("")
        if widget_placed:
            log_success("All set - the noctwhspr widget is live in your bar")
        else:
            log_info("Last step - put the widget in your bar:")
            log_info("  Noctalia Settings -> Bar -> add widget -> noctwhspr")
            log_info(f"  (or add \"{NOCTALIA_PLUGIN_ID}:status\" to a bar's widget list in settings.toml)")
        return True

    elif mode == 'remove':
        log_info("Removing Noctalia integration...")

        _noctalia_msg('plugins', 'disable', NOCTALIA_PLUGIN_ID)
        _noctalia_msg('plugins', 'disable', NOCTALIA_LEGACY_PLUGIN_ID)
        if dst['plugin_dir'].is_dir():
            shutil.rmtree(dst['plugin_dir'], ignore_errors=True)
            log_success("Plugin removed")
        if dst['legacy_plugin_dir'].is_dir() and not os.environ.get('HYPRWHSPR_GENERATION'):
            shutil.rmtree(dst['legacy_plugin_dir'], ignore_errors=True)
            log_success("Legacy plugin removed")

        for key in ('template_input', 'legacy_template_input', 'template_output'):
            if dst[key].exists():
                dst[key].unlink()
        log_success("Theme template files removed")

        _noctalia_removal_guidance()
        return True


def _noctalia_removal_guidance():
    log_info("If present, also remove from Noctalia's settings.toml:")
    log_info(f"  - the [theme.templates.user.{NOCTALIA_TEMPLATE_ID}] section")
    log_info(f"  - the [theme.templates.user.{NOCTALIA_LEGACY_TEMPLATE_ID}] section")
    log_info(f"  - \"{NOCTALIA_PLUGIN_ID}:status\" from the bar's widget list")
    log_info(f"  - \"{NOCTALIA_LEGACY_PLUGIN_ID}:status\" from the bar's widget list")


def _print_noctalia_template_snippet(dst: dict):
    log_info("  Add this to Noctalia's settings.toml, then run: noctalia msg templates-apply")
    log_info(f"    [theme.templates.user.{NOCTALIA_TEMPLATE_ID}]")
    log_info(f"    input_path = \"{dst['template_input']}\"")
    log_info(f"    output_path = \"{dst['template_output']}\"")


def noctalia_status():
    """Check the state of the Noctalia integration"""
    dst = _noctalia_paths()
    ok = True

    if (dst['plugin_dir'] / 'plugin.toml').exists():
        log_success(f"Plugin installed: {dst['plugin_dir']}")
    else:
        log_warning("Plugin not installed")
        ok = False

    if dst['template_input'].exists():
        log_success(f"Theme template input: {dst['template_input']}")
    else:
        log_warning("Theme template input not installed")
        ok = False

    if dst['template_output'].exists():
        log_success(f"Rendered palette CSS: {dst['template_output']}")
    else:
        # Not a failure: rendering only happens once a running Noctalia
        # applies templates, which legitimately postdates a correct install.
        log_info("Palette CSS not rendered yet (run: noctalia msg templates-apply)")

    if dst['settings'].exists():
        try:
            content = dst['settings'].read_text(encoding='utf-8')
            if f'theme.templates.user.{NOCTALIA_TEMPLATE_ID}' in content:
                log_success("Theme template registered in Noctalia settings")
            else:
                log_warning("Theme template not registered in Noctalia settings")
                ok = False
            if f'{NOCTALIA_PLUGIN_ID}:status' in content:
                log_success("Bar widget present in a bar widget list")
            else:
                log_warning("Bar widget not in any bar widget list (add via Noctalia Settings -> Bar)")
                ok = False
            if f'{NOCTALIA_LEGACY_PLUGIN_ID}:status' in content:
                log_warning(f"Legacy widget id still present: {NOCTALIA_LEGACY_PLUGIN_ID}:status")
                ok = False
        except OSError:
            log_warning(f"Could not read {dst['settings']}")
            ok = False
    else:
        log_warning(f"Noctalia settings not found: {dst['settings']}")
        ok = False

    return ok


def _prepare_widget(tray):
    """The widget patched to prefer the tray shim, or None with a warning."""
    from managed_integrations import managed_widget_content
    try:
        source = Path(HYPRWHSPR_ROOT) / 'config/noctalia/plugin/widget.luau'
        return managed_widget_content(source.read_text(encoding='utf-8'), tray)
    except (OSError, UnicodeError, RuntimeError) as exc:
        log_warning(f'Noctalia installation incomplete; cannot prepare the widget: {exc}')
        return None


def _plugin_disabled_for_removal() -> bool:
    """Disable a running plugin first; removing its files under it would strand it."""
    if shutil.which('noctalia') and not _noctalia_msg('plugins', 'disable', NOCTALIA_PLUGIN_ID):
        log_warning('Could not disable the Noctalia plugin; plugin files were preserved. Start Noctalia and retry removal.')
        _noctalia_removal_guidance()
        return False
    return True


def _setup_nix_noctalia(mode: str):
    """Point the widget at a shim; store paths vanish after garbage collection."""
    # Not noctalia-tray: the managed installer owns that file.
    tray = DATA_DIR / 'integrations' / 'nix-noctalia-tray'
    if mode == 'install':
        widget_content = _prepare_widget(tray)
        if widget_content is None:
            return False
        return _setup_noctalia('install', widget_content=widget_content, tray=tray)
    if mode == 'remove' and not _plugin_disabled_for_removal():
        return False
    result = _setup_noctalia(mode)
    if mode == 'remove':
        tray.unlink(missing_ok=True)
    return result


def setup_noctalia(mode: str = 'install'):
    if is_nix_install():
        return _setup_nix_noctalia(mode)
    if not os.environ.get('HYPRWHSPR_GENERATION'):
        return _setup_noctalia(mode)
    from managed_integrations import edit_files
    from managed_install import Installation
    dst = _noctalia_paths()
    source = Path(HYPRWHSPR_ROOT) / 'config/noctalia/plugin'
    paths = [dst['plugin_dir'] / p.relative_to(source) for p in source.rglob('*') if p.is_file()]
    tray = Installation().data / 'integrations/noctalia-tray'
    paths.extend([dst['settings'], dst['template_input'], dst['template_output'], tray])
    if mode == 'remove' and not _plugin_disabled_for_removal():
        return False
    if mode == 'install':
        widget_content = _prepare_widget(tray)
        if widget_content is None:
            return False
    removal = {}
    with edit_files(paths, [dst['settings'], dst['template_output']], mode, removal=removal, allow_shared_preservation=True) as proceed:
        if not proceed:
            if mode == 'remove':
                _noctalia_removal_guidance()
            return removal.get('complete', False) if mode == 'remove' else False
        if mode == 'install':
            return _setup_noctalia('install', widget_content=widget_content, tray=tray)
        return _setup_noctalia(mode)
