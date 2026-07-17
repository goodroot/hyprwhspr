"""
Configuration management commands for hyprwhspr
"""

import os
import json
import subprocess
from typing import Optional

from rich.prompt import Prompt, Confirm

try:
    from ..config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from ..output_control import log_info, log_success, log_warning, log_error
except ImportError:
    from output_control import log_info, log_success, log_warning, log_error

from ._shared import USER_CONFIG_DIR


# ==================== Config Commands ====================

def config_command(action: str, show_all: bool = False):
    """Handle config subcommands"""
    if action == 'init':
        setup_config()
    elif action == 'show':
        show_config(show_all=show_all)
    elif action == 'edit':
        edit_config()
    elif action == 'secondary-shortcut':
        configure_secondary_shortcut()
    elif action == 'focused-window':
        show_focused_window_config_identifiers()
    else:
        log_error(f"Unknown config action: {action}")


def show_focused_window_config_identifiers():
    """Print identifiers that can be used in the applications config object."""
    try:
        from text_injector import TextInjector

        injector = TextInjector(config_manager=ConfigManager(verbose=False))
        try:
            window_info = injector._get_active_window_info()
            identifiers = TextInjector.focused_window_identifiers(window_info)
        finally:
            injector.close()

        print("\nFocused window:")
        if not window_info:
            print("  not detected")
            print("\nNo identifiers available.")
            return

        for key in ('source', 'class', 'app_id', 'title'):
            value = window_info.get(key)
            if value:
                print(f"  {key}: {value}")

        print("\nConfig identifiers:")
        if identifiers:
            for identifier in identifiers:
                print(f"  {identifier}")
        else:
            print("  none")
    except Exception as e:
        log_error(f"Failed to inspect focused window: {e}")


def setup_config(backend: Optional[str] = None, model: Optional[str] = None, remote_config: Optional[dict] = None):
    """Create or update user config"""
    log_info("Setting up user config...")
    
    USER_CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    config_file = USER_CONFIG_DIR / 'config.json'
    
    if not config_file.exists():
        # Create default config using ConfigManager
        config = ConfigManager()
        # Override with user selections if provided
        if backend:
            config.set_setting('transcription_backend', backend)
        if model:
            config.set_setting('model', model)
        
        # Apply remote configuration if provided
        if remote_config:
            for key, value in remote_config.items():
                config.set_setting(key, value)
        
        if config.save_config():
            log_success(f"Created {config_file}")
        else:
            log_error(f"Failed to create {config_file}")
    else:
        log_info(f"Config already exists at {config_file}")
        # Update existing config via ConfigManager (handles migrations, sparse save)
        try:
            config = ConfigManager()

            if backend:
                # Map old values for backward compatibility
                if backend == 'local':
                    backend = 'cpu'
                elif backend == 'remote':
                    backend = 'rest-api'
                config.set_setting('transcription_backend', backend)

            if remote_config:
                for key, value in remote_config.items():
                    config.set_setting(key, value)

            if model:
                config.set_setting('model', model)

            if config.save_config():
                log_success("Updated existing config")
            else:
                log_error("Failed to save updated config")
        except Exception as e:
            log_error(f"Failed to update config: {e}")


def show_config(show_all: bool = False):
    """Display current config"""
    try:
        if show_all:
            config = ConfigManager().get_all_settings()
        else:
            config_file = USER_CONFIG_DIR / 'config.json'
            if not config_file.exists():
                log_error("Config file not found. Run 'hyprwhspr config init' first.")
                return
            with open(config_file, 'r', encoding='utf-8') as f:
                config = json.load(f)
        print(json.dumps(config, indent=2))
    except (json.JSONDecodeError, IOError) as e:
        log_error(f"Failed to read config: {e}")


def edit_config():
    """Open config in editor"""
    config_file = USER_CONFIG_DIR / 'config.json'
    
    if not config_file.exists():
        log_error("Config file not found. Run 'hyprwhspr config init' first.")
        return
    
    editor = os.environ.get('EDITOR', 'nano')
    try:
        subprocess.run([editor, str(config_file)], check=True)
        log_success("Config edited")
    except Exception as e:
        log_error(f"Failed to open editor: {e}")


def configure_secondary_shortcut():
    """Configure secondary shortcut and language"""
    from rich.prompt import Prompt, Confirm
    
    config = ConfigManager()
    
    print("\n" + "="*60)
    print("Secondary Shortcut Configuration")
    print("="*60)
    print("\nConfigure a second hotkey that will use a specific language for transcription.")
    print("The primary shortcut will continue to use the default language from config.")
    print()
    
    # Check if already configured
    current_shortcut = config.get_setting('secondary_shortcut')
    current_language = config.get_setting('secondary_language')
    
    if current_shortcut:
        print(f"Current secondary shortcut: {current_shortcut}")
        if current_language:
            print(f"Current secondary language: {current_language}")
        print()
        if not Confirm.ask("Do you want to change the secondary shortcut?", default=False):
            return
    
    # Prompt for shortcut
    print("\nEnter the secondary shortcut key combination.")
    print("Examples: SUPER+ALT+I, CTRL+SHIFT+L, F11")
    print("Leave blank to disable secondary shortcut.")
    shortcut = Prompt.ask("Secondary shortcut", default=current_shortcut or "")
    
    if not shortcut or shortcut.strip() == "":
        # Disable secondary shortcut
        config.set_setting('secondary_shortcut', None)
        config.set_setting('secondary_language', None)
        config.save_config()
        log_success("Secondary shortcut disabled")
        return
    
    # Prompt for language
    print("\nEnter the language code for this shortcut.")
    print("Examples: 'it' (Italian), 'en' (English), 'fr' (French), 'de' (German), 'es' (Spanish)")
    print("Leave blank to disable secondary shortcut.")
    language = Prompt.ask("Language code", default=current_language or "")
    
    if not language or language.strip() == "":
        log_warning("Language code is required. Secondary shortcut not configured.")
        return
    
    # Validate language code (basic check - 2-3 letter code)
    language = language.strip().lower()
    if len(language) < 2 or len(language) > 3:
        log_warning("Language code should be 2-3 letters (e.g., 'it', 'en', 'fr')")
        if not Confirm.ask("Continue anyway?", default=False):
            return
    
    # Save configuration
    config.set_setting('secondary_shortcut', shortcut.strip())
    config.set_setting('secondary_language', language)
    config.save_config()
    
    log_success(f"Secondary shortcut configured: {shortcut.strip()} (language: {language})")
    print("\nNote: Restart hyprwhspr service for changes to take effect:")
    print("  systemctl --user restart hyprwhspr")
