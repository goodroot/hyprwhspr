"""
CLI command implementations for hyprwhspr
"""

import os
import sys
import json
import subprocess
import getpass
from pathlib import Path
from typing import Optional

try:
    from rich.prompt import Prompt, Confirm
except ImportError:
    # Fallback if rich is not available (shouldn't happen in production)
    # This would cause errors, but rich is in requirements.txt
    Prompt = None
    Confirm = None

try:
    from .config_manager import ConfigManager
except ImportError:
    from config_manager import ConfigManager

try:
    from .backend_installer import install_backend
except ImportError:
    from backend_installer import install_backend


# Constants
HYPRWHSPR_ROOT = os.environ.get('HYPRWHSPR_ROOT', '/usr/lib/hyprwhspr')
SERVICE_NAME = 'hyprwhspr.service'
YDOTOOL_UNIT = 'ydotool.service'
USER_HOME = Path.home()
USER_CONFIG_DIR = USER_HOME / '.config' / 'hyprwhspr'
USER_SYSTEMD_DIR = USER_HOME / '.config' / 'systemd' / 'user'
PYWHISPERCPP_MODELS_DIR = Path(os.environ.get('XDG_DATA_HOME', USER_HOME / '.local' / 'share')) / 'pywhispercpp' / 'models'


def _strip_jsonc(text: str) -> str:
    """Strip // and /* */ comments from JSONC while preserving strings."""
    result = []
    i = 0
    in_str = False
    esc = False
    in_line = False
    in_block = False
    n = len(text)

    while i < n:
        ch = text[i]
        nxt = text[i + 1] if i + 1 < n else ""

        if in_line:
            if ch == "\n":
                in_line = False
                result.append(ch)
            i += 1
            continue

        if in_block:
            if ch == "*" and nxt == "/":
                in_block = False
                i += 2
            else:
                i += 1
            continue

        if not in_str:
            if ch == "/" and nxt == "/":
                in_line = True
                i += 2
                continue
            if ch == "/" and nxt == "*":
                in_block = True
                i += 2
                continue

        if ch == '"' and not esc:
            in_str = not in_str

        if ch == "\\" and in_str:
            esc = not esc
        else:
            esc = False

        result.append(ch)
        i += 1

    return "".join(result)


def _load_jsonc(path: Path):
    """Load JSONC file by stripping comments first."""
    with open(path, 'r', encoding='utf-8') as f:
        stripped = _strip_jsonc(f.read())
    return json.loads(stripped)


def _validate_hyprwhspr_root() -> bool:
    """Validate that HYPRWHSPR_ROOT exists and contains expected files"""
    root_path = Path(HYPRWHSPR_ROOT)
    is_development = root_path != Path('/usr/lib/hyprwhspr')
    
    if not root_path.exists():
        log_error(f"HYPRWHSPR_ROOT does not exist: {HYPRWHSPR_ROOT}")
        log_error("")
        if is_development:
            log_error("Development installation detected (not /usr/lib/hyprwhspr)")
            log_error("Check that HYPRWHSPR_ROOT environment variable is set correctly.")
            log_error("For development, ensure you're running from the repository root.")
            log_error("")
            log_error("Example: export HYPRWHSPR_ROOT=$(pwd)")
        else:
            log_error("This appears to be an AUR installation issue.")
            log_error("Try reinstalling: yay -S hyprwhspr")
        log_error("")
        return False
    
    # Check for expected files
    required_files = [
        root_path / 'bin' / 'hyprwhspr',
        root_path / 'lib' / 'main.py',
    ]
    
    missing_files = []
    for file_path in required_files:
        if not file_path.exists():
            missing_files.append(str(file_path.relative_to(root_path)))
    
    if missing_files:
        log_error(f"HYPRWHSPR_ROOT is missing required files: {', '.join(missing_files)}")
        log_error(f"Root path: {HYPRWHSPR_ROOT}")
        log_error("")
        if is_development:
            log_error("Development installation detected (not /usr/lib/hyprwhspr)")
            log_error("This may be a development installation issue.")
            log_error("Ensure you're running from the repository root.")
            log_error("")
            log_error("Expected structure:")
            log_error(f"  {root_path}/bin/hyprwhspr")
            log_error(f"  {root_path}/lib/main.py")
        else:
            log_error("This appears to be a corrupted AUR installation.")
            log_error("Try reinstalling: yay -S hyprwhspr")
        log_error("")
        return False
    
    return True


def log_info(msg: str):
    """Print info message"""
    print(f"[INFO] {msg}")


def log_success(msg: str):
    """Print success message"""
    print(f"[SUCCESS] {msg}")


def log_warning(msg: str):
    """Print warning message"""
    print(f"[WARNING] {msg}")


def log_error(msg: str):
    """Print error message"""
    print(f"[ERROR] {msg}", file=sys.stderr)


def run_command(cmd: list, check: bool = True, capture_output: bool = False) -> subprocess.CompletedProcess:
    """Run a shell command"""
    try:
        result = subprocess.run(
            cmd,
            check=check,
            capture_output=capture_output,
            text=True
        )
        return result
    except subprocess.CalledProcessError:
        log_error(f"Command failed: {' '.join(cmd)}")
        raise
    except FileNotFoundError:
        log_error(f"Command not found: {cmd[0]}")
        raise


def run_sudo_command(cmd: list, check: bool = True, input_data: Optional[bytes] = None) -> subprocess.CompletedProcess:
    """Run a command with sudo"""
    sudo_cmd = ['sudo'] + cmd
    try:
        result = subprocess.run(
            sudo_cmd,
            check=check,
            input=input_data,
            text=False if input_data else True
        )
        return result
    except subprocess.CalledProcessError:
        log_error(f"Command failed: {' '.join(sudo_cmd)}")
        raise
    except FileNotFoundError:
        log_error(f"Command not found: {sudo_cmd[0]}")
        raise


# ==================== Setup Command ====================

def _prompt_backend_selection():
    """Prompt user for backend selection (CPU, NVIDIA, AMD, or Remote)"""
    print("\n" + "="*60)
    print("Backend Selection")
    print("="*60)
    print("\nChoose your transcription backend:")
    print()
    print("  [1] CPU - CPU-only, works on all systems")
    print("  [2] NVIDIA - NVIDIA GPU acceleration (CUDA)")
    print("  [3] AMD - AMD GPU acceleration (ROCm)")
    print("  [4] Remote - Use external API/backend (requires network)")
    print()
    
    while True:
        try:
            choice = Prompt.ask("Select backend", choices=['1', '2', '3', '4'], default='1')
            backend_map = {
                '1': 'cpu',
                '2': 'nvidia',
                '3': 'amd',
                '4': 'remote'
            }
            selected = backend_map[choice]
            
            backend_names = {
                'cpu': 'CPU',
                'nvidia': 'NVIDIA (CUDA)',
                'amd': 'AMD (ROCm)',
                'remote': 'Remote'
            }
            print(f"\n✓ Selected: {backend_names[selected]}")
            return selected
        except (ValueError, IndexError, KeyboardInterrupt):
            print("\nInvalid selection. Please try again.")
            continue


def _prompt_model_selection():
    """Prompt user for model selection"""
    multilingual_models = [
        ('tiny', 'Fastest, least accurate'),
        ('base', 'Good balance (recommended)'),
        ('small', 'Better accuracy'),
        ('medium', 'High accuracy'),
        ('large', 'Best accuracy, requires GPU'),
        ('large-v3', 'Latest large model, requires GPU')
    ]
    
    english_only_models = [
        ('tiny.en', 'Fastest, least accurate (English only)'),
        ('base.en', 'Good balance (English only, recommended)'),
        ('small.en', 'Better accuracy (English only)'),
        ('medium.en', 'High accuracy (English only)')
    ]
    
    print("\n" + "="*60)
    print("Model Selection")
    print("="*60)
    print("\nChoose your default Whisper model:")
    print()
    print("Multilingual models (support all languages, auto-detect):")
    for i, (model, desc) in enumerate(multilingual_models, 1):
        print(f"  [{i}] {model:12} - {desc}")
    
    print("\nEnglish-only models (smaller, faster, English only):")
    for i, (model, desc) in enumerate(english_only_models, len(multilingual_models) + 1):
        print(f"  [{i}] {model:12} - {desc}")
    print()
    
    all_models = [m[0] for m in multilingual_models] + [m[0] for m in english_only_models]
    choices = [str(i) for i in range(1, len(all_models) + 1)]
    
    while True:
        try:
            choice = Prompt.ask("Select model", choices=choices, default='2')  # default to base
            selected_model = all_models[int(choice) - 1]
            print(f"\n✓ Selected: {selected_model}")
            return selected_model
        except (ValueError, IndexError, KeyboardInterrupt):
            print("\nInvalid selection. Please try again.")
            continue


def setup_command():
    """Interactive full initial setup"""
    print("\n" + "="*60)
    print("hyprwhspr setup")
    print("="*60)
    print("\nThis setup will guide you through configuring hyprwhspr.")
    print("Skip any step by answering 'no'.\n")
    
    # Step 1: Backend selection
    backend = _prompt_backend_selection()
    if not backend:
        log_error("Backend selection is required. Exiting.")
        return
    
    # Step 1.5: Backend installation (if not remote)
    if backend != 'remote':
        print("\n" + "="*60)
        print("Backend Installation")
        print("="*60)
        print(f"\nThis will install the {backend.upper()} backend for pywhispercpp.")
        print("This may take several minutes as it compiles from source.")
        if not Confirm.ask("Proceed with backend installation?", default=True):
            log_warning("Skipping backend installation. You can install it later.")
            log_warning("Backend installation is required for local transcription to work.")
        else:
            if not install_backend(backend):
                log_error("Backend installation failed. Setup cannot continue.")
                return
    
    # Step 2: Model selection (if local backend)
    selected_model = None
    if backend != 'remote':
        selected_model = _prompt_model_selection()
    
    # Step 3: Waybar integration
    print("\n" + "="*60)
    print("Waybar Integration")
    print("="*60)
    waybar_config_path = Path.home() / '.config' / 'waybar' / 'config.jsonc'
    waybar_style_path = Path.home() / '.config' / 'waybar' / 'style.css'
    waybar_installed = waybar_config_path.exists() or waybar_style_path.exists()
    
    if waybar_installed:
        print(f"\nWaybar configuration detected at: {waybar_config_path.parent}")
        setup_waybar_choice = Confirm.ask("Configure Waybar integration?", default=True)
    else:
        print("\nWaybar configuration not found.")
        setup_waybar_choice = Confirm.ask("Set up Waybar integration anyway?", default=False)
    
    # Step 4: Systemd setup
    print("\n" + "="*60)
    print("Systemd Service")
    print("="*60)
    print("\nSystemd user service will run hyprwhspr in the background.")
    setup_systemd_choice = Confirm.ask("Set up systemd user service?", default=True)
    
    # Step 5: Permissions setup
    print("\n" + "="*60)
    print("Permissions Setup")
    print("="*60)
    print("\nAdds you to required groups (input, audio, tty)")
    print("and configures uinput device permissions.")
    print("Note: Requires sudo access.")
    setup_permissions_choice = Confirm.ask("Set up permissions?", default=True)
    
    # Summary
    print("\n" + "="*60)
    print("Setup Summary")
    print("="*60)
    print(f"\nBackend: {backend}")
    if selected_model:
        print(f"Model: {selected_model}")
    print(f"Waybar integration: {'Yes' if setup_waybar_choice else 'No'}")
    print(f"Systemd service: {'Yes' if setup_systemd_choice else 'No'}")
    print(f"Permissions: {'Yes' if setup_permissions_choice else 'No'}")
    print()
    
    # Final confirmation
    if not Confirm.ask("Proceed with setup?", default=True):
        print("\nSetup cancelled.")
        return
    
    print("\n" + "="*60)
    print("Running Setup")
    print("="*60 + "\n")
    
    # Execute selected steps
    try:
        # Step 1: Config
        setup_config(backend=backend, model=selected_model)
        
        # Step 2: Waybar
        if setup_waybar_choice:
            setup_waybar('install')
        else:
            log_info("Skipping Waybar integration")
        
        # Step 3: Systemd
        if setup_systemd_choice:
            setup_systemd('install')
        else:
            log_info("Skipping systemd setup")
        
        # Step 4: Permissions
        if setup_permissions_choice:
            setup_permissions()
        else:
            log_info("Skipping permissions setup")
        
        # Step 5: Model download (if local backend)
        if backend != 'remote' and selected_model:
            try:
                # Try modern layout first
                try:
                    from pywhispercpp.model import Model  # noqa: F401
                    pywhispercpp_available = True
                except ImportError:
                    # Fallback for flat layout
                    try:
                        from pywhispercpp import Model  # noqa: F401
                        pywhispercpp_available = True
                    except ImportError:
                        pywhispercpp_available = False
            except ImportError:
                pywhispercpp_available = False
            
            if pywhispercpp_available:
                print(f"\nDownloading model: {selected_model}")
                download_model(selected_model)
            else:
                log_warning(f"{backend.upper()} backend selected but pywhispercpp not available")
                log_warning("Backend installation may have failed or is incomplete")
                log_info("You can download the model later with: hyprwhspr model download")
        
        # Step 6: Validation
        print("\n" + "="*60)
        print("Validation")
        print("="*60 + "\n")
        validate_command()
        
        print("\n" + "="*60)
        log_success("Setup completed!")
        print("="*60)
        print("\nNext steps:")
        if setup_systemd_choice:
            print("  1. Log out and back in (for group permissions to take effect)")
            print("  2. Enable and start the service: systemctl --user enable --now hyprwhspr")
        else:
            print("  1. Log out and back in (if permissions were set up)")
            print("  2. Run hyprwhspr manually or set up systemd service later")
        print("  3. Press Super+Alt+D to start dictation!")
        print()
        
    except KeyboardInterrupt:
        print("\n\nSetup interrupted by user.")
        sys.exit(1)
    except Exception as e:
        log_error(f"Setup failed: {e}")
        sys.exit(1)


# ==================== Config Commands ====================

def config_command(action: str):
    """Handle config subcommands"""
    if action == 'init':
        setup_config()
    elif action == 'show':
        show_config()
    elif action == 'edit':
        edit_config()
    else:
        log_error(f"Unknown config action: {action}")


def setup_config(backend: Optional[str] = None, model: Optional[str] = None):
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
        config.save_config()
        log_success(f"Created {config_file}")
    else:
        log_info(f"Config already exists at {config_file}")
        # Update existing config if needed
        try:
            with open(config_file, 'r', encoding='utf-8') as f:
                existing_config = json.load(f)
            
            # Update backend if provided (accept both old 'local' and new backend types)
            if backend:
                # Map old 'local' to 'cpu' for backward compatibility
                if backend == 'local':
                    backend = 'cpu'
                existing_config['transcription_backend'] = backend
            
            # Update model if provided, otherwise default to base.en if missing
            if model:
                existing_config['model'] = model
            elif 'model' not in existing_config:
                existing_config['model'] = 'base.en'
            
            # Add audio_feedback if missing
            if 'audio_feedback' not in existing_config:
                existing_config['audio_feedback'] = True
                existing_config['start_sound_volume'] = 0.5
                existing_config['stop_sound_volume'] = 0.5
                existing_config['start_sound_path'] = 'ping-up.ogg'
                existing_config['stop_sound_path'] = 'ping-down.ogg'
            
            with open(config_file, 'w', encoding='utf-8') as f:
                json.dump(existing_config, f, indent=2)
            
            log_success("Updated existing config")
        except Exception as e:
            log_error(f"Failed to update config: {e}")


def show_config():
    """Display current config"""
    config_file = USER_CONFIG_DIR / 'config.json'
    
    if not config_file.exists():
        log_error("Config file not found. Run 'hyprwhspr config init' first.")
        return
    
    try:
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


# ==================== Systemd Commands ====================

def systemd_command(action: str):
    """Handle systemd subcommands"""
    if action == 'install':
        setup_systemd('install')
    elif action == 'enable':
        setup_systemd('enable')
    elif action == 'disable':
        setup_systemd('disable')
    elif action == 'status':
        systemd_status()
    elif action == 'restart':
        systemd_restart()
    else:
        log_error(f"Unknown systemd action: {action}")


def setup_systemd(mode: str = 'install'):
    """Setup systemd user service"""
    log_info("Configuring systemd user services...")
    
    # Validate HYPRWHSPR_ROOT
    if not _validate_hyprwhspr_root():
        return False
    
    # Validate main executable exists
    main_exec = Path(HYPRWHSPR_ROOT) / 'bin' / 'hyprwhspr'
    if not main_exec.exists() or not os.access(main_exec, os.X_OK):
        log_error(f"Main executable not found or not executable: {main_exec}")
        return False
    
    # Create user systemd directory
    USER_SYSTEMD_DIR.mkdir(parents=True, exist_ok=True)
    
    # Read service file template and substitute paths
    service_source = Path(HYPRWHSPR_ROOT) / 'config' / 'systemd' / SERVICE_NAME
    service_dest = USER_SYSTEMD_DIR / SERVICE_NAME
    
    if not service_source.exists():
        log_error(f"Service file not found: {service_source}")
        return False
    
    # Read template and substitute HYPRWHSPR_ROOT
    try:
        with open(service_source, 'r', encoding='utf-8') as f:
            service_content = f.read()
        
        # Substitute hardcoded path with actual HYPRWHSPR_ROOT
        service_content = service_content.replace('/usr/lib/hyprwhspr', HYPRWHSPR_ROOT)
        
        # Write substituted content to user directory
        with open(service_dest, 'w', encoding='utf-8') as f:
            f.write(service_content)
        
        log_success("User service file created with correct paths")
    except IOError as e:
        log_error(f"Failed to read/write service file: {e}")
        return False
    
    # Reload systemd daemon
    run_command(['systemctl', '--user', 'daemon-reload'], check=False)
    
    if mode in ('install', 'enable'):
        # Enable & start services
        run_command(['systemctl', '--user', 'enable', '--now', YDOTOOL_UNIT], check=False)
        run_command(['systemctl', '--user', 'enable', '--now', SERVICE_NAME], check=False)
        log_success("Systemd user services enabled and started")
    elif mode == 'disable':
        run_command(['systemctl', '--user', 'disable', '--now', SERVICE_NAME], check=False)
        log_success("Systemd user service disabled")
    
    return True


def systemd_status():
    """Show systemd service status"""
    try:
        run_command(['systemctl', '--user', 'status', SERVICE_NAME], check=False)
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to get status: {e}")


def systemd_restart():
    """Restart systemd service"""
    log_info("Restarting service...")
    try:
        run_command(['systemctl', '--user', 'restart', SERVICE_NAME], check=False)
        log_success("Service restarted")
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to restart service: {e}")


# ==================== Waybar Commands ====================

def waybar_command(action: str):
    """Handle waybar subcommands"""
    if action == 'install':
        setup_waybar('install')
    elif action == 'remove':
        setup_waybar('remove')
    elif action == 'status':
        waybar_status()
    else:
        log_error(f"Unknown waybar action: {action}")


def setup_waybar(mode: str = 'install'):
    """Setup or remove waybar integration"""
    if mode == 'install':
        log_info("Setting up Waybar integration...")
    else:
        log_info("Removing Waybar integration...")
    
    # Validate HYPRWHSPR_ROOT
    if not _validate_hyprwhspr_root():
        return False
    
    # Validate required files exist
    tray_script = Path(HYPRWHSPR_ROOT) / 'config' / 'hyprland' / 'hyprwhspr-tray.sh'
    css_file = Path(HYPRWHSPR_ROOT) / 'config' / 'waybar' / 'hyprwhspr-style.css'
    
    if not tray_script.exists():
        log_error(f"Tray script not found: {tray_script}")
        return False
    
    if not css_file.exists():
        log_error(f"Waybar CSS not found: {css_file}")
        return False
    
    waybar_config = USER_HOME / '.config' / 'waybar' / 'config.jsonc'
    waybar_style = USER_HOME / '.config' / 'waybar' / 'style.css'
    user_module_config = USER_HOME / '.config' / 'waybar' / 'hyprwhspr-module.jsonc'
    
    if mode == 'install':
        # Create waybar config directory
        waybar_config.parent.mkdir(parents=True, exist_ok=True)
        
        # Create basic waybar config if it doesn't exist
        if not waybar_config.exists():
            log_info("Creating basic Waybar config...")
            basic_config = {
                "layer": "top",
                "position": "top",
                "height": 30,
                "modules-left": ["hyprland/workspaces"],
                "modules-center": ["hyprland/window"],
                "modules-right": ["custom/hyprwhspr", "clock", "tray"],
                "include": [str(user_module_config)]
            }
            with open(waybar_config, 'w', encoding='utf-8') as f:
                json.dump(basic_config, f, indent=2)
            log_success("Created basic Waybar config")
        
        # Create user module config
        module_config = {
            "custom/hyprwhspr": {
                "format": "{}",
                "exec": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh status",
                "interval": 1,
                "return-type": "json",
                "exec-on-event": True,
                "on-click": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh toggle",
                "on-click-right": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh start",
                "on-click-middle": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh restart",
                "tooltip": True
            }
        }
        
        with open(user_module_config, 'w', encoding='utf-8') as f:
            json.dump(module_config, f, indent=2)
        
        # Update main waybar config
        try:
            config = _load_jsonc(waybar_config)
            
            # Add include if not present
            if 'include' not in config:
                config['include'] = []
            
            if str(user_module_config) not in config['include']:
                config['include'].append(str(user_module_config))
            
            # Add module to modules-right if not present
            if 'modules-right' not in config:
                config['modules-right'] = []
            
            if 'custom/hyprwhspr' not in config['modules-right']:
                config['modules-right'].insert(0, 'custom/hyprwhspr')
            
            # Write back
            with open(waybar_config, 'w', encoding='utf-8') as f:
                json.dump(config, f, indent=2, separators=(',', ': '))
            
            log_success("Waybar config updated")
        except json.JSONDecodeError as e:
            log_error(f"Failed to parse waybar config.jsonc (after stripping comments): {e}")
            log_error("Please check your config.jsonc for JSON syntax errors.")
            return False
        except IOError as e:
            log_error(f"Failed to read/write waybar config: {e}")
            return False
        
        # Add CSS import
        if waybar_style.exists():
            with open(waybar_style, 'r', encoding='utf-8') as f:
                css_content = f.read()
            
            import_line = f'@import "{css_file}";'
            
            if import_line not in css_content:
                with open(waybar_style, 'w', encoding='utf-8') as f:
                    f.write(import_line + '\n' + css_content)
                log_success("CSS import added to waybar style.css")
            else:
                log_info("CSS import already present")
        else:
            log_warning("No waybar style.css found - user will need to add CSS import manually")
        
        log_success("Waybar integration installed")
    
    elif mode == 'remove':
        # Remove from config
        if waybar_config.exists():
            try:
                config = _load_jsonc(waybar_config)
                
                # Remove from include
                if 'include' in config and str(user_module_config) in config['include']:
                    config['include'].remove(str(user_module_config))
                
                # Remove from modules-right
                if 'modules-right' in config and 'custom/hyprwhspr' in config['modules-right']:
                    config['modules-right'].remove('custom/hyprwhspr')
                
                with open(waybar_config, 'w', encoding='utf-8') as f:
                    json.dump(config, f, indent=2)
                
                log_success("Removed from waybar config")
            except json.JSONDecodeError as e:
                log_error(f"Failed to parse waybar config.jsonc (after stripping comments): {e}")
                log_error("Please check your config.jsonc for JSON syntax errors.")
            except IOError as e:
                log_error(f"Failed to read/write waybar config: {e}")
        
        # Remove module config file
        if user_module_config.exists():
            user_module_config.unlink()
            log_success("Removed waybar module config")
        
        # Remove CSS import
        if waybar_style.exists():
            try:
                with open(waybar_style, 'r', encoding='utf-8') as f:
                    css_content = f.read()
                
                import_line = f'@import "{css_file}";'
                if import_line in css_content:
                    # Remove with newline
                    css_content = css_content.replace(import_line + '\n', '')
                    # Remove without newline (in case it's at the end)
                    css_content = css_content.replace(import_line, '')
                    
                    with open(waybar_style, 'w', encoding='utf-8') as f:
                        f.write(css_content)
                    log_success("Removed CSS import")
            except IOError as e:
                log_error(f"Failed to remove CSS import: {e}")
    
    return True


def waybar_status():
    """Check if waybar is configured"""
    waybar_config = USER_HOME / '.config' / 'waybar' / 'config.jsonc'
    user_module_config = USER_HOME / '.config' / 'waybar' / 'hyprwhspr-module.jsonc'
    
    if not waybar_config.exists():
        log_warning("Waybar config not found")
        return False
    
    try:
        config = _load_jsonc(waybar_config)
        
        has_module = 'custom/hyprwhspr' in config.get('modules-right', [])
        has_include = str(user_module_config) in config.get('include', [])
        has_module_file = user_module_config.exists()
        
        if has_module and has_include and has_module_file:
            log_success("Waybar is configured for hyprwhspr")
            return True
        else:
            log_warning("Waybar is partially configured")
            if not has_module:
                log_info("  - Module not in modules-right")
            if not has_include:
                log_info("  - Module config not in include")
            if not has_module_file:
                log_info("  - Module config file missing")
            return False
    except json.JSONDecodeError as e:
        log_error(f"Failed to parse waybar config.jsonc (after stripping comments): {e}")
        return False
    except IOError as e:
        log_error(f"Failed to check waybar status: {e}")
        return False


# ==================== Model Commands ====================

def model_command(action: str, model_name: str = 'base.en'):
    """Handle model subcommands"""
    if action == 'download':
        download_model(model_name)
    elif action == 'list':
        list_models()
    elif action == 'status':
        model_status()
    else:
        log_error(f"Unknown model action: {action}")


def download_model(model_name: str = 'base.en'):
    """Download pywhispercpp model"""
    log_info(f"Downloading pywhispercpp model: {model_name}")
    
    PYWHISPERCPP_MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    model_file = PYWHISPERCPP_MODELS_DIR / f'ggml-{model_name}.bin'
    model_url = f"https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-{model_name}.bin"
    
    if model_file.exists():
        file_size = model_file.stat().st_size
        if file_size > 100000000:  # > 100MB, probably valid
            log_success(f"Model already exists: {model_file}")
            return True
        else:
            log_warning("Existing model appears invalid, re-downloading...")
            model_file.unlink()
    
    log_info(f"Fetching {model_url}")
    try:
        import urllib.request
        urllib.request.urlretrieve(model_url, model_file)
        log_success(f"Model downloaded: {model_file}")
        return True
    except (urllib.error.URLError, IOError) as e:
        log_error(f"Failed to download model: {e}")
        return False


def list_models():
    """List available models"""
    # Multilingual models (support all languages, auto-detect)
    multilingual_models = [
        'tiny',      # Fastest, least accurate
        'base',      # Good balance (recommended)
        'small',     # Better accuracy
        'medium',    # High accuracy
        'large',     # Best accuracy, requires GPU
        'large-v3'   # Latest large model, requires GPU
    ]
    
    # English-only models (smaller, faster, English only)
    english_only_models = [
        'tiny.en',   # Fastest, least accurate
        'base.en',   # Good balance
        'small.en',  # Better accuracy
        'medium.en'  # High accuracy
    ]
    
    print("Available models:\n")
    
    print("Multilingual models (support all languages, auto-detect):")
    for model in multilingual_models:
        size_note = " (requires GPU)" if model in ('large', 'large-v3') else ""
        print(f"  - {model}{size_note}")
    
    print("\nEnglish-only models (smaller, faster, English only):")
    for model in english_only_models:
        print(f"  - {model}")
    
    print("\nNote: Use multilingual models for non-English languages or mixed-language content.")
    print("      Use English-only (.en) models for English-only content (smaller file size).")


def model_status():
    """Check installed models"""
    if not PYWHISPERCPP_MODELS_DIR.exists():
        log_warning("Models directory does not exist")
        return
    
    models = list(PYWHISPERCPP_MODELS_DIR.glob('ggml-*.bin'))
    
    if not models:
        log_warning("No models installed")
        return
    
    print("Installed models:")
    for model in sorted(models):
        size = model.stat().st_size / (1024 * 1024)  # MB
        print(f"  - {model.name} ({size:.1f} MB)")


# ==================== Status Command ====================

def status_command():
    """Overall status check"""
    log_info("Checking hyprwhspr status...")
    
    # Check systemd service
    print("\n[Systemd Service]")
    try:
        result = run_command(
            ['systemctl', '--user', 'is-active', SERVICE_NAME],
            check=False,
            capture_output=True
        )
        if result.returncode == 0:
            log_success(f"Service is active")
        else:
            log_warning(f"Service is not active")
    except subprocess.CalledProcessError as e:
        log_error(f"Failed to check service: {e}")
    
    # Check waybar config
    print("\n[Waybar Integration]")
    waybar_status()
    
    # Check user config
    print("\n[User Config]")
    config_file = USER_CONFIG_DIR / 'config.json'
    if config_file.exists():
        log_success(f"Config exists: {config_file}")
    else:
        log_warning("Config file not found")
    
    # Check models
    print("\n[Models]")
    model_status()
    
    # Check permissions
    print("\n[Permissions]")
    check_permissions()


def check_permissions():
    """Check user permissions"""
    import grp
    
    # Get username safely
    username = os.environ.get('SUDO_USER') or os.environ.get('USER') or getpass.getuser()
    if not username:
        log_error("Could not determine username for permissions check")
        return
    
    # Check groups
    user_groups = [g.gr_name for g in grp.getgrall() if username in g.gr_mem]
    user_groups.append(grp.getgrgid(os.getgid()).gr_name)
    
    required_groups = ['input', 'audio', 'tty']
    for group in required_groups:
        if group in user_groups:
            log_success(f"User in '{group}' group")
        else:
            log_warning(f"User NOT in '{group}' group")
    
    # Check uinput
    uinput_path = Path('/dev/uinput')
    if uinput_path.exists():
        if os.access(uinput_path, os.R_OK | os.W_OK):
            log_success("/dev/uinput is accessible")
        else:
            log_warning("/dev/uinput exists but is not accessible")
    else:
        log_warning("/dev/uinput does not exist")


# ==================== Permissions Setup ====================

def setup_permissions():
    """Setup permissions (requires sudo)"""
    log_info("Setting up permissions...")
    
    # Safer way to get username
    username = os.environ.get('SUDO_USER') or os.environ.get('USER') or getpass.getuser()
    if not username:
        log_error("Could not determine username for permissions setup.")
        return False
    
    # Add user to required groups
    try:
        run_sudo_command(['usermod', '-a', '-G', 'input,audio,tty', username], check=False)
        log_success("Added user to required groups")
    except Exception as e:
        log_warning(f"Failed to add user to groups: {e}")
    
    # Load uinput module
    if not Path('/dev/uinput').exists():
        log_info("Loading uinput module...")
        run_sudo_command(['modprobe', 'uinput'], check=False)
        import time
        time.sleep(2)
    
    # Create udev rule
    udev_rule = Path('/etc/udev/rules.d/99-uinput.rules')
    if not udev_rule.exists():
        log_info("Creating udev rule...")
        rule_content = '# Allow members of the input group to access uinput device\nKERNEL=="uinput", GROUP="input", MODE="0660"\n'
        try:
            run_sudo_command(['tee', str(udev_rule)], input_data=rule_content.encode(), check=False)
            log_success("udev rule created")
        except Exception as e:
            log_warning(f"Failed to create udev rule: {e}")
    else:
        log_info("udev rule already exists")
    
    # Reload udev
    try:
        run_sudo_command(['udevadm', 'control', '--reload-rules'], check=False)
        run_sudo_command(['udevadm', 'trigger', '--name-match=uinput'], check=False)
        log_success("udev rules reloaded")
    except Exception as e:
        log_warning(f"Failed to reload udev rules: {e}")
    
    log_warning("You may need to log out/in for new group memberships to apply")


# ==================== Validate Command ====================

def validate_command():
    """Validate installation"""
    log_info("Validating installation...")
    
    all_ok = True
    
    # Validate HYPRWHSPR_ROOT first
    if not _validate_hyprwhspr_root():
        all_ok = False
        return all_ok
    
    # Check static files
    required_files = [
        Path(HYPRWHSPR_ROOT) / 'bin' / 'hyprwhspr',
        Path(HYPRWHSPR_ROOT) / 'lib' / 'main.py',
        Path(HYPRWHSPR_ROOT) / 'config' / 'systemd' / SERVICE_NAME,
    ]
    
    for file_path in required_files:
        if file_path.exists():
            log_success(f"✓ {file_path.name} exists")
        else:
            log_error(f"✗ {file_path.name} missing")
            all_ok = False
    
    # Check Python imports
    try:
        import sounddevice  # noqa: F401
        log_success("✓ sounddevice available")
    except ImportError:
        log_error("✗ sounddevice not available")
        all_ok = False
    
    # Check pywhispercpp (optional)
    try:
        # Try modern layout first
        try:
            from pywhispercpp.model import Model  # noqa: F401
            pywhispercpp_available = True
        except ImportError:
            # Fallback for flat layout
            try:
                from pywhispercpp import Model  # noqa: F401
                pywhispercpp_available = True
            except ImportError:
                pywhispercpp_available = False
    except ImportError:
        pywhispercpp_available = False
    
    if pywhispercpp_available:
        log_success("✓ pywhispercpp available")
    else:
        log_warning("⚠ pywhispercpp not available")
        print("")
        print("To use local transcription, run: hyprwhspr setup")
        print("This will install the backend (CPU/NVIDIA/AMD) of your choice.")
        print("(or use remote backend by setting 'transcription_backend': 'remote' in config.json)")
        print("")
    
    # Check base model
    model_file = PYWHISPERCPP_MODELS_DIR / 'ggml-base.en.bin'
    if model_file.exists():
        log_success(f"✓ Base model exists: {model_file}")
    else:
        log_warning(f"⚠ Base model missing: {model_file}")
    
    if all_ok:
        log_success("Validation passed")
    else:
        log_error("Validation failed - some components are missing")
    
    return all_ok

