"""
Waybar integration commands for hyprwhspr
"""

import json
from pathlib import Path

try:
    from ..output_control import log_info, log_success, log_warning, log_error
except ImportError:
    from output_control import log_info, log_success, log_warning, log_error

from ._shared import (HYPRWHSPR_ROOT, USER_HOME, _load_jsonc,
                      _validate_hyprwhspr_root)


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
                "on-click": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh record",
                "on-click-right": f"{HYPRWHSPR_ROOT}/config/hyprland/hyprwhspr-tray.sh restart",
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
                # Try to insert after group/tray-expander
                try:
                    tray_index = config['modules-right'].index('group/tray-expander')
                    config['modules-right'].insert(tray_index + 1, 'custom/hyprwhspr')
                except ValueError:
                    # group/tray-expander not found, append to end
                    config['modules-right'].append('custom/hyprwhspr')
            
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
