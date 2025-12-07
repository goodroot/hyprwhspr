#!/usr/bin/env python3
"""
hyprwhspr CLI - Command-line interface for managing hyprwhspr
"""

import sys
import argparse
from pathlib import Path

# Add the src directory to the Python path
src_path = Path(__file__).parent / 'src'
sys.path.insert(0, str(src_path))

from cli_commands import (
    setup_command,
    config_command,
    waybar_command,
    systemd_command,
    model_command,
    status_command,
    validate_command,
)


def main():
    """Main CLI entry point"""
    parser = argparse.ArgumentParser(
        prog='hyprwhspr',
        description='hyprwhspr - Voice dictation service for Hyprland',
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # setup command
    subparsers.add_parser('setup', help='Full initial setup')
    
    # config command
    config_parser = subparsers.add_parser('config', help='Configuration management')
    config_subparsers = config_parser.add_subparsers(dest='config_action', help='Config actions')
    config_subparsers.add_parser('init', help='Create default config')
    config_subparsers.add_parser('show', help='Display current config')
    config_subparsers.add_parser('edit', help='Open config in editor')
    
    # waybar command
    waybar_parser = subparsers.add_parser('waybar', help='Waybar integration')
    waybar_subparsers = waybar_parser.add_subparsers(dest='waybar_action', help='Waybar actions')
    waybar_subparsers.add_parser('install', help='Add module to waybar config')
    waybar_subparsers.add_parser('remove', help='Remove module from waybar config')
    waybar_subparsers.add_parser('status', help='Check if waybar is configured')
    
    # systemd command
    systemd_parser = subparsers.add_parser('systemd', help='Systemd service management')
    systemd_subparsers = systemd_parser.add_subparsers(dest='systemd_action', help='Systemd actions')
    systemd_subparsers.add_parser('install', help='Copy service, enable, start')
    systemd_subparsers.add_parser('enable', help='Enable service')
    systemd_subparsers.add_parser('disable', help='Disable service')
    systemd_subparsers.add_parser('status', help='Show service status')
    systemd_subparsers.add_parser('restart', help='Restart service')
    
    # model command
    model_parser = subparsers.add_parser('model', help='Model management')
    model_subparsers = model_parser.add_subparsers(dest='model_action', help='Model actions')
    model_download_parser = model_subparsers.add_parser('download', help='Download model')
    model_download_parser.add_argument('name', nargs='?', default='base.en', help='Model name (default: base.en)')
    model_subparsers.add_parser('list', help='List available models')
    model_subparsers.add_parser('status', help='Check installed models')
    
    # status command
    subparsers.add_parser('status', help='Overall status check')
    
    # validate command
    subparsers.add_parser('validate', help='Validate installation')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        sys.exit(1)
    
    # Route to appropriate command handler
    try:
        if args.command == 'setup':
            setup_command()
        elif args.command == 'config':
            if not args.config_action:
                config_parser.print_help()
                sys.exit(1)
            config_command(args.config_action)
        elif args.command == 'waybar':
            if not args.waybar_action:
                waybar_parser.print_help()
                sys.exit(1)
            waybar_command(args.waybar_action)
        elif args.command == 'systemd':
            if not args.systemd_action:
                systemd_parser.print_help()
                sys.exit(1)
            systemd_command(args.systemd_action)
        elif args.command == 'model':
            if not args.model_action:
                model_parser.print_help()
                sys.exit(1)
            model_name = getattr(args, 'name', 'base.en')
            model_command(args.model_action, model_name)
        elif args.command == 'status':
            status_command()
        elif args.command == 'validate':
            validate_command()
    except KeyboardInterrupt:
        print("\nOperation cancelled by user")
        sys.exit(1)
    except Exception as e:
        print(f"Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == '__main__':
    main()

