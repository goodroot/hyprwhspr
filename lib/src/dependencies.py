"""Dependency validation utilities for hyprwhspr"""
import sys
from typing import Any, Optional


def require_package(
    module_name: str,
    package_name: Optional[str] = None,
    install_hint: Optional[str] = None
) -> Any:
    """Import and validate required package.

    Args:
        module_name: Python module name (e.g., 'sounddevice')
        package_name: System package name (e.g., 'python-sounddevice')
                     Defaults to 'python-{module_name}'
        install_hint: Custom install command (optional)

    Returns:
        Imported module object

    Raises:
        SystemExit(1) if package cannot be imported
    """
    try:
        return __import__(module_name)
    except (ImportError, ModuleNotFoundError) as e:
        pkg = package_name or f"python-{module_name}"
        print(f"ERROR: {pkg} is not available in this Python environment.", file=sys.stderr)
        print(f"ImportError: {e}", file=sys.stderr)
        print("\nThis is a required dependency. Please install it:", file=sys.stderr)
        if install_hint:
            print(f"  {install_hint}", file=sys.stderr)
        else:
            print(f"  Arch:          pacman -S {pkg}", file=sys.stderr)
            print(f"  Debian/Ubuntu: apt install python3-{module_name}", file=sys.stderr)
            print(f"  Fedora:        dnf install python3-{module_name}", file=sys.stderr)
            print(f"  openSUSE:      zypper install python3-{module_name}", file=sys.stderr)
            print(f"  Or via pip:    pip install {module_name}", file=sys.stderr)
        sys.exit(1)
