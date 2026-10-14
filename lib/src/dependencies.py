"""Dependency validation utilities for hyprwhspr"""
import sys
from typing import Any, Optional

try:
    from .installation import repair_hint
except ImportError:
    from installation import repair_hint


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
        hint = repair_hint(install_hint or 'hyprwhspr setup\n  Source checkout: ./bin/hyprwhspr setup')
        print(f"  {hint}", file=sys.stderr)
        sys.exit(1)
