"""
hyprwhspr CLI command package.

Command implementations are split into one module per command family;
shared constants and environment helpers live in _shared.
"""

import importlib.util
import sys

# Probe for python-rich here so every command module can import it bare: the
# package init runs before any submodule, and a missing python-rich gets
# actionable guidance instead of a raw traceback. find_spec checks
# availability without importing, keeping light commands (record) fast.
if 'rich' not in sys.modules and importlib.util.find_spec('rich') is None:
    # Hard fail – rich is required for the CLI
    print("ERROR: python-rich is not available in this Python environment.", file=sys.stderr)
    print(f"\nPython being used: {sys.executable}", file=sys.stderr)
    print(f"Python version:    {sys.version.split()[0]}", file=sys.stderr)
    print("\nThis usually means python-rich is installed for a different Python version.", file=sys.stderr)
    print("The CLI requires system Python with distro packages installed.", file=sys.stderr)
    print("\nTry installing it using your package manager:", file=sys.stderr)
    print("  Arch:          pacman -S python-rich", file=sys.stderr)
    print("  Debian/Ubuntu: apt install python3-rich", file=sys.stderr)
    print("  Fedora:        dnf install python3-rich", file=sys.stderr)
    print("  Or via pip:    pip install rich>=13.0.0", file=sys.stderr)
    print("\nIf using a Python version manager (pyenv, mise, asdf), ensure", file=sys.stderr)
    print("python-rich is installed for your system Python (/usr/bin/python3).", file=sys.stderr)
    sys.exit(1)
