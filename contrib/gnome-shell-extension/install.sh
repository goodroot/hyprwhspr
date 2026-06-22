#!/usr/bin/env bash
#
# Install the hyprwhspr Waveform OSD GNOME Shell extension.
#
# Copies the extension into ~/.local/share/gnome-shell/extensions and enables it.
# On Wayland a brand-new extension can't be hot-loaded, so if `enable` can't see
# it yet we add it to the enabled list and it activates on your next login.

set -euo pipefail

UUID="hyprwhspr-waveform@ninyawee.github.io"
SRC="$(cd "$(dirname "$0")" && pwd)/$UUID"
DEST="${XDG_DATA_HOME:-$HOME/.local/share}/gnome-shell/extensions/$UUID"

if [[ ! -d "$SRC" ]]; then
    echo "error: extension source not found at $SRC" >&2
    exit 1
fi

mkdir -p "$(dirname "$DEST")"
rm -rf "$DEST"
cp -r "$SRC" "$DEST"
echo "Installed -> $DEST"

if gnome-extensions enable "$UUID" 2>/dev/null; then
    echo "Enabled. If you don't see the overlay, log out and back in."
else
    # Shell hasn't scanned the new dir yet (typical on Wayland): queue it so it
    # auto-enables after the next login.
    python3 - "$UUID" <<'PY'
import sys, ast, subprocess
uuid = sys.argv[1]
cur = subprocess.run(["gsettings", "get", "org.gnome.shell", "enabled-extensions"],
                     capture_output=True, text=True).stdout.strip()
try:
    lst = ast.literal_eval(cur) if cur and cur not in ("@as []", "") else []
    if not isinstance(lst, list):
        lst = []
except Exception:
    lst = []
if uuid not in lst:
    lst.append(uuid)
val = "[" + ", ".join("'%s'" % x for x in lst) + "]"
subprocess.run(["gsettings", "set", "org.gnome.shell", "enabled-extensions", val], check=True)
PY
    echo "Queued for activation — log out and back in (Wayland can't hot-load a new extension)."
fi

echo
echo "Then start dictation (your hyprwhspr shortcut) and the waveform pill"
echo "appears at the bottom-centre of the screen while recording."
