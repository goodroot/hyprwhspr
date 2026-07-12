#!/usr/bin/env bash
#
# hyprwhspr bootstrap installer
#
# Usage:
#   curl -fsSL https://hyprwhspr.com/install.sh | bash
#
# Clones (or updates) hyprwhspr to a managed location, installs distro
# dependencies, then runs interactive setup. Re-running is the update path.

set -euo pipefail

REPO_URL="https://github.com/goodroot/hyprwhspr.git"
CLONE_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/hyprwhspr/src"

BLUE='\033[0;34m'
GREEN='\033[0;32m'
RED='\033[0;31m'
NC='\033[0m'

log()  { echo -e "${BLUE}[INFO]${NC} $1"; }
ok()   { echo -e "${GREEN}[OK]${NC} $1"; }
die()  { echo -e "${RED}[ERROR]${NC} $1" >&2; exit 1; }

echo ""
echo -e "${BLUE}============================================================${NC}"
echo -e "${BLUE}  hyprwhspr installer${NC}"
echo -e "${BLUE}============================================================${NC}"
echo ""

if [[ -f /etc/os-release ]]; then
    # shellcheck disable=SC1091
    . /etc/os-release
    if [[ "${ID:-}" == "arch" || " ${ID_LIKE:-} " == *" arch "* ]]; then
        log "Arch-based system detected — hyprwhspr is on the AUR:"
        echo ""
        echo "  yay -S hyprwhspr        # stable"
        echo "  yay -S hyprwhspr-git    # bleeding edge"
        echo ""
        echo "Then run: hyprwhspr setup"
        exit 0
    fi
fi

{ : </dev/tty; } 2>/dev/null || die "This installer is interactive and needs a terminal.
Run it from a shell: curl -fsSL https://hyprwhspr.com/install.sh | bash"

command -v git >/dev/null 2>&1 || die "git is required — install it with your package manager and re-run."

if [[ -d "$CLONE_DIR/.git" ]]; then
    log "Existing install found — updating $CLONE_DIR"
    git -C "$CLONE_DIR" pull --ff-only || die "Update failed — resolve manually in $CLONE_DIR and re-run."
else
    log "Cloning hyprwhspr to $CLONE_DIR"
    mkdir -p "$(dirname "$CLONE_DIR")"
    git clone "$REPO_URL" "$CLONE_DIR"
fi
ok "Source ready at $CLONE_DIR"

log "Installing distro dependencies..."
bash "$CLONE_DIR/scripts/install-deps.sh"

log "Starting interactive setup..."
"$CLONE_DIR/bin/hyprwhspr" setup </dev/tty

echo ""
ok "Done!"
echo "  Installed at:  $CLONE_DIR"
echo "  Update:        re-run this installer"
echo "  Uninstall:     hyprwhspr uninstall"
echo ""
