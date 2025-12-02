#!/bin/bash
# Automatically pulls latest changes and updates the installation

set -euo pipefail

# ----------------------- Colors & logging -----------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ----------------------- Detect repository root --------------------
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"

# ----------------------- Check if we're in a git repository --------
if [ ! -d "$REPO_ROOT/.git" ]; then
  log_error "Not in a git repository. Cannot update."
  log_info "This script is designed for git-based installations."
  log_info "If you installed via AUR, use your package manager to update."
  exit 1
fi

# ----------------------- Pull latest changes -----------------------
log_info "Pulling latest changes from git repository..."
cd "$REPO_ROOT"

if ! git pull; then
  log_error "Failed to pull latest changes from git repository"
  log_info "Please check your git configuration and network connection"
  exit 1
fi

log_success "Git repository updated"

# ----------------------- Run installer with force flag --------------
log_info "Updating installation..."
log_info "Running installer with --force flag to overwrite existing files"

if [ ! -f "$SCRIPT_DIR/install-omarchy.sh" ]; then
  log_error "Installer script not found: $SCRIPT_DIR/install-omarchy.sh"
  exit 1
fi

# Execute the installer with --force flag
"$SCRIPT_DIR/install-omarchy.sh" --force

log_success "✓ hyprwhspr update completed!"

