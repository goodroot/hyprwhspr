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

# Detect current branch state
CURRENT_BRANCH=$(git rev-parse --abbrev-ref HEAD 2>/dev/null || echo "HEAD")
DEFAULT_BRANCH="main"

# Check if we're in a detached HEAD state
if [ "$CURRENT_BRANCH" = "HEAD" ]; then
  log_warning "Repository is in detached HEAD state"
  log_info "Checking out default branch: $DEFAULT_BRANCH"
  
  # Try to checkout the default branch
  if git checkout "$DEFAULT_BRANCH" 2>/dev/null; then
    CURRENT_BRANCH="$DEFAULT_BRANCH"
    log_success "Switched to branch: $DEFAULT_BRANCH"
  else
    # If branch doesn't exist locally, try to track it from origin
    log_info "Branch $DEFAULT_BRANCH not found locally, attempting to track from origin"
    if git checkout -b "$DEFAULT_BRANCH" "origin/$DEFAULT_BRANCH" 2>/dev/null; then
      CURRENT_BRANCH="$DEFAULT_BRANCH"
      log_success "Created and switched to branch: $DEFAULT_BRANCH"
    else
      log_error "Failed to checkout branch. Please manually checkout a branch and try again"
      exit 1
    fi
  fi
fi

# Use shallow fetch for faster updates
log_info "Fetching latest changes (shallow)..."
if ! git fetch --depth=1 origin "$CURRENT_BRANCH" 2>/dev/null; then
  # Fallback to regular fetch if shallow fails
  log_warning "Shallow fetch failed, trying regular fetch..."
  if ! git fetch origin "$CURRENT_BRANCH"; then
    log_error "Failed to fetch latest changes from git repository"
    log_info "Please check your git configuration and network connection"
    exit 1
  fi
fi

# Merge or fast-forward to latest
log_info "Updating to latest changes..."
if ! git merge --ff-only "origin/$CURRENT_BRANCH" 2>/dev/null; then
  # If fast-forward fails, try regular merge
  log_warning "Fast-forward merge not possible, attempting regular merge..."
  if ! git merge "origin/$CURRENT_BRANCH"; then
    log_error "Failed to merge latest changes"
    log_info "You may have local changes that conflict. Please resolve conflicts manually."
    exit 1
  fi
fi

log_success "Git repository updated (branch: $CURRENT_BRANCH)"

# ----------------------- Run installer with update flag --------------
log_info "Updating installation..."

if [ ! -f "$SCRIPT_DIR/install-omarchy.sh" ]; then
  log_error "Installer script not found: $SCRIPT_DIR/install-omarchy.sh"
  exit 1
fi

# Execute the installer with --force and --update flags
"$SCRIPT_DIR/install-omarchy.sh" --force --update

