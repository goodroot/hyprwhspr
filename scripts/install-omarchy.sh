#!/bin/bash
# hyprwhspr Omarchy/Arch Installation Script
#   • Static files: /usr/lib/hyprwhspr (read-only system files)
#   • Runtime data: ~/.local/share/hyprwhspr (user space, always writable)

set -euo pipefail

# ----------------------- Colors & logging -----------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# Legacy shorthands used in a few places
ok()   { log_success "$@"; }
warn() { log_warning "$@"; }
err()  { log_error "$@"; }

# ----------------------- Configuration -------------------------
PACKAGE_NAME="hyprwhspr"
INSTALL_DIR="/usr/lib/hyprwhspr"  # Always read-only system files
SERVICE_NAME="hyprwhspr.service"
YDOTOOL_UNIT="ydotool.service"

# Always use user space for runtime data (consistent across all installations)
USER_BASE="${XDG_DATA_HOME:-$HOME/.local/share}/hyprwhspr"
VENV_DIR="$USER_BASE/venv"                    # Python virtual environment
USER_WC_DIR="$USER_BASE/whisper.cpp"          # Built whisper.cpp
USER_MODELS_DIR="$USER_WC_DIR/models"         # Downloaded models
USER_BIN_DIR="$HOME/.local/bin"               # User's local bin directory
STATE_DIR="${XDG_STATE_HOME:-$HOME/.local/state}/hyprwhspr"
STATE_FILE="$STATE_DIR/install-state.json"    # Persistent installation state

# ----------------------- Detect actual user --------------------
if [ "$EUID" -eq 0 ]; then
  if [ -n "${SUDO_USER:-}" ]; then
    ACTUAL_USER="$SUDO_USER"
  else
    ACTUAL_USER=$(stat -c '%U' /home 2>/dev/null | head -1 || echo "root")
  fi
else
  ACTUAL_USER="$USER"
fi
USER_HOME=$(getent passwd "$ACTUAL_USER" | cut -d: -f6)
USER_CONFIG_DIR="$USER_HOME/.config/hyprwhspr"

# ----------------------- Command line options ------------------
CHECK_MODE=false
if [ "${1:-}" = "--check" ]; then
  CHECK_MODE=true
  log_info "Running in check mode - no changes will be made"
fi

# ----------------------- Preconditions -------------------------
command -v pacman >/dev/null 2>&1 || { log_error "Arch Linux required."; exit 1; }
log_info "Setting up hyprwhspr for user: $ACTUAL_USER"
log_info "Unified installation approach"
log_info "INSTALL_DIR=$INSTALL_DIR (static application files)"
log_info "USER_BASE=$USER_BASE (runtime data)"
log_info "VENV_DIR=$VENV_DIR"
log_info "USER_WC_DIR=$USER_WC_DIR"

# ----------------------- Helpers -------------------------------
have_system_whisper() { command -v whisper-cli >/dev/null 2>&1; }

# Validate that INSTALL_DIR contains required files
validate_install_dir() {
  local required_files=(
    "bin/hyprwhspr"
    "lib/main.py"
    "requirements.txt"
    "config/hyprland/hyprwhspr-tray.sh"
    "config/systemd/hyprwhspr.service"
  )
  
  for file in "${required_files[@]}"; do
    if [ ! -f "$INSTALL_DIR/$file" ]; then
      log_error "Required file missing: $INSTALL_DIR/$file"
      return 1
    fi
  done
  log_success "All required files present in $INSTALL_DIR"
  return 0
}

ensure_path_contains_local_bin() {
  case ":$PATH:" in *":$USER_BIN_DIR:"*) ;; *) export PATH="$USER_BIN_DIR:$PATH" ;; esac
}

detect_cuda_host_compiler() {
  # Allow explicit override
  if [[ -n "${HYPRWHSPR_CUDA_HOST:-}" && -x "$HYPRWHSPR_CUDA_HOST" ]]; then
    echo "$HYPRWHSPR_CUDA_HOST"; return 0
  fi
  local gcc_major
  gcc_major=$(gcc -dumpfullversion 2>/dev/null | cut -d. -f1 || echo 0)
  # If GCC >= 15, prefer gcc14 if present (nvcc header compat)
  if [[ "$gcc_major" -ge 15 ]] && command -v g++-14 >/dev/null 2>&1; then
    echo "/usr/bin/g++-14"; return 0
  fi
  if command -v g++ >/dev/null 2>&1; then
    command -v g++; return 0
  fi
  echo ""
}

ensure_user_bin_symlink() {
  mkdir -p "$USER_BIN_DIR"
  if [ -x "$USER_WC_DIR/build/bin/whisper-cli" ] && [ ! -e "$USER_BIN_DIR/whisper-cli" ]; then
    ln -s "$USER_WC_DIR/build/bin/whisper-cli" "$USER_BIN_DIR/whisper-cli" || true
    log_info "Linked whisper-cli → $USER_BIN_DIR/whisper-cli"
  fi
}

# ----------------------- State Management ------------------------
# Initialize state directory and file
init_state() {
  mkdir -p "$STATE_DIR"
  if [ ! -f "$STATE_FILE" ]; then
    echo '{}' > "$STATE_FILE"
  fi
}

# Get a value from the state file
get_state() {
  local key="$1"
  if [ -f "$STATE_FILE" ]; then
    python3 -c "import json, sys; data=json.load(open('$STATE_FILE')); print(data.get('$key', ''))" 2>/dev/null || echo ""
  else
    echo ""
  fi
}

# Set a value in the state file
set_state() {
  local key="$1"
  local value="$2"
  init_state
  python3 -c "
import json, sys
try:
    with open('$STATE_FILE', 'r') as f:
        data = json.load(f)
except:
    data = {}
data['$key'] = '$value'
with open('$STATE_FILE', 'w') as f:
    json.dump(data, f, indent=2)
" 2>/dev/null || true
}

# Compute SHA256 hash of a file
compute_file_hash() {
  local file="$1"
  if [ -f "$file" ]; then
    sha256sum "$file" | awk '{print $1}'
  else
    echo ""
  fi
}

# Check if model file is valid (returns 0 if valid, 1 if not)
check_model_validity() {
  local model_file="$1"
  local file_size
  local stored_hash
  local current_hash
  
  if [ ! -f "$model_file" ]; then
    return 1  # File doesn't exist
  fi
  
  file_size=$(stat -c%s "$model_file" 2>/dev/null || echo "0")
  stored_hash=$(get_state "model_base_en_hash")
  current_hash=$(compute_file_hash "$model_file")
  
  # If we have a stored hash and it matches, it's valid
  if [ -n "$stored_hash" ] && [ "$current_hash" = "$stored_hash" ]; then
    return 0
  fi
  
  # If file is reasonable size (>100MB), it's probably valid
  if [ "$file_size" -gt 100000000 ]; then
    return 0
  fi
  
  return 1  # File is corrupted or too small
}


# ----------------------- Installation Plan ----------------------
generate_installation_plan() {
  log_info "Installation Plan:"
  
  # Check Python environment
  local cur_req_hash
  cur_req_hash=$(compute_file_hash "$INSTALL_DIR/requirements.txt")
  local stored_req_hash
  stored_req_hash=$(get_state "requirements_hash")
  
  if [ "$cur_req_hash" != "$stored_req_hash" ] || [ -z "$stored_req_hash" ]; then
    log_info "  • Python env: UPDATE (requirements.txt changed)"
  else
    log_info "  • Python env: OK (up to date)"
  fi
  
  # Check whisper.cpp build
  if [ -d "$USER_WC_DIR/.git" ]; then
    local current_rev
    current_rev=$(cd "$USER_WC_DIR" && git rev-parse HEAD 2>/dev/null || echo "")
    local stored_rev
    stored_rev=$(get_state "whisper_cpp_rev")
    
    if [ "$current_rev" != "$stored_rev" ] || [ -z "$stored_rev" ]; then
      log_info "  • whisper.cpp: BUILD (new revision or not built)"
    else
      log_info "  • whisper.cpp: OK (up to date)"
    fi
  else
    log_info "  • whisper.cpp: BUILD (not cloned)"
  fi
  
  # Check models
  local model_file="$USER_MODELS_DIR/ggml-base.en.bin"
  if check_model_validity "$model_file"; then
    local file_size
    file_size=$(stat -c%s "$model_file" 2>/dev/null || echo "0")
    local stored_hash
    stored_hash=$(get_state "model_base_en_hash")
    
    if [ -n "$stored_hash" ]; then
      log_info "  • model: OK (verified)"
    else
      log_info "  • model: OK (appears valid, ${file_size} bytes)"
    fi
  else
    log_info "  • model: DOWNLOAD (missing or corrupted)"
  fi
  
  # Check waybar config (simplified check)
  local waybar_config="$USER_HOME/.config/waybar/config.jsonc"
  if [ -f "$waybar_config" ] && grep -q "custom/hyprwhspr" "$waybar_config"; then
    log_info "  • waybar: OK (configured)"
  else
    log_info "  • waybar: UPDATE (needs configuration)"
  fi
  
  # Check systemd services
  if systemctl --user is-enabled "$SERVICE_NAME" >/dev/null 2>&1; then
    log_info "  • systemd: OK (enabled)"
  else
    log_info "  • systemd: UPDATE (needs enabling)"
  fi
}

# ----------------------- Install dependencies ------------------
install_system_dependencies() {
  log_info "Ensuring system dependencies..."
  local pkgs=(cmake make git base-devel python pipewire pipewire-alsa pipewire-pulse pipewire-jack ydotool curl)
  
  # Always install waybar when this script runs (it's designed for full setup)
  pkgs+=(waybar)
  log_info "Installing waybar as part of hyprwhspr setup"
  
  local to_install=()
  for p in "${pkgs[@]}"; do pacman -Q "$p" &>/dev/null || to_install+=("$p"); done
  if ((${#to_install[@]})); then
    log_info "Installing: ${to_install[*]}"
    sudo pacman -S --needed --noconfirm "${to_install[@]}"
  fi
  log_info "python-pip not required (venv pip used)"
  log_success "Dependencies ready"
}

# ----------------------- Python environment --------------------
setup_python_environment() {
  log_info "Setting up Python virtual environment…"
  
  # Validate requirements.txt exists
  if [ ! -f "$INSTALL_DIR/requirements.txt" ]; then
    log_error "requirements.txt not found at $INSTALL_DIR/requirements.txt"
    return 1
  fi
  
  # Check if pip install is needed based on requirements.txt hash
  local cur_req_hash
  cur_req_hash=$(compute_file_hash "$INSTALL_DIR/requirements.txt")
  local stored_req_hash
  stored_req_hash=$(get_state "requirements_hash")
  
  # Always use user space for venv
  if [ ! -d "$VENV_DIR" ]; then
    log_info "Creating venv at $VENV_DIR"
    mkdir -p "$(dirname "$VENV_DIR")"
    python -m venv "$VENV_DIR"
  else
    log_info "Venv already exists at $VENV_DIR"
  fi
  
  # Install dependencies from system files
  source "$VENV_DIR/bin/activate"
  pip install --upgrade pip wheel
  
  if [ "$cur_req_hash" != "$stored_req_hash" ] || [ -z "$stored_req_hash" ]; then
    log_info "Installing Python dependencies (requirements.txt changed)"
    pip install -r "$INSTALL_DIR/requirements.txt"
    set_state "requirements_hash" "$cur_req_hash"
    log_success "Python dependencies installed"
  else
    log_info "Python dependencies up to date (skipping pip install)"
  fi
}

# ----------------------- NVIDIA support -----------------------
setup_nvidia_support() {
  log_info "GPU check…"
  if command -v nvidia-smi >/dev/null 2>&1; then
    log_success "NVIDIA GPU detected"
    # nvcc path (prefer canonical CUDA location)
    if [ -x /opt/cuda/bin/nvcc ] || command -v nvcc >/dev/null 2>&1; then
      export PATH="/opt/cuda/bin:$PATH"
      export CUDACXX="${CUDACXX:-/opt/cuda/bin/nvcc}"
      log_success "CUDA toolkit present"
    else
      log_warning "CUDA toolkit not found; installing…"
      sudo pacman -S --needed --noconfirm cuda || true
      if [ -x /opt/cuda/bin/nvcc ]; then
        export PATH="/opt/cuda/bin:$PATH"
        export CUDACXX="/opt/cuda/bin/nvcc"
        log_success "CUDA installed"
      else
        log_warning "nvcc still not visible; will build CPU-only"
        return 0
      fi
    fi

    # Choose a host compiler for NVCC
    local hostc
    hostc="$(detect_cuda_host_compiler)"
    if [[ -n "$hostc" ]]; then
      export CUDAHOSTCXX="$hostc"
      log_info "CUDA host compiler: $CUDAHOSTCXX"
      if [[ "$hostc" == "/usr/bin/g++" ]]; then
        local gcc_major
        gcc_major=$(gcc -dumpfullversion 2>/dev/null | cut -d. -f1 || echo 0)
        if [[ "$gcc_major" -ge 15 ]]; then
          log_warning "GCC $gcc_major with NVCC can fail; consider:"
          log_warning "  yay -S gcc14 gcc14-libs"
          log_warning "  HYPRWHSPR_CUDA_HOST=/usr/bin/g++-14 hyprwhspr-setup"
        fi
      fi
    else
      log_warning "No suitable host compiler found; will build CPU-only"
    fi
  else
    log_info "No NVIDIA GPU detected (CPU mode)"
  fi
}

# ----------------------- whisper.cpp build ---------------------
setup_whisper() {
  log_info "Setting up whisper.cpp…"

  # Always use user space for whisper.cpp
  mkdir -p "$USER_WC_DIR"
  cd "$USER_WC_DIR"

  # ---------- clone or update ----------
  if [ ! -d ".git" ]; then
    log_info "Cloning whisper.cpp → $PWD"
    git clone https://github.com/ggml-org/whisper.cpp.git .
  else
    git pull --ff-only || true
  fi

  # ---------- CUDA availability ----------
  if [ -z "${CUDACXX:-}" ] && [ -x /opt/cuda/bin/nvcc ]; then
    export CUDACXX="/opt/cuda/bin/nvcc"
    export PATH="/opt/cuda/bin:$PATH"
  fi

  local use_cuda=false
  if [[ -n "${CUDACXX:-}" && -x "${CUDACXX}" ]]; then
    if [ -z "${CUDAHOSTCXX:-}" ]; then
      CUDAHOSTCXX="$(detect_cuda_host_compiler)"; export CUDAHOSTCXX
    fi
    if [[ -n "${CUDAHOSTCXX:-}" && -x "${CUDAHOSTCXX}" ]]; then
      use_cuda=true
      log_info "CUDA detected: building with GPU support (host: $CUDAHOSTCXX)"
    else
      log_warning "CUDA present but no suitable host compiler; building CPU-only"
    fi
  else
    log_info "Building CPU-only"
  fi

  # ---------- configure & build ----------
  local build_success=false
  local cuda_build_failed=false
  
  if [[ "$use_cuda" == true ]]; then
    log_info "Attempting CUDA build..."
    if timeout 300 cmake -B build \
      -DGGML_CUDA=ON \
      -DCMAKE_BUILD_TYPE=Release \
      -DCMAKE_CUDA_HOST_COMPILER="${CUDAHOSTCXX}" 2>/dev/null; then
      
      log_info "CUDA configuration successful, building..."
      if timeout 800 cmake --build build -j --config Release 2>/dev/null; then
        # Verify CUDA binary was built correctly
        if [ -x "build/bin/whisper-cli" ] && ldd build/bin/whisper-cli | grep -qi cuda; then
          log_success "CUDA build completed successfully"
          build_success=true
        else
          log_warning "CUDA build completed but binary not CUDA-linked"
          cuda_build_failed=true
        fi
      else
        log_warning "CUDA build timed out or failed"
        cuda_build_failed=true
      fi
    else
      log_warning "CUDA configuration failed"
      cuda_build_failed=true
    fi
  fi
  
  # Fallback to CPU-only build if CUDA failed or wasn't requested
  if [[ "$build_success" != true ]]; then
    if [[ "$cuda_build_failed" == true ]]; then
      log_info "Falling back to CPU-only build due to CUDA issues..."
      log_info "This is common with older NVIDIA cards (RTX 20xx series) and newer CUDA toolchains"
    else
      log_info "Building CPU-only (no CUDA requested)"
    fi
    
    rm -rf build
    cmake -B build \
      -DGGML_CUDA=OFF \
      -DCMAKE_BUILD_TYPE=Release
    
    if cmake --build build -j --config Release; then
      log_success "CPU-only build completed successfully"
      build_success=true
    else
      log_error "CPU-only build failed"
      build_success=false
    fi
  fi

  # ---------- verify binary ----------
  if [[ "$build_success" != true ]] || [ ! -x "build/bin/whisper-cli" ]; then
    err "whisper.cpp build failed"
    exit 1
  fi

  # ---------- link into ~/.local/bin for PATH ----------
  ensure_path_contains_local_bin
  ensure_user_bin_symlink

  ok "whisper.cpp ready"
}

# ----------------------- Models --------------------------------
download_models() {
  log_info "Downloading Whisper base model…"
  mkdir -p "$USER_MODELS_DIR"
  
  local model_file="$USER_MODELS_DIR/ggml-base.en.bin"
  local model_url="https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-base.en.bin"
  
  # Check if model is already valid
  if check_model_validity "$model_file"; then
    local file_size
    file_size=$(stat -c%s "$model_file" 2>/dev/null || echo "0")
    local stored_hash
    stored_hash=$(get_state "model_base_en_hash")
    
    if [ -n "$stored_hash" ]; then
      log_info "Model already present and verified (${file_size} bytes)"
    else
      log_info "Model file exists and appears valid (${file_size} bytes) - storing hash"
      local current_hash
      current_hash=$(compute_file_hash "$model_file")
      set_state "model_base_en_hash" "$current_hash"
    fi
    return 0
  fi
  
  # Model is missing or corrupted, download it
  if [ -f "$model_file" ]; then
    local file_size
    file_size=$(stat -c%s "$model_file" 2>/dev/null || echo "0")
    log_warning "Model file exists but appears corrupted (${file_size} bytes) - re-downloading"
    rm -f "$model_file"
  fi
  
  log_info "Downloading model from $model_url"
  if curl -L --fail -o "$model_file" "$model_url"; then
    local downloaded_hash
    downloaded_hash=$(compute_file_hash "$model_file")
    local file_size
    file_size=$(stat -c%s "$model_file" 2>/dev/null || echo "0")
    set_state "model_base_en_hash" "$downloaded_hash"
    log_success "Model downloaded and verified (${file_size} bytes)"
  else
    log_error "Failed to download model"
    return 1
  fi
}

# ----------------------- Systemd (user) ------------------------
setup_systemd_service() {
  log_info "Configuring systemd user services…"

  # Validate main executable exists
  if [ ! -x "$INSTALL_DIR/bin/hyprwhspr" ]; then
    log_error "Main executable not found or not executable: $INSTALL_DIR/bin/hyprwhspr"
    return 1
  fi

  # Create user systemd directory
  mkdir -p "$USER_HOME/.config/systemd/user"
  
  # Copy service file from package
  if [ -f "$INSTALL_DIR/config/systemd/$SERVICE_NAME" ]; then
    cp "$INSTALL_DIR/config/systemd/$SERVICE_NAME" "$USER_HOME/.config/systemd/user/" || true
  fi
  
  log_success "User services created"

  # Reload systemd daemon
  systemctl --user daemon-reload
  # Enable & start services
  systemctl --user enable --now "$YDOTOOL_UNIT" 2>/dev/null || true
  systemctl --user enable --now "$SERVICE_NAME"

  log_success "Systemd user services enabled and started"
}

# ----------------------- Hyprland integration ------------------
setup_hyprland_integration() {
  log_info "Setting up Hyprland integration…"
  
  # Validate tray script exists
  if [ ! -f "$INSTALL_DIR/config/hyprland/hyprwhspr-tray.sh" ]; then
    log_error "Tray script not found: $INSTALL_DIR/config/hyprland/hyprwhspr-tray.sh"
    return 1
  fi
  
  mkdir -p "$USER_HOME/.config/hypr/scripts"
  cp "$INSTALL_DIR/config/hyprland/hyprwhspr-tray.sh" "$USER_HOME/.config/hypr/scripts/"
  chmod +x "$USER_HOME/.config/hypr/scripts/hyprwhspr-tray.sh"
  
  # NO sed replacement needed - file handles both modes dynamically
  log_success "Hyprland integration configured"
}

# ----------------------- Waybar integration --------------------
setup_waybar_integration() {
  log_info "Waybar integration…"
  
  # Validate required files exist
  if [ ! -f "$INSTALL_DIR/config/hyprland/hyprwhspr-tray.sh" ]; then
    log_error "Tray script not found: $INSTALL_DIR/config/hyprland/hyprwhspr-tray.sh"
    return 1
  fi
  
  if [ ! -f "$INSTALL_DIR/config/waybar/hyprwhspr-style.css" ]; then
    log_error "Waybar CSS not found: $INSTALL_DIR/config/waybar/hyprwhspr-style.css"
    return 1
  fi

  # Check if waybar is currently running and warn user
  if pgrep -x waybar >/dev/null 2>&1; then
    log_warning "Waybar is currently running - it may disappear if config changes cause errors"
    log_info "If waybar disappears, run 'waybar' from CLI to see error details"
  fi

  local waybar_config="$USER_HOME/.config/waybar/config.jsonc"
  if [ ! -f "$waybar_config" ]; then
    log_warning "Waybar config not found ($waybar_config)"
    log_info "Creating basic Waybar config with hyprwhspr integration..."
    
    # Create basic waybar config
    mkdir -p "$USER_HOME/.config/waybar"
    cat > "$waybar_config" <<'WAYBAR_CONFIG'
{
  "layer": "top",
  "position": "top",
  "height": 30,
  "modules-left": ["hyprland/workspaces"],
  "modules-center": ["hyprland/window"],
  "modules-right": ["custom/hyprwhspr", "clock", "tray"],
  "include": ["hyprwhspr-module.jsonc"]
}
WAYBAR_CONFIG
    log_success "Created basic Waybar config"
  fi

  mkdir -p "$USER_HOME/.config/waybar"
  cat > "$USER_HOME/.config/waybar/hyprwhspr-module.jsonc" <<EOF
{
  "custom/hyprwhspr": {
    "format": "{}",
    "exec": "$INSTALL_DIR/config/hyprland/hyprwhspr-tray.sh status",
    "interval": 1,
    "return-type": "json",
    "exec-on-event": true,
    "on-click": "$INSTALL_DIR/config/hyprland/hyprwhspr-tray.sh toggle",
    "on-click-right": "$INSTALL_DIR/config/hyprland/hyprwhspr-tray.sh start",
    "on-click-middle": "$INSTALL_DIR/config/hyprland/hyprwhspr-tray.sh restart",
    "tooltip": true
  }
}
EOF

  # Validate the generated JSON
  if ! python3 -m json.tool "$USER_HOME/.config/waybar/hyprwhspr-module.jsonc" >/dev/null 2>&1; then
    log_error "Generated waybar module JSON is invalid"
    return 1
  fi

  # Create backup of waybar config before modifying
  local backup_file="$waybar_config.backup-$(date +%Y%m%d-%H%M%S)"
  if cp "$waybar_config" "$backup_file"; then
    log_info "Backup created: $backup_file"
  else
    log_warning "Could not create backup of waybar config"
  fi

  # Use Python to safely modify the waybar config with proper JSON handling
  "$VENV_DIR/bin/python3" <<EOF
import sys
import json

config_path = "$waybar_config"
module_path = "$USER_HOME/.config/waybar/hyprwhspr-module.jsonc"

try:
    # Read existing config with standard JSON parser
    with open(config_path, 'r') as f:
        config = json.load(f)
    
    # Add include if not present
    if 'include' not in config:
        config['include'] = []
    
    if 'hyprwhspr-module.jsonc' not in config['include']:
        config['include'].append('hyprwhspr-module.jsonc')
        print("Added hyprwhspr-module.jsonc to include list")
    
    # Add module to modules-right if not present
    if 'modules-right' not in config:
        config['modules-right'] = []
    
    if 'custom/hyprwhspr' not in config['modules-right']:
        config['modules-right'].insert(0, 'custom/hyprwhspr')
        print("Added custom/hyprwhspr to modules-right array")
    
    # Write back the config with proper JSON formatting
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2, separators=(',', ': '))

    print("Waybar config updated successfully")
    
except json.JSONDecodeError as e:
    print(f"ERROR: Invalid JSON in waybar config: {e}", file=sys.stderr)
    print("Please check your waybar config for syntax errors", file=sys.stderr)
    sys.exit(1)
except Exception as e:
    print(f"Error updating waybar config: {e}", file=sys.stderr)
    sys.exit(1)
EOF

  if [ $? -ne 0 ]; then
    log_error "Failed to update waybar config"
    return 1
  fi

  # Test the updated config with waybar to ensure it can be parsed
  log_info "Testing waybar config..."
  if timeout 2s waybar --config "$waybar_config" --log-level error 2>&1 | head -5 | grep -i "error parsing" >/dev/null 2>&1; then
    log_error "Waybar cannot parse the updated config"
    log_info "Attempting to restore backup..."
    if [ -f "$backup_file" ]; then
      cp "$backup_file" "$waybar_config"
      log_warning "Config restored from backup. Please check your waybar config manually."
    fi
    return 1
  else
    log_success "✓ Waybar config validated successfully"
  fi

  # Kill the test waybar instance
  pkill -f "waybar --config $waybar_config" 2>/dev/null || true

  if [ -f "$INSTALL_DIR/config/waybar/hyprwhspr-style.css" ]; then
    log_info "Copying waybar CSS file to user config..."
    if cp "$INSTALL_DIR/config/waybar/hyprwhspr-style.css" "$USER_HOME/.config/waybar/"; then
      log_success "✓ Waybar CSS file copied to user config"
    else
      log_error "✗ Failed to copy waybar CSS file to user config"
      return 1
    fi
    
    local waybar_style="$USER_HOME/.config/waybar/style.css"
    if [ -f "$waybar_style" ] && ! grep -q "hyprwhspr-style.css" "$waybar_style"; then
      log_info "Adding CSS import to waybar style.css..."
      if grep -q "^@import" "$waybar_style"; then
        awk '/^@import/ { print; last_import = NR } !/^@import/ { if (last_import && NR == last_import + 1) { print "@import \"hyprwhspr-style.css\";"; } print }' "$waybar_style" > "$waybar_style.tmp" && mv "$waybar_style.tmp" "$waybar_style"
      else
        echo -e "@import \"hyprwhspr-style.css\";\n$(cat "$waybar_style")" > "$waybar_style.tmp" && mv "$waybar_style.tmp" "$waybar_style"
      fi
      log_success "✓ CSS import added to waybar style.css"
    elif [ -f "$waybar_style" ]; then
      log_info "CSS import already present in waybar style.css"
    else
      log_warning "No waybar style.css found - user will need to add CSS import manually"
    fi
  else
    log_error "✗ Waybar CSS file not found at $INSTALL_DIR/config/waybar/hyprwhspr-style.css"
    return 1
  fi

  log_success "Waybar integration updated"
}

# ----------------------- User config ---------------------------
setup_user_config() {
  log_info "User config…"
  mkdir -p "$USER_CONFIG_DIR"
  if [ ! -f "$USER_CONFIG_DIR/config.json" ]; then
    cat > "$USER_CONFIG_DIR/config.json" <<'CFG'
{
  "primary_shortcut": "SUPER+ALT+D",
  "model": "base.en",
  "audio_feedback": true,
  "start_sound_volume": 0.5,
  "stop_sound_volume": 0.5,
  "start_sound_path": "ping-up.ogg",
  "stop_sound_path": "ping-down.ogg",
  "word_overrides": {}
}
CFG
    log_success "Created $USER_CONFIG_DIR/config.json"
  else
    sed -i 's|"model": "[^"]*"|"model": "base.en"|' "$USER_CONFIG_DIR/config.json"
    if ! grep -q "\"audio_feedback\"" "$USER_CONFIG_DIR/config.json"; then
      sed -i 's|"word_overrides": {}|"audio_feedback": true,\n    "start_sound_volume": 0.5,\n    "stop_sound_volume": 0.5,\n    "start_sound_path": "ping-up.ogg",\n    "stop_sound_path": "ping-down.ogg",\n    "word_overrides": {}|' "$USER_CONFIG_DIR/config.json"
    fi
    log_success "Updated existing config"
  fi
}

# ----------------------- Permissions & uinput ------------------
setup_permissions() {
  log_info "Permissions & uinput…"
  
  # Add user to required groups (including tty group that was missing)
  sudo usermod -a -G input,audio,tty "$ACTUAL_USER" || true

  # Ensure uinput module is loaded FIRST
  if [ ! -e "/dev/uinput" ]; then
    log_info "Loading uinput module..."
    sudo modprobe uinput || true
    sleep 2  # Give it more time to create the device
  fi
  
  # Verify device exists and is accessible
  if [ ! -e "/dev/uinput" ]; then
    log_warning "uinput device not created, skipping udev trigger"
    return 0
  fi

  if [ ! -f "/etc/udev/rules.d/99-uinput.rules" ]; then
    log_info "Creating /etc/udev/rules.d/99-uinput.rules"
    sudo tee /etc/udev/rules.d/99-uinput.rules > /dev/null <<'RULE'
# Allow members of the input group to access uinput device
KERNEL=="uinput", GROUP="input", MODE="0660"
RULE
    log_success "udev rule created"
  else
    log_info "udev rule for uinput already present"
  fi

  # Now reload udev rules and trigger uinput (after module is loaded)
  log_info "Reloading udev rules..."
  sudo udevadm control --reload-rules
  
  # Try to trigger uinput, but don't fail if it doesn't work
  if sudo udevadm trigger --name-match=uinput 2>/dev/null; then
    log_success "udev rules reloaded and uinput triggered"
  else
    log_warning "udev rules reloaded, but uinput trigger failed (device may not be ready)"
    log_info "This is normal on fresh systems - permissions will apply after reboot"
  fi

  log_warning "You may need to log out/in for new group memberships to apply"
}

# ----------------------- Audio devices ------------------------
setup_audio_devices() {
  log_info "Audio devices…"
  systemctl --user is-active --quiet pipewire || { systemctl --user start pipewire; systemctl --user start pipewire-pulse; }
  log_info "Available audio input devices:"
  pactl list short sources | grep input || log_warning "No audio input devices found"
}

# ----------------------- Validation ---------------------------
validate_installation() {
  log_info "Validating installation…"
  
  # Validate static files
  validate_install_dir || return 1
  
  # Validate runtime files
  [ -x "$VENV_DIR/bin/python" ] || { log_error "Venv missing ($VENV_DIR)"; return 1; }
  
  # Validate whisper-cli
  if have_system_whisper; then
    log_success "whisper-cli on PATH: $(command -v whisper-cli)"
  elif [ -x "$USER_WC_DIR/build/bin/whisper-cli" ]; then
    log_success "whisper-cli present at $USER_WC_DIR/build/bin/whisper-cli"
  else
    log_error "whisper-cli missing"
    return 1
  fi
  
  # Validate models
  if [ ! -f "$USER_MODELS_DIR/ggml-base.en.bin" ] && [ ! -f "$USER_MODELS_DIR/base.en.bin" ]; then
    log_error "Model missing (${USER_MODELS_DIR})"
    return 1
  fi
  
  log_success "Validation passed"
}

# ----------------------- Functional checks --------------------
verify_permissions_and_functionality() {
  log_info "Verifying permissions & functionality…"
  local all_ok=true

  if [ -e "/dev/uinput" ] && [ -r "/dev/uinput" ] && [ -w "/dev/uinput" ]; then
    log_success "✓ /dev/uinput accessible"
  else
    log_error "✗ /dev/uinput not accessible"; all_ok=false
  fi

  groups "$ACTUAL_USER" | grep -q "\binput\b"  && log_success "✓ user in 'input'"  || { log_error "✗ user NOT in 'input'"; all_ok=false; }
  groups "$ACTUAL_USER" | grep -q "\baudio\b"  && log_success "✓ user in 'audio'"  || { log_error "✗ user NOT in 'audio'"; all_ok=false; }

  command -v ydotool >/dev/null && timeout 5s ydotool help >/dev/null 2>&1 \
    && log_success "✓ ydotool responds" || { log_error "✗ ydotool problem"; all_ok=false; }

  command -v pactl >/dev/null && pactl list short sources | grep -q input \
    && log_success "✓ audio inputs present" || log_warning "⚠ no audio inputs detected"

  if have_system_whisper || [ -x "$USER_WC_DIR/build/bin/whisper-cli" ]; then
    timeout 10s whisper-cli --help >/dev/null 2>&1 \
      && log_success "✓ whisper-cli responds" || { log_error "✗ whisper-cli not responding"; all_ok=false; }
  fi

  if [ -x "$VENV_DIR/bin/python" ]; then
    timeout 5s "$VENV_DIR/bin/python" -c "import sounddevice" >/dev/null 2>&1 \
      && log_success "✓ Python audio libs present" || { log_error "✗ Python audio libs missing"; all_ok=false; }
  fi

  $all_ok && return 0 || return 1
}

# ----------------------- Smoke test ---------------------------
test_installation() {
  log_info "Testing installation…"
  
  # Test static files
  [ -f "$INSTALL_DIR/bin/hyprwhspr" ] || { log_error "Main executable missing"; return 1; }
  [ -f "$INSTALL_DIR/requirements.txt" ] || { log_error "Requirements file missing"; return 1; }
  
  # Test runtime files
  [ -d "$USER_BASE/venv" ] || { log_error "Python venv missing"; return 1; }
  
  # Test service start
  validate_installation || { log_error "Validation failed"; return 1; }

  if systemctl --user start "$SERVICE_NAME"; then
    log_success "Service started"
    systemctl --user stop "$SERVICE_NAME"
  else
    log_error "Failed to start service"
    return 1
  fi

  if "$USER_HOME/.config/hypr/scripts/hyprwhspr-tray.sh" status >/dev/null 2>&1; then
    log_success "Tray script working"
  else
    log_warning "Tray script not found or not executable (ok if Hyprland not configured)"
  fi

  log_success "Installation test passed"
}

# ----------------------- Main ---------------------------------
main() {
  if [ "$CHECK_MODE" = true ]; then
    log_info "Checking installation plan for $INSTALL_DIR"
    generate_installation_plan
    log_info "Check mode complete - no changes made"
    return 0
  fi
  
  log_info "Installing to $INSTALL_DIR"
  
  # Always copy files to system directory
  sudo mkdir -p "$INSTALL_DIR"
  sudo cp -r . "$INSTALL_DIR/"
  sudo chown -R "$ACTUAL_USER:$ACTUAL_USER" "$INSTALL_DIR"
  
  # Verify critical files were copied
  if [ -f "$INSTALL_DIR/config/waybar/hyprwhspr-style.css" ]; then
    log_success "✓ Waybar CSS file copied successfully"
  else
    log_error "✗ Waybar CSS file missing after copy operation"
    exit 1
  fi
  
  if [ -d "$INSTALL_DIR/share/assets" ]; then
    log_success "✓ Assets directory copied successfully"
  else
    log_error "✗ Assets directory missing after copy operation"
    exit 1
  fi
  
  # Validate that required files exist
  validate_install_dir || { log_error "Installation validation failed"; exit 1; }
  
  # Ensure user space exists for runtime data
  mkdir -p "$USER_BASE"
  log_info "User runtime data directory: $USER_BASE"

  install_system_dependencies
  setup_python_environment
  setup_nvidia_support
  setup_whisper
  download_models
  setup_systemd_service   # <— auto-enable & start (all modes)
  setup_hyprland_integration
  setup_user_config
  setup_permissions
  setup_audio_devices
  setup_waybar_integration
  validate_installation
  verify_permissions_and_functionality
  test_installation 

  # Final service restart to ensure everything is fresh
  log_info "Performing final service restart..."
  systemctl --user restart "$YDOTOOL_UNIT" 2>/dev/null || true
  systemctl --user restart "$SERVICE_NAME"
  log_success "Services restarted successfully"

  log_success "✓ hyprwhspr installation completed!"
  log_info ""
  log_info "Next steps:"
  log_info "  • Reboot your system to apply all changes"
  log_info "  • After reboot, hyprwhspr will be ready to use!"
  log_info ""
  log_info "Service status:"
  log_info "  systemctl --user status $YDOTOOL_UNIT $SERVICE_NAME"
  log_info ""
  log_info "View logs:"
  log_info "  journalctl --user -u $SERVICE_NAME"
  log_info ""
}

main "$@"
