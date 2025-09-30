#!/bin/bash
# hyprwhspr Reset Script
# Cleans up hyprwhspr installation for fresh reinstall

set -euo pipefail

# ----------------------- Colors & logging -----------------------
RED='\033[0;31m'; GREEN='\033[0;32m'; YELLOW='\033[1;33m'; BLUE='\033[0;34m'; NC='\033[0m'
log_info()    { echo -e "${BLUE}[INFO]${NC} $*"; }
log_success() { echo -e "${GREEN}[SUCCESS]${NC} $*"; }
log_warning() { echo -e "${YELLOW}[WARNING]${NC} $*"; }
log_error()   { echo -e "${RED}[ERROR]${NC} $*"; }

# ----------------------- Configuration -------------------------
PACKAGE_NAME="hyprwhspr"
INSTALL_DIR="/opt/hyprwhspr"
SERVICE_NAME="hyprwhspr.service"
YDOTOOL_UNIT="ydotool.service"

# Detect actual user
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

# ----------------------- Helpers -------------------------------
confirm() {
  local prompt="$1"
  local default="${2:-n}"
  local response
  
  if [[ "$default" == "y" ]]; then
    prompt="$prompt [Y/n]: "
  else
    prompt="$prompt [y/N]: "
  fi
  
  read -p "$prompt" response
  response="${response:-$default}"
  [[ "$response" =~ ^[Yy]$ ]]
}

# ----------------------- Stop services ------------------------
stop_services() {
  log_info "Stopping hyprwhspr services..."
  
  # Stop and disable user services
  if systemctl --user is-active --quiet "$SERVICE_NAME" 2>/dev/null; then
    log_info "Stopping $SERVICE_NAME..."
    systemctl --user stop "$SERVICE_NAME" || true
  fi
  
  if systemctl --user is-active --quiet "$YDOTOOL_UNIT" 2>/dev/null; then
    log_info "Stopping $YDOTOOL_UNIT..."
    systemctl --user stop "$YDOTOOL_UNIT" || true
  fi
  
  # Disable services
  systemctl --user disable "$SERVICE_NAME" 2>/dev/null || true
  systemctl --user disable "$YDOTOOL_UNIT" 2>/dev/null || true
  
  log_success "Services stopped and disabled"
}

# ----------------------- Remove installation ------------------
remove_installation() {
  log_info "Removing installation files..."
  
  if [ -d "$INSTALL_DIR" ]; then
    log_info "Removing $INSTALL_DIR..."
    sudo rm -rf "$INSTALL_DIR"
    log_success "✓ Installation directory removed"
  else
    log_info "No installation directory found at $INSTALL_DIR"
  fi
}

# ----------------------- Remove user data ---------------------
remove_user_data() {
  log_info "Removing user data..."
  
  local user_data_dirs=(
    "$USER_HOME/.config/hyprwhspr"
    "$USER_HOME/.local/share/hyprwhspr"
    "$USER_HOME/.config/systemd/user/$SERVICE_NAME"
    "$USER_HOME/.config/systemd/user/$YDOTOOL_UNIT"
  )
  
  for dir in "${user_data_dirs[@]}"; do
    if [ -e "$dir" ]; then
      log_info "Removing $dir..."
      rm -rf "$dir"
      log_success "✓ Removed $dir"
    else
      log_info "No user data found at $dir"
    fi
  done
}

# ----------------------- Remove waybar integration ------------
remove_waybar_integration() {
  log_info "Cleaning up waybar integration..."
  
  local waybar_config="$USER_HOME/.config/waybar/config.jsonc"
  local waybar_style="$USER_HOME/.config/waybar/style.css"
  
  # Remove waybar module file
  if [ -f "$USER_HOME/.config/waybar/hyprwhspr-module.jsonc" ]; then
    log_info "Removing waybar module file..."
    rm -f "$USER_HOME/.config/waybar/hyprwhspr-module.jsonc"
    log_success "✓ Waybar module file removed"
  fi
  
  # Remove CSS file
  if [ -f "$USER_HOME/.config/waybar/hyprwhspr-style.css" ]; then
    log_info "Removing waybar CSS file..."
    rm -f "$USER_HOME/.config/waybar/hyprwhspr-style.css"
    log_success "✓ Waybar CSS file removed"
  fi
  
  # Remove CSS import from style.css
  if [ -f "$waybar_style" ] && grep -q "hyprwhspr-style.css" "$waybar_style"; then
    log_info "Removing CSS import from waybar style.css..."
    sed -i '/hyprwhspr-style.css/d' "$waybar_style"
    log_success "✓ CSS import removed from waybar style.css"
  fi
  
  # Remove module include from config
  if [ -f "$waybar_config" ] && grep -q "hyprwhspr-module.jsonc" "$waybar_config"; then
    log_info "Removing module include from waybar config..."
    sed -i '/hyprwhspr-module.jsonc/d' "$waybar_config"
    log_success "✓ Module include removed from waybar config"
  fi
}

# ----------------------- Remove hyprland integration ----------
remove_hyprland_integration() {
  log_info "Cleaning up hyprland integration..."
  
  if [ -f "$USER_HOME/.config/hypr/scripts/hyprwhspr-tray.sh" ]; then
    log_info "Removing hyprland tray script..."
    rm -f "$USER_HOME/.config/hypr/scripts/hyprwhspr-tray.sh"
    log_success "✓ Hyprland tray script removed"
  fi
}

# ----------------------- Remove symlinks ----------------------
remove_symlinks() {
  log_info "Removing symlinks..."
  
  local user_bin_dir="$USER_HOME/.local/bin"
  
  if [ -L "$user_bin_dir/whisper-cli" ]; then
    log_info "Removing whisper-cli symlink..."
    rm -f "$user_bin_dir/whisper-cli"
    log_success "✓ whisper-cli symlink removed"
  fi
}

# ----------------------- Clean systemd ------------------------
clean_systemd() {
  log_info "Reloading systemd daemon..."
  systemctl --user daemon-reload
  log_success "✓ Systemd daemon reloaded"
}

# ----------------------- Main ---------------------------------
main() {
  log_info "hyprwhspr reset"
  log_info "This will completely remove hyprwhspr and all its data."
  log_warning "This action cannot be undone!"
  
  echo
  log_info "The following will be removed:"
  echo "  • Installation directory: $INSTALL_DIR"
  echo "  • User config: $USER_HOME/.config/hyprwhspr"
  echo "  • User data: $USER_HOME/.local/share/hyprwhspr"
  echo "  • Systemd services: $SERVICE_NAME, $YDOTOOL_UNIT"
  echo "  • Waybar integration files and config"
  echo "  • Hyprland integration scripts"
  echo "  • Symlinks and binaries"
  echo
  
  if ! confirm "Do you want to continue?" "n"; then
    log_info "Reset cancelled by user"
    exit 0
  fi
  
  echo
  log_info "Starting reset process..."
  
  stop_services
  remove_installation
  remove_user_data
  remove_waybar_integration
  remove_hyprland_integration
  remove_symlinks
  clean_systemd
  
  echo
  log_success "hyprwhspr reset completed!"
  log_info "All hyprwhspr files and data have been removed."
  log_info "You can now run the installer again for a fresh installation."
  echo
  log_info "To reinstall:"
  log_info "  ./scripts/install-omarchy.sh"
}

main "$@"
