#!/usr/bin/env bash
#
# hyprwhspr Dependency Installation Script
#
# This script installs dependencies for hyprwhspr on Linux systems.
# Supports: Ubuntu, Debian, Fedora, openSUSE
#
# It also handles the ydotool version issue on Ubuntu/Debian where the
# apt repositories contain an outdated 0.1.x version that is incompatible.
#
# Usage:
#   curl -fsSL https://raw.githubusercontent.com/goodroot/hyprwhspr/main/scripts/install-deps.sh | bash
#   # or
#   ./scripts/install-deps.sh
#

set -euo pipefail

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Minimum required ydotool version
MIN_YDOTOOL_VERSION="1.0.0"

# Detected distro info
DISTRO=""
DISTRO_VERSION=""
PKG_MANAGER=""

print_header() {
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${BLUE}  hyprwhspr - Dependency Installation${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
}

log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[OK]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARN]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Detect Linux distribution
detect_distro() {
    if [[ ! -f /etc/os-release ]]; then
        log_error "Cannot detect distribution (/etc/os-release not found)"
        exit 1
    fi

    . /etc/os-release

    DISTRO_VERSION="${VERSION_ID:-unknown}"

    if [[ "$ID" == "ubuntu" ]]; then
        DISTRO="ubuntu"
        PKG_MANAGER="apt"
    elif [[ "$ID" == "debian" ]]; then
        DISTRO="debian"
        PKG_MANAGER="apt"
    elif [[ "$ID" == "fedora" ]]; then
        DISTRO="fedora"
        PKG_MANAGER="dnf"
    elif [[ "$ID" == "opensuse-tumbleweed" || "$ID" == "opensuse-leap" || "$ID" == "opensuse" ]]; then
        DISTRO="opensuse"
        PKG_MANAGER="zypper"
    elif [[ "${ID_LIKE:-}" == *"ubuntu"* || "${ID_LIKE:-}" == *"debian"* ]]; then
        DISTRO="debian-like"
        PKG_MANAGER="apt"
    elif [[ "${ID_LIKE:-}" == *"fedora"* ]]; then
        DISTRO="fedora-like"
        PKG_MANAGER="dnf"
    elif [[ "${ID_LIKE:-}" == *"suse"* ]]; then
        DISTRO="opensuse-like"
        PKG_MANAGER="zypper"
    else
        log_error "Unsupported distribution: $ID"
        log_info "Supported: Ubuntu, Debian, Fedora, openSUSE (and derivatives)"
        log_info "For Arch Linux, use: yay -S hyprwhspr"
        exit 1
    fi

    log_success "Detected: ${PRETTY_NAME:-$ID} (using $PKG_MANAGER)"
}

# Maximum Python version compatible with ML packages (onnxruntime, etc.)
MAX_PYTHON_MAJOR=3
MAX_PYTHON_MINOR=14

# Check Python version compatibility
check_python_version() {
    local python_path=""
    local python_version=""
    local major=""
    local minor=""

    # Find the default python3
    if command -v python3 &> /dev/null; then
        python_path="python3"
    elif command -v python &> /dev/null; then
        python_path="python"
    else
        log_warning "Python not found. It will be installed with dependencies."
        return 0
    fi

    # Get version string
    python_version=$($python_path --version 2>&1 | grep -oP 'Python \K[0-9]+\.[0-9]+' || echo "")
    if [[ -z "$python_version" ]]; then
        log_warning "Could not detect Python version."
        return 0
    fi

    # Parse major.minor
    major=$(echo "$python_version" | cut -d. -f1)
    minor=$(echo "$python_version" | cut -d. -f2)

    log_info "System Python version: $python_version"

    # Check if version is too new
    if [[ "$major" -gt "$MAX_PYTHON_MAJOR" ]] || \
       [[ "$major" -eq "$MAX_PYTHON_MAJOR" && "$minor" -gt "$MAX_PYTHON_MINOR" ]]; then
        echo ""
        log_warning "Python $python_version is too new for ML packages (onnxruntime, etc.)"
        echo ""
        echo -e "${YELLOW}hyprwhspr's local transcription requires Python ${MAX_PYTHON_MAJOR}.${MAX_PYTHON_MINOR} or earlier.${NC}"
        echo -e "${YELLOW}ML packages like onnxruntime don't have wheels for Python $python_version yet.${NC}"
        echo ""
        echo "Options:"
        echo ""
        echo "  1) Install Python ${MAX_PYTHON_MAJOR}.${MAX_PYTHON_MINOR} or ${MAX_PYTHON_MAJOR}.$((MAX_PYTHON_MINOR-1)) alongside your current Python:"
        case "$PKG_MANAGER" in
            dnf)
                echo "     sudo dnf install python3.14  # or python3.13"
                ;;
            apt)
                echo "     sudo apt install python3.13  # or use deadsnakes PPA for newer"
                ;;
            zypper)
                echo "     sudo zypper install python314  # or python313"
                ;;
        esac
        echo ""
        echo "  2) Use cloud transcription (no local Python requirement):"
        echo "     During 'hyprwhspr setup', select 'REST API' or 'Realtime WS'"
        echo ""
        echo "  3) Continue anyway (installation may fail for local backends)"
        echo ""
        read -p "Continue with dependency installation? [y/N] " -n 1 -r
        echo ""
        if [[ ! $REPLY =~ ^[Yy]$ ]]; then
            log_info "Installation cancelled."
            log_info "Install Python ${MAX_PYTHON_MAJOR}.${MAX_PYTHON_MINOR} or earlier, then re-run this script."
            exit 0
        fi
        echo ""
        log_warning "Continuing with Python $python_version - local backends may not work."
        log_info "You can specify an alternative Python during setup: hyprwhspr setup --python /path/to/python3.14"
        echo ""
    else
        log_success "Python version $python_version is compatible"
    fi
}

# Compare version strings (returns 0 if $1 >= $2)
version_gte() {
    printf '%s\n%s\n' "$2" "$1" | sort -V -C
}

# Get ydotool version
get_ydotool_version() {
    if ! command -v ydotool &> /dev/null; then
        echo ""
        return 0
    fi

    local version=""

    # Method 1: Try apt show (works reliably on Debian/Ubuntu)
    if command -v apt &> /dev/null; then
        version=$(apt show ydotool 2>/dev/null | awk -F': ' '/^Version:/ {
            v = $2
            sub(/^[0-9]+:/, "", v)  # remove epoch
            sub(/-.*/, "", v)        # remove debian revision
            print v
            exit
        }')
        if [[ -n "$version" ]]; then
            echo "$version"
            return 0
        fi
    fi

    # Method 2: Try ydotoold --version with timeout
    # (ydotool has no --version flag, only ydotoold does)
    # Note: Old ydotoold 0.1.x has no --version and may hang, hence the timeout
    if command -v ydotoold &> /dev/null; then
        local version_output
        version_output=$(timeout 3s ydotoold --version 2>&1 || echo "")

        # Handle "UNKNOWN" response from some builds
        if [[ "$version_output" != "UNKNOWN" ]] && [[ "$version_output" =~ ([0-9]+\.[0-9]+\.?[0-9]*) ]]; then
            echo "${BASH_REMATCH[1]}"
            return 0
        fi
    fi

    # Fallback: ydotool exists but we can't determine version - assume old
    echo "0.1.0"
}

# Check if ydotool version is sufficient
check_ydotool_version() {
    local version
    version=$(get_ydotool_version)

    if [[ -z "$version" ]]; then
        log_warning "ydotool not installed"
        return 1
    fi

    if version_gte "$version" "$MIN_YDOTOOL_VERSION"; then
        log_success "ydotool version $version (>= $MIN_YDOTOOL_VERSION required)"
        return 0
    else
        log_warning "ydotool version $version is too old (>= $MIN_YDOTOOL_VERSION required)"
        return 1
    fi
}

# Install ydotool from Debian backports (for Ubuntu/Debian)
install_ydotool_backports() {
    log_info "Installing ydotool from Debian trixie-backports..."

    local tmp_dir
    tmp_dir=$(mktemp -d)
    local deb_url="http://deb.debian.org/debian/pool/main/y/ydotool/ydotool_1.0.4-2~bpo13+1_amd64.deb"
    local deb_file="$tmp_dir/ydotool.deb"

    log_info "Downloading ydotool 1.0.4..."
    if ! wget -q --show-progress -O "$deb_file" "$deb_url"; then
        log_error "Failed to download ydotool package"
        rm -rf "$tmp_dir"
        return 1
    fi

    # Remove old ydotool and ydotoold if installed
    # (old versions had ydotool and ydotoold as separate packages)
    if dpkg -l ydotool &> /dev/null 2>&1; then
        log_info "Removing old ydotool version..."
        sudo apt remove -y ydotool ydotoold 2>/dev/null || sudo apt remove -y ydotool || true
    fi

    # Install the new package
    log_info "Installing ydotool 1.0.4..."
    if ! sudo dpkg -i "$deb_file"; then
        log_info "Fixing dependencies..."
        sudo apt install -f -y
    fi

    rm -rf "$tmp_dir"

    if check_ydotool_version; then
        log_success "ydotool installed successfully"
        return 0
    else
        log_error "ydotool installation failed"
        return 1
    fi
}

# Install dependencies for Debian/Ubuntu
install_deps_apt() {
    log_info "Installing system dependencies via apt..."

    sudo apt update

    # Core dependencies (excluding ydotool - handled separately on Debian/Ubuntu)
    sudo apt install -y \
        python3 \
        python3-pip \
        python3-venv \
        git \
        cmake \
        make \
        build-essential \
        python3-dev \
        libportaudio2 \
        python3-numpy \
        python3-scipy \
        python3-evdev \
        python3-requests \
        python3-psutil \
        python3-rich \
        python3-pulsectl \
        python3-pyudev \
        python3-dbus \
        python3-gi \
        gir1.2-gtk-4.0 \
        pipewire \
        pipewire-pulse \
        pulseaudio-utils \
        wl-clipboard \
        wget

    # Optional Python packages (not present on all Debian/Ubuntu releases)
    local optional_packages=(
        python3-sounddevice
        python3-pyperclip
        python3-websocket
    )
    local available_packages=()
    local pkg
    for pkg in "${optional_packages[@]}"; do
        if apt-cache show "$pkg" &> /dev/null; then
            available_packages+=("$pkg")
        else
            log_warning "Package $pkg not found in apt repos - will use pip if needed"
        fi
    done
    if [[ ${#available_packages[@]} -gt 0 ]]; then
        sudo apt install -y "${available_packages[@]}"
    fi

    # gtk4-layer-shell (may not be available on older Ubuntu)
    if apt-cache show gir1.2-gtk4layershell-1.0 &> /dev/null 2>&1; then
        sudo apt install -y gir1.2-gtk4layershell-1.0
        log_success "gtk4-layer-shell installed (mic-osd visualizer available)"
    else
        log_warning "gir1.2-gtk4layershell-1.0 not available - mic-osd visualizer will be disabled"
    fi

    log_success "System dependencies installed"
}

# Install dependencies for Fedora
install_deps_dnf() {
    log_info "Installing system dependencies via dnf..."

    sudo dnf install -y \
        python3 \
        python3-pip \
        python3-devel \
        git \
        cmake \
        make \
        gcc-c++ \
        python3-sounddevice \
        python3-numpy \
        python3-scipy \
        python3-evdev \
        python3-pyperclip \
        python3-requests \
        python3-psutil \
        python3-rich \
        python3-pulsectl \
        python3-pyudev \
        python3-dbus \
        python3-gobject \
        gtk4 \
        gtk4-layer-shell \
        pipewire \
        pipewire-pulseaudio \
        ydotool \
        wl-clipboard

    log_success "System dependencies installed"
}

# Install dependencies for openSUSE
install_deps_zypper() {
    log_info "Installing system dependencies via zypper..."

    sudo zypper install -y \
        python3 \
        python3-pip \
        python3-devel \
        git \
        cmake \
        make \
        gcc-c++ \
        python3-sounddevice \
        python3-numpy \
        python3-scipy \
        python3-evdev \
        python3-pyperclip \
        python3-requests \
        python3-psutil \
        python3-rich \
        python3-pulsectl \
        python3-pyudev \
        python3-gobject \
        typelib-1_0-Gtk-4_0 \
        pipewire \
        pipewire-pulseaudio \
        ydotool \
        wl-clipboard

    # Optional dbus package naming differs across openSUSE releases
    if sudo zypper info -t package python3-dbus-python &> /dev/null; then
        sudo zypper install -y python3-dbus-python
    elif sudo zypper info -t package python3-dbus &> /dev/null; then
        sudo zypper install -y python3-dbus
    else
        log_warning "python3-dbus package not found - dbus integration may be unavailable"
    fi

    # gtk4-layer-shell (Tumbleweed only, from community repo)
    log_info "Attempting to install gtk4-layer-shell..."
    if sudo zypper install -y gtk4-layer-shell 2>/dev/null; then
        log_success "gtk4-layer-shell installed (mic-osd visualizer available)"
    else
        log_warning "gtk4-layer-shell not available - mic-osd visualizer will be disabled"
        log_info "For Tumbleweed, you can try adding the community repo:"
        log_info "  sudo zypper addrepo https://download.opensuse.org/repositories/devel:languages:zig/openSUSE_Tumbleweed/devel:languages:zig.repo"
        log_info "  sudo zypper refresh && sudo zypper install gtk4-layer-shell"
    fi

    log_success "System dependencies installed"
}

# Install Python packages that aren't in distro repos
install_pip_packages() {
    log_info "Checking Python packages..."

    local need_sounddevice=false
    local need_pyperclip=false
    local need_pulsectl=false
    local need_pyudev=false
    local need_websocket=false

    # Check if packages are already available (Fedora/openSUSE include them)
    if ! python3 -c "import sounddevice" 2>/dev/null; then
        need_sounddevice=true
    fi

    if ! python3 -c "import pyperclip" 2>/dev/null; then
        need_pyperclip=true
    fi
    
    if ! python3 -c "import pulsectl" 2>/dev/null; then
        need_pulsectl=true
    fi

    if ! python3 -c "import pyudev" 2>/dev/null; then
        need_pyudev=true
    fi

    if ! python3 -c "import websocket" 2>/dev/null; then
        need_websocket=true
    fi

    if $need_sounddevice || $need_pyperclip || $need_pulsectl || $need_pyudev || $need_websocket; then
        log_info "Installing Python packages via pip..."

        local packages=""
        $need_sounddevice && packages="$packages sounddevice"
        $need_pyperclip && packages="$packages pyperclip"
        $need_pulsectl && packages="$packages pulsectl"
        $need_pyudev && packages="$packages pyudev"
        $need_websocket && packages="$packages websocket-client"

        # Try with --break-system-packages first (needed on newer systems)
        # Fall back to without it for older systems
        python3 -m pip install --user --break-system-packages $packages 2>/dev/null || \
        python3 -m pip install --user $packages

        log_success "Python packages installed"
    else
        log_success "Python packages already available"
    fi

    if ! python3 -c "import dbus" 2>/dev/null; then
        log_warning "python3-dbus is missing; install via your package manager (e.g., python3-dbus)"
    fi
}

# Handle ydotool for Debian/Ubuntu (special case due to outdated apt version)
handle_ydotool_debian() {
    echo ""
    echo -e "${BLUE}ydotool Version Check${NC}"
    echo "------------------------------------------------------------"

    # Check if ydotool is installed and what version
    if ! command -v ydotool &> /dev/null; then
        log_warning "ydotool is not installed."
        echo ""
        echo "Ubuntu/Debian apt repositories contain an outdated ydotool (0.1.x)"
        echo "that is incompatible with hyprwhspr. hyprwhspr requires ydotool 1.0+."
        echo ""
        echo "Options:"
        echo "  1) Install ydotool 1.0.4 from Debian backports (recommended)"
        echo "  2) Skip - I'll install ydotool manually later"
        echo ""
        read -p "Choice [1/2]: " -n 1 -r
        echo ""

        if [[ $REPLY == "1" ]]; then
            install_ydotool_backports
        else
            log_warning "Skipping ydotool installation."
            log_warning "You must install ydotool 1.0+ manually before using hyprwhspr."
            echo ""
            echo "Manual installation options:"
            echo "  - Debian backports: wget http://deb.debian.org/debian/pool/main/y/ydotool/ydotool_1.0.4-2~bpo13+1_amd64.deb && sudo dpkg -i ydotool_1.0.4-2~bpo13+1_amd64.deb"
            echo "  - Build from source: https://github.com/ReimuNotMoe/ydotool"
        fi
    elif ! check_ydotool_version; then
        echo ""
        echo "Your installed ydotool version is too old."
        echo ""
        echo "Ubuntu/Debian apt contains ydotool 0.1.x which uses incompatible syntax."
        echo "hyprwhspr requires ydotool 1.0+ for paste injection to work correctly."
        echo "With the old version, paste will output garbage like '244442' instead of text."
        echo ""
        echo "Options:"
        echo "  1) Replace with ydotool 1.0.4 from Debian backports (recommended)"
        echo "  2) Skip - I'll handle this manually later"
        echo ""
        read -p "Choice [1/2]: " -n 1 -r
        echo ""

        if [[ $REPLY == "1" ]]; then
            install_ydotool_backports
        else
            log_warning "Keeping old ydotool version."
            log_warning "hyprwhspr paste injection WILL NOT WORK until you upgrade to ydotool 1.0+"
        fi
    fi
}

# Handle ydotool for Fedora/openSUSE (usually fine from repos)
handle_ydotool_other() {
    echo ""
    echo -e "${BLUE}ydotool Version Check${NC}"
    echo "------------------------------------------------------------"

    if ! command -v ydotool &> /dev/null; then
        log_error "ydotool was not installed. Please install it manually."
        return 1
    fi

    check_ydotool_version || {
        log_warning "ydotool version may be incompatible."
        log_warning "If paste injection doesn't work, you may need to build ydotool from source."
        log_info "See: https://github.com/ReimuNotMoe/ydotool"
    }
}

# Main installation flow
main() {
    print_header

    # Detect distribution
    detect_distro

    # Check Python version compatibility early
    check_python_version

    echo ""
    log_info "This script will install dependencies for hyprwhspr."
    echo ""
    read -p "Continue? [Y/n] " -n 1 -r
    echo ""

    if [[ $REPLY =~ ^[Nn]$ ]]; then
        log_info "Installation cancelled."
        exit 0
    fi

    # Install system dependencies based on distro
    echo ""
    echo -e "${BLUE}Step 1: System Dependencies${NC}"
    echo "------------------------------------------------------------"

    case "$PKG_MANAGER" in
        apt)
            install_deps_apt
            ;;
        dnf)
            install_deps_dnf
            ;;
        zypper)
            install_deps_zypper
            ;;
    esac

    # Handle ydotool (special handling for Debian/Ubuntu)
    echo ""
    echo -e "${BLUE}Step 2: ydotool${NC}"
    echo "------------------------------------------------------------"

    case "$PKG_MANAGER" in
        apt)
            handle_ydotool_debian
            ;;
        *)
            handle_ydotool_other
            ;;
    esac

    # Install Python packages if needed
    echo ""
    echo -e "${BLUE}Step 3: Python Packages${NC}"
    echo "------------------------------------------------------------"
    install_pip_packages

    # Done!
    echo ""
    echo -e "${BLUE}============================================================${NC}"
    echo -e "${GREEN}  Dependencies Installed!${NC}"
    echo -e "${BLUE}============================================================${NC}"
    echo ""
    echo "Next steps:"
    echo ""
    echo "  1. Clone hyprwhspr (if you haven't already):"
    echo "     git clone https://github.com/goodroot/hyprwhspr.git ~/hyprwhspr"
    echo ""
    echo "  2. Run the setup wizard:"
    echo "     cd ~/hyprwhspr"
    echo "     ./bin/hyprwhspr setup"
    echo ""
    echo "  3. Log out and back in (for group permissions)"
    echo ""
    echo "  4. Press Super+Alt+D to start dictating!"
    echo ""
}

main "$@"
