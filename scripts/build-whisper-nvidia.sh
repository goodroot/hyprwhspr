#!/bin/bash

# hyprwhspr - Multi-GPU Whisper.cpp Build Script
# This script builds whisper.cpp with CUDA or ROCm/HIP support for GPU acceleration

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Logging functions
log_info() {
    echo -e "${BLUE}[INFO]${NC} $1"
}

log_success() {
    echo -e "${GREEN}[SUCCESS]${NC} $1"
}

log_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

log_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Configuration
INSTALL_DIR="/usr/lib/hyprwhspr"
WHISPER_DIR="${XDG_DATA_HOME:-$HOME/.local/share}/hyprwhspr/whisper.cpp"

# Check if running as root
if [[ $EUID -eq 0 ]]; then
    log_error "This script should not be run as root. It will use sudo when needed."
    exit 1
fi

# Check if we're in the right directory
if [ ! -d "$WHISPER_DIR" ]; then
    log_error "whisper.cpp directory not found at $WHISPER_DIR"
    log_error "Please run the main installation script first"
    exit 1
fi

# Function to detect GPU backend
detect_gpu_backend() {
    log_info "Detecting GPU backend..."
    
    # Check NVIDIA CUDA
    if command -v nvidia-smi &> /dev/null; then
        log_info "NVIDIA GPU detected:"
        nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader,nounits
        
        if command -v nvcc &> /dev/null; then
            log_success "CUDA toolkit found - using CUDA backend"
            echo "cuda"
            return
        else
            log_warning "nvcc (CUDA compiler) not found."
            log_warning "Installing CUDA toolkit..."
            sudo pacman -S --needed --noconfirm cuda
            echo "cuda"
            return
        fi
    fi
    
    # Check AMD ROCm
    if command -v rocm-smi &> /dev/null || [ -d "/opt/rocm" ]; then
        log_info "AMD ROCm detected"
        log_success "Using ROCm/HIP backend"
        echo "rocm"
        return
    fi
    
    # Check Vulkan
    if command -v vulkaninfo &> /dev/null; then
        log_info "Vulkan detected"
        log_success "Using Vulkan backend"
        echo "vulkan"
        return
    fi
    
    log_warning "No GPU backend detected - falling back to CPU"
    echo "cpu"
}

# Function to build with GPU support
build_with_gpu() {
    local backend=$1
    log_info "Building whisper.cpp with $backend support..."
    
    cd "$WHISPER_DIR"
    
    # Clean previous build
    rm -rf build
    
    # Create build directory
    mkdir -p build
    cd build
    
    # Configure with appropriate GPU support
    case $backend in
        "cuda")
            log_info "Configuring CMake with CUDA..."
            cmake .. -DWHISPER_CUDA=ON -DCMAKE_BUILD_TYPE=Release
            ;;
        "rocm")
            log_info "Configuring CMake with ROCm/HIP..."
            cmake .. -DGGML_HIP=ON -DCMAKE_BUILD_TYPE=Release
            ;;
        "vulkan")
            log_info "Configuring CMake with Vulkan..."
            cmake .. -DGGML_VULKAN=ON -DCMAKE_BUILD_TYPE=Release
            ;;
        "cpu")
            log_info "Configuring CMake for CPU-only..."
            cmake .. -DCMAKE_BUILD_TYPE=Release
            ;;
    esac
    
    # Build with optimizations
    log_info "Building with $backend support..."
    cmake --build . -j --config Release
    
    # Verify binary was created
    if [ ! -f "bin/whisper-cli" ]; then
        log_error "Failed to build whisper.cpp binary"
        exit 1
    fi
    
    log_success "whisper.cpp built with $backend support"
}

# Function to test GPU acceleration
test_gpu_acceleration() {
    log_info "Testing GPU acceleration..."
    
    cd "$WHISPER_DIR"
    
    # Check if CUDA is available in the binary
    if ldd build/bin/whisper-cli | grep -q cuda; then
        log_success "CUDA libraries linked successfully"
    else
        log_warning "CUDA libraries not linked - falling back to CPU"
    fi
    
    # Test with a small audio file if available
    if [ -f "samples/jfk.wav" ]; then
        log_info "Testing transcription with GPU acceleration..."
        timeout 30s ./build/bin/whisper-cli -m models/ggml-base.en.bin -f samples/jfk.wav --no-prints --language en || log_warning "Test timed out or failed"
    fi
    
    log_success "GPU acceleration test completed"
}

# Function to show usage information
show_usage() {
    log_info "Usage: $0 [OPTIONS]"
    echo ""
    echo "Options:"
    echo "  --help     Show this help message"
    echo "  --test     Run GPU acceleration test after build"
    echo ""
    echo "This script builds whisper.cpp with multi-GPU support for hyprwhspr."
    echo "Supported backends: NVIDIA CUDA, AMD ROCm/HIP, Vulkan, CPU-only"
    echo "Make sure you have:"
    echo "  - Appropriate GPU drivers installed"
    echo "  - Required toolkits (CUDA, ROCm, etc.)"
    echo "  - Run the main installation script first"
}

# Main execution
main() {
    local test_gpu=false
    
    # Parse arguments
    while [[ $# -gt 0 ]]; do
        case $1 in
            --help)
                show_usage
                exit 0
                ;;
            --test)
                test_gpu=true
                shift
                ;;
            *)
                log_error "Unknown option: $1"
                show_usage
                exit 1
                ;;
        esac
    done
    
    log_info "Starting multi-GPU whisper.cpp build for hyprwhspr..."
    
    # Detect GPU backend
    BACKEND=$(detect_gpu_backend)
    
    # Build with detected backend
    build_with_gpu "$BACKEND"
    
    # Test if requested
    if [ "$test_gpu" = true ]; then
        test_gpu_acceleration
    fi
    
    log_success "Multi-GPU build completed!"
    log_info ""
    log_info "Your hyprwhspr installation now has GPU acceleration!"
    log_info "The service will automatically use GPU acceleration when available."
}

# Run main function
main "$@"
