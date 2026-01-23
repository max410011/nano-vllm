#!/bin/bash
# =============================================================================
# Nano-vLLM Environment Setup Script
# =============================================================================
# This script sets up the development environment for nano-vllm using uv.
# It handles CUDA dependencies, flash-attn compilation, and lm-eval installation.
#
# Usage:
#   ./scripts/setup_env.sh [--cuda-version 12.4] [--python-version 3.11]
#
# Requirements:
#   - uv (https://github.com/astral-sh/uv)
#   - CUDA toolkit installed on system
# =============================================================================

set -e  # Exit on error

# Default values
CUDA_VERSION="${CUDA_VERSION:-12.4}"
PYTHON_VERSION="${PYTHON_VERSION:-3.11}"
VENV_DIR=".venv"

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

log_info() { echo -e "${GREEN}[INFO]${NC} $1"; }
log_warn() { echo -e "${YELLOW}[WARN]${NC} $1"; }
log_error() { echo -e "${RED}[ERROR]${NC} $1"; }

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --cuda-version) CUDA_VERSION="$2"; shift 2 ;;
        --python-version) PYTHON_VERSION="$2"; shift 2 ;;
        --help) echo "Usage: $0 [--cuda-version X.Y] [--python-version X.Y]"; exit 0 ;;
        *) log_error "Unknown option: $1"; exit 1 ;;
    esac
done

log_info "=== Nano-vLLM Environment Setup ==="
log_info "Python version: ${PYTHON_VERSION}"
log_info "CUDA version: ${CUDA_VERSION}"

# Check uv installation
if ! command -v uv &> /dev/null; then
    log_error "uv is not installed. Please install it first:"
    echo "  curl -LsSf https://astral.sh/uv/install.sh | sh"
    exit 1
fi
log_info "uv version: $(uv --version)"

# Check NVIDIA driver
if ! command -v nvidia-smi &> /dev/null; then
    log_error "nvidia-smi not found. Please ensure NVIDIA drivers are installed."
    exit 1
fi
log_info "GPU detected:"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader | head -1

# Create virtual environment
log_info "Creating virtual environment with Python ${PYTHON_VERSION}..."
uv venv "${VENV_DIR}" --python "${PYTHON_VERSION}"

# Activate virtual environment
source "${VENV_DIR}/bin/activate"
log_info "Virtual environment activated: ${VENV_DIR}"

# Install PyTorch with CUDA support
log_info "Installing PyTorch with CUDA ${CUDA_VERSION} support..."
CUDA_SHORT=$(echo $CUDA_VERSION | tr -d '.')
uv pip install torch torchvision torchaudio --index-url "https://download.pytorch.org/whl/cu${CUDA_SHORT:0:3}"

# Install triton
log_info "Installing triton..."
uv pip install triton

# Install flash-attn (this may take a while to compile)
log_info "Installing flash-attn (this may take several minutes)..."
uv pip install flash-attn --no-build-isolation

# Install other dependencies
log_info "Installing other dependencies..."
uv pip install transformers xxhash tqdm loguru pyyaml

# Install lm-evaluation-harness
log_info "Installing lm-evaluation-harness..."
uv pip install lm-eval

# Install nano-vllm in editable mode
log_info "Installing nano-vllm in editable mode..."
uv pip install -e .

# Install development dependencies
log_info "Installing development dependencies..."
uv pip install pytest pytest-asyncio ipython

# Verify installation
log_info "Verifying installation..."
python scripts/verify_env.py

log_info "=== Setup Complete ==="
log_info "To activate the environment, run:"
echo "  source ${VENV_DIR}/bin/activate"

