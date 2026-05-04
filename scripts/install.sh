#!/usr/bin/env bash
# =============================================================================
#  AXON — Install Script  (macOS + Linux)
#  Detects GPU (NVIDIA CUDA / Apple Silicon MPS) and installs accordingly.
#  CPU-only fallback when no compatible GPU is found.
#  Run once from the project root:  bash scripts/install.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

VENV="$PROJECT_ROOT/.venv"
PYTHON=""

# ── Colours ──────────────────────────────────────────────────────────────────
RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'
CYN='\033[0;36m'; GRY='\033[0;90m'; RST='\033[0m'

step()  { echo -e "${CYN}  [$1]  $2${RST}"; }
ok()    { echo -e "${GRN}  [OK]  $1${RST}"; }
warn()  { echo -e "${YEL}  [WARN] $1${RST}"; }
skip()  { echo -e "${GRY}  [SKIP] $1${RST}"; }
fatal() { echo -e "${RED}  [FATAL] $1${RST}"; exit 1; }

echo ""
echo -e "${CYN}  =============================================${RST}"
echo -e "${CYN}       A X O N  --  Emerging Intelligence     ${RST}"
echo -e "${CYN}  =============================================${RST}"
echo ""

# =============================================================================
# [0] Python 3.12
# =============================================================================
step "0/10" "Locating Python 3.12..."
for candidate in python3.12 python3 python; do
    if command -v "$candidate" &>/dev/null; then
        ver=$("$candidate" --version 2>&1 | grep -oE '[0-9]+\.[0-9]+')
        if [[ "$ver" == "3.12" ]]; then
            PYTHON="$candidate"
            ok "Python 3.12 found: $(command -v $candidate)"
            break
        fi
    fi
done
if [[ -z "$PYTHON" ]]; then
    fatal "Python 3.12 not found. Install from https://python.org or via your package manager."
fi

# =============================================================================
# [1/10] Virtual environment
# =============================================================================
step "1/10" "Setting up virtual environment..."
if [[ ! -f "$VENV/bin/python" ]]; then
    "$PYTHON" -m venv "$VENV"
    ok "venv created at .venv"
else
    skip "venv already exists"
fi
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

# =============================================================================
# [2/10] pip + setuptools
# =============================================================================
step "2/10" "Upgrading pip + setuptools..."
"$PY" -m pip install --upgrade pip wheel --quiet
"$PIP" install "setuptools<82" --quiet

# =============================================================================
# [3/10] GPU detection
# =============================================================================
step "3/10" "Detecting GPU..."

GPU_TYPE="cpu"   # cpu | cuda | mps

# Check for NVIDIA GPU (Linux/Mac)
if command -v nvidia-smi &>/dev/null; then
    GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -1 || echo "")
    if [[ -n "$GPU_NAME" ]]; then
        GPU_TYPE="cuda"
        ok "NVIDIA GPU detected: $GPU_NAME"
    fi
fi

# Check for Apple Silicon (macOS MPS)
if [[ "$GPU_TYPE" == "cpu" ]] && [[ "$(uname)" == "Darwin" ]]; then
    ARCH=$(uname -m)
    if [[ "$ARCH" == "arm64" ]]; then
        GPU_TYPE="mps"
        ok "Apple Silicon detected ($ARCH) — MPS backend will be used"
    fi
fi

if [[ "$GPU_TYPE" == "cpu" ]]; then
    warn "No compatible GPU found — installing CPU-only PyTorch"
    warn "Performance will be lower. For GPU: install NVIDIA drivers + CUDA 12.x"
fi

# =============================================================================
# [4/10] PyTorch
# =============================================================================
step "4/10" "Installing PyTorch ($GPU_TYPE)..."

TORCH_OK=$("$PY" -c "import torch; print('ok')" 2>/dev/null || echo "")
if [[ "$TORCH_OK" == "ok" ]]; then
    # Verify it matches the desired backend
    if [[ "$GPU_TYPE" == "cuda" ]]; then
        CUDA_OK=$("$PY" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "")
        if [[ "$CUDA_OK" == "True" ]]; then
            TV=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null)
            GPU=$("$PY" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            skip "torch $TV | GPU: $GPU"
        else
            warn "torch installed but CUDA not available — reinstalling with CUDA support..."
            "$PIP" uninstall torch torchvision torchaudio -y --quiet 2>/dev/null || true
            TORCH_OK=""
        fi
    elif [[ "$GPU_TYPE" == "mps" ]]; then
        MPS_OK=$("$PY" -c "import torch; print(torch.backends.mps.is_available())" 2>/dev/null || echo "")
        if [[ "$MPS_OK" == "True" ]]; then
            TV=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null)
            skip "torch $TV | MPS available"
        else
            "$PIP" uninstall torch torchvision torchaudio -y --quiet 2>/dev/null || true
            TORCH_OK=""
        fi
    else
        TV=$("$PY" -c "import torch; print(torch.__version__)" 2>/dev/null)
        skip "torch $TV | CPU mode"
    fi
fi

if [[ "$TORCH_OK" != "ok" ]]; then
    if [[ "$GPU_TYPE" == "cuda" ]]; then
        # Detect CUDA version from nvidia-smi
        CUDA_VER=$(nvidia-smi 2>/dev/null | grep -oP 'CUDA Version: \K[0-9]+\.[0-9]+' | head -1 || echo "12.8")
        CUDA_MAJOR=$(echo "$CUDA_VER" | cut -d. -f1)
        CUDA_MINOR=$(echo "$CUDA_VER" | cut -d. -f2)
        echo -e "${GRY}  CUDA $CUDA_VER detected${RST}"

        # Map to PyTorch wheel index
        if (( CUDA_MAJOR >= 12 && CUDA_MINOR >= 8 )); then
            TORCH_INDEX="https://download.pytorch.org/whl/nightly/cu128"
            TORCH_ARGS="--pre"
        elif (( CUDA_MAJOR >= 12 && CUDA_MINOR >= 4 )); then
            TORCH_INDEX="https://download.pytorch.org/whl/cu124"
            TORCH_ARGS=""
        elif (( CUDA_MAJOR >= 12 && CUDA_MINOR >= 1 )); then
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            TORCH_ARGS=""
        else
            warn "CUDA $CUDA_VER may not have an exact wheel — using cu121 as closest match"
            TORCH_INDEX="https://download.pytorch.org/whl/cu121"
            TORCH_ARGS=""
        fi

        "$PIP" install $TORCH_ARGS torch torchvision torchaudio \
            --index-url "$TORCH_INDEX"

        CUDA_CHECK=$("$PY" -c "import torch; print(torch.cuda.is_available())" 2>/dev/null || echo "")
        if [[ "$CUDA_CHECK" == "True" ]]; then
            GPU=$("$PY" -c "import torch; print(torch.cuda.get_device_name(0))" 2>/dev/null)
            ok "CUDA ready: $GPU"
        else
            warn "CUDA still not available. Check NVIDIA driver >= 525 + CUDA $CUDA_VER."
            warn "Falling back to CPU-only torch for this session."
            "$PIP" uninstall torch torchvision torchaudio -y --quiet 2>/dev/null || true
            "$PIP" install torch torchvision torchaudio \
                --index-url https://download.pytorch.org/whl/cpu --quiet
        fi
    elif [[ "$GPU_TYPE" == "mps" ]]; then
        # macOS MPS — standard pip torch works
        "$PIP" install torch torchvision torchaudio --quiet
        ok "PyTorch with MPS support installed"
    else
        # CPU only
        "$PIP" install torch torchvision torchaudio \
            --index-url https://download.pytorch.org/whl/cpu --quiet
        ok "PyTorch (CPU-only) installed"
    fi
    # Re-pin numpy after torch (torchvision may downgrade it)
    "$PIP" install "numpy>=2.0" --upgrade --quiet
fi

# =============================================================================
# [5/10] Core deps
# =============================================================================
step "5/10" "Core dependencies..."
"$PIP" install --quiet \
    flask flask-socketio flask-cors eventlet \
    anthropic openai-whisper \
    sounddevice \
    opencv-python \
    edge-tts pygame \
    scipy requests

# macOS: pyaudio needs portaudio from brew
if [[ "$(uname)" == "Darwin" ]]; then
    if command -v brew &>/dev/null; then
        brew list portaudio &>/dev/null 2>&1 || brew install portaudio --quiet
        "$PIP" install pyaudio --quiet || warn "pyaudio install failed (non-fatal)"
    else
        warn "Homebrew not found — pyaudio skipped (voice input may be limited)"
    fi
else
    # Linux: needs portaudio dev headers
    if command -v apt-get &>/dev/null; then
        sudo apt-get install -y portaudio19-dev python3-dev &>/dev/null 2>&1 || true
    elif command -v dnf &>/dev/null; then
        sudo dnf install -y portaudio-devel &>/dev/null 2>&1 || true
    elif command -v pacman &>/dev/null; then
        sudo pacman -S --noconfirm portaudio &>/dev/null 2>&1 || true
    fi
    "$PIP" install pyaudio --quiet || warn "pyaudio install failed (non-fatal)"
fi

# numpy check
NPV=$("$PY" -c "import numpy; print(numpy.__version__)" 2>/dev/null || echo "")
if [[ ! "$NPV" =~ ^2\. ]]; then
    echo -e "${CYN}  Upgrading numpy to 2.x...${RST}"
    "$PIP" install "numpy>=2.0" --quiet
else
    skip "numpy $NPV"
fi

# =============================================================================
# [6/10] Vision: YOLOv8 + emotion backends
# =============================================================================
step "6/10" "Vision (YOLOv8 + emotion backends)..."

ULTRA_OK=$("$PY" -c "import ultralytics; print('ok')" 2>/dev/null || echo "")
if [[ "$ULTRA_OK" != "ok" ]]; then
    "$PIP" install ultralytics
else skip "ultralytics ok"; fi

TFKERAS_OK=$("$PY" -c "import tf_keras; print('ok')" 2>/dev/null || echo "")
if [[ "$TFKERAS_OK" != "ok" ]]; then
    "$PIP" install tf-keras --quiet
else skip "tf-keras ok"; fi

FER_OK=$("$PY" -c "from fer import FER; print('ok')" 2>/dev/null || echo "")
if [[ "$FER_OK" != "ok" ]]; then
    "$PIP" install fer --no-deps --quiet
    "$PIP" install tensorflow --quiet 2>/dev/null || \
        "$PIP" install tensorflow-cpu --quiet 2>/dev/null || \
        warn "tensorflow install failed — fer will use opencv backend"
    "$PIP" install "numpy>=2.0" --upgrade --quiet
else skip "fer ok"; fi

DF_OK=$("$PY" -c "from deepface import DeepFace; print('ok')" 2>/dev/null || echo "")
if [[ "$DF_OK" != "ok" ]]; then
    "$PIP" install deepface --quiet
    "$PIP" install "numpy>=2.0" --upgrade --quiet
else skip "deepface ok"; fi

# =============================================================================
# [7/10] Face recognition (dlib)
# =============================================================================
step "7/10" "Face recognition (dlib + face_recognition)..."

FR_OK=$("$PY" -c "import face_recognition; print('ok')" 2>/dev/null || echo "")
if [[ "$FR_OK" != "ok" ]]; then
    # Install cmake if missing
    command -v cmake &>/dev/null || {
        if command -v apt-get &>/dev/null; then
            sudo apt-get install -y cmake build-essential libopenblas-dev &>/dev/null 2>&1 || true
        elif command -v brew &>/dev/null; then
            brew install cmake &>/dev/null 2>&1 || true
        elif command -v dnf &>/dev/null; then
            sudo dnf install -y cmake gcc &>/dev/null 2>&1 || true
        elif command -v pacman &>/dev/null; then
            sudo pacman -S --noconfirm cmake &>/dev/null 2>&1 || true
        fi
    }
    "$PIP" install dlib --quiet 2>/dev/null || true
    DLIB_OK=$("$PY" -c "import dlib; print('ok')" 2>/dev/null || echo "")
    if [[ "$DLIB_OK" == "ok" ]]; then
        "$PIP" install face_recognition --quiet
        ok "face_recognition installed"
    else
        warn "dlib build failed — face identity will be disabled"
        warn "To enable: install cmake + build-essential then re-run install.sh"
    fi
else skip "face_recognition ok"; fi

# =============================================================================
# [8/10] Audio emotion (librosa + soundfile)
# =============================================================================
step "8/10" "Audio emotion (librosa + soundfile)..."
LB_OK=$("$PY" -c "import librosa; print('ok')" 2>/dev/null || echo "")
if [[ "$LB_OK" != "ok" ]]; then
    "$PIP" install librosa soundfile --quiet
    ok "librosa installed"
else skip "librosa ok"; fi

SF_OK=$("$PY" -c "import soundfile; print('ok')" 2>/dev/null || echo "")
[[ "$SF_OK" != "ok" ]] && "$PIP" install soundfile --quiet

# =============================================================================
# [9/10] Document parsing
# =============================================================================
step "9/10" "Document parsing (PDF / DOCX / EPUB)..."
for pkg in pdfplumber python-docx EbookLib beautifulsoup4; do
    import_name="${pkg,,}"
    [[ "$pkg" == "python-docx" ]] && import_name="docx"
    [[ "$pkg" == "EbookLib"    ]] && import_name="ebooklib"
    [[ "$pkg" == "beautifulsoup4" ]] && import_name="bs4"
    CHK=$("$PY" -c "import $import_name; print('ok')" 2>/dev/null || echo "")
    if [[ "$CHK" != "ok" ]]; then
        "$PIP" install "$pkg" --quiet
        ok "$pkg installed"
    else skip "$pkg ok"; fi
done

# =============================================================================
# [10/10] Write GPU config for runtime
# =============================================================================
step "10/10" "Writing GPU config..."
mkdir -p "$PROJECT_ROOT/data"
cat > "$PROJECT_ROOT/data/gpu_config.json" << JSONEOF
{
  "gpu_type": "$GPU_TYPE",
  "installed_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "platform": "$(uname -s)",
  "arch": "$(uname -m)"
}
JSONEOF
ok "GPU config written: gpu_type=$GPU_TYPE"

# =============================================================================
# Done
# =============================================================================
echo ""
echo -e "${GRN}  =============================================${RST}"
echo -e "${GRN}  AXON install complete!  Backend: $GPU_TYPE   ${RST}"
echo -e "${GRN}  =============================================${RST}"
echo ""
echo -e "  Start AXON:  ${CYN}bash scripts/launch.sh${RST}"
echo ""
