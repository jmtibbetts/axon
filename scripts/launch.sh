#!/usr/bin/env bash
# =============================================================================
#  AXON — Launch Script  (macOS + Linux)
#  Reads gpu_config.json written by install.sh to set device env vars.
#  Run from project root:  bash scripts/launch.sh
# =============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"
cd "$PROJECT_ROOT"

VENV="$PROJECT_ROOT/.venv"
PY="$VENV/bin/python"
PIP="$VENV/bin/pip"

RED='\033[0;31m'; YEL='\033[0;33m'; GRN='\033[0;32m'
CYN='\033[0;36m'; GRY='\033[0;90m'; RST='\033[0m'

echo ""
echo -e "${CYN}  =============================================${RST}"
echo -e "${CYN}       A X O N  --  Emerging Intelligence     ${RST}"
echo -e "${CYN}  =============================================${RST}"
echo ""

# ── Venv check ────────────────────────────────────────────────────────────────
if [[ ! -f "$PY" ]]; then
    echo -e "${RED}  [FATAL] Virtual environment not found.${RST}"
    echo -e "  Run first:  ${CYN}bash scripts/install.sh${RST}"
    exit 1
fi

# ── Read GPU config ───────────────────────────────────────────────────────────
GPU_CONFIG="$PROJECT_ROOT/data/gpu_config.json"
GPU_TYPE="cpu"
if [[ -f "$GPU_CONFIG" ]]; then
    GPU_TYPE=$(python3 -c "import json,sys; d=json.load(open('$GPU_CONFIG')); print(d.get('gpu_type','cpu'))" 2>/dev/null || echo "cpu")
fi
echo -e "${GRY}  Backend: $GPU_TYPE${RST}"

# ── Set device environment variables ─────────────────────────────────────────
export AXON_DEVICE="$GPU_TYPE"
if [[ "$GPU_TYPE" == "cpu" ]]; then
    export CUDA_VISIBLE_DEVICES=""          # hide CUDA from any stray imports
    export PYTORCH_ENABLE_MPS_FALLBACK=1    # safe no-op on Linux
elif [[ "$GPU_TYPE" == "mps" ]]; then
    export PYTORCH_ENABLE_MPS_FALLBACK=1    # fallback ops not yet on MPS
fi

# ── .env ─────────────────────────────────────────────────────────────────────
if [[ -f "$PROJECT_ROOT/.env" ]]; then
    set -a; source "$PROJECT_ROOT/.env"; set +a
    echo -e "${GRY}  .env loaded${RST}"
elif [[ -f "$PROJECT_ROOT/axon.env.example" ]]; then
    echo -e "${YEL}  [WARN] No .env found. Copy axon.env.example → .env and add API keys.${RST}"
fi

# ── Quick preflight ───────────────────────────────────────────────────────────
echo -e "${GRY}  Running preflight check...${RST}"
PREFLIGHT_OUT=$("$PY" - 2>&1 <<'PYEOF'
import sys

required = [
    ("flask",     "flask"),
    ("flask_socketio", "flask-socketio"),
    ("numpy",     "numpy"),
    ("cv2",       "opencv-python"),
    ("torch",     "torch"),
    ("whisper",   "openai-whisper"),
]
optional = [
    ("face_recognition", "face_recognition (face identity)"),
    ("ultralytics",      "ultralytics (YOLOv8 vision)"),
    ("fer",              "fer (facial emotion)"),
    ("librosa",          "librosa (audio emotion)"),
]

missing_req = []
missing_opt = []
for mod, label in required:
    try: __import__(mod)
    except ImportError: missing_req.append(f"{label}::run  pip install {label}")

for mod, label in optional:
    try: __import__(mod)
    except ImportError: missing_opt.append(label)

if missing_req:
    print("REQUIRED_MISSING:" + "|".join(missing_req))
    sys.exit(1)
if missing_opt:
    print("OPTIONAL_MISSING:" + ",".join(missing_opt))
sys.exit(0)
PYEOF
)
PREFLIGHT_CODE=$?

if [[ $PREFLIGHT_CODE -ne 0 ]]; then
    echo ""
    if echo "$PREFLIGHT_OUT" | grep -q "^REQUIRED_MISSING:"; then
        echo -e "${RED}  +------------------------------------------------------+${RST}"
        echo -e "${RED}  |  AXON CANNOT START — required dependencies missing   |${RST}"
        echo -e "${RED}  +------------------------------------------------------+${RST}"
        ITEMS=$(echo "$PREFLIGHT_OUT" | grep "^REQUIRED_MISSING:" | sed 's/^REQUIRED_MISSING://')
        IFS='|' read -ra PARTS <<< "$ITEMS"
        for part in "${PARTS[@]}"; do
            IFS='::' read -ra FIELDS <<< "$part"
            echo -e "${RED}  [X]  ${FIELDS[0]}${RST}"
            [[ ${#FIELDS[@]} -gt 1 ]] && echo -e "${GRY}       ${FIELDS[1]}${RST}"
        done
    else
        echo -e "${RED}  Preflight error:${RST}"
        echo "$PREFLIGHT_OUT"
    fi
    echo ""
    echo -e "  Re-run:  ${CYN}bash scripts/install.sh${RST}"
    exit 1
fi

echo -e "${GRN}  [OK] All required dependencies verified.${RST}"

if echo "$PREFLIGHT_OUT" | grep -q "^OPTIONAL_MISSING:"; then
    NAMES=$(echo "$PREFLIGHT_OUT" | grep "^OPTIONAL_MISSING:" | sed 's/^OPTIONAL_MISSING://')
    echo ""
    echo -e "${GRY}  Optional features unavailable (non-fatal):${RST}"
    IFS=',' read -ra OPTS <<< "$NAMES"
    for o in "${OPTS[@]}"; do
        echo -e "${GRY}    [-] $o${RST}"
    done
fi

# ── Launch ────────────────────────────────────────────────────────────────────
echo ""
echo -e "${CYN}  +--------------------------------------+${RST}"
echo -e "${CYN}  |  Dashboard : http://localhost:7777   |${RST}"
echo -e "${CYN}  |  Press Ctrl+C to stop                |${RST}"
echo -e "${CYN}  +--------------------------------------+${RST}"
echo ""

# Auto-open browser (best-effort)
if command -v open &>/dev/null; then
    sleep 1 && open "http://localhost:7777" &
elif command -v xdg-open &>/dev/null; then
    sleep 1 && xdg-open "http://localhost:7777" &
fi

"$PY" -m axon.ui.app
