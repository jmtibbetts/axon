# AXON -- One-click install + launch
$Host.UI.RawUI.WindowTitle = "AXON"
$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot
Write-Host ""
Write-Host "  ==============================================" -ForegroundColor Cyan
Write-Host "       A X O N  --  Emerging Intelligence      " -ForegroundColor Cyan
Write-Host "  ==============================================" -ForegroundColor Cyan
Write-Host ""

# ── Python 3.12 ──────────────────────────────────────────────────────────────
$PY = $null
try { $v = & py -3.12 --version 2>$null; if ($v -match "3\.12") { $PY = "py -3.12" } } catch {}
if (-not $PY) { try { $v = & python --version 2>$null; if ($v -match "3\.12") { $PY = "python" } } catch {} }
if (-not $PY) {
    Write-Host "  [FATAL] Python 3.12 not found." -ForegroundColor Red
    Write-Host "          Download from https://python.org" -ForegroundColor DarkGray
    pause; exit 1
}
Write-Host "  Python: $PY" -ForegroundColor DarkGray

# ── venv ─────────────────────────────────────────────────────────────────────
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    Invoke-Expression "$PY -m venv .venv"
}
$venvPy  = "$projectRoot\.venv\Scripts\python.exe"
$venvPip = "$projectRoot\.venv\Scripts\pip.exe"

# ── [1/9] pip + setuptools ────────────────────────────────────────────────────
Write-Host ""
Write-Host "  [1/9] Upgrading pip + setuptools..." -ForegroundColor Yellow
& $venvPip install --upgrade pip wheel --quiet
& $venvPip install "setuptools<82" --quiet

# ── [2/9] Remove packages that conflict with modern PyTorch ──────────────────
Write-Host "  [2/9] Removing conflicting packages (fer, facenet-pytorch)..." -ForegroundColor Yellow
& $venvPip uninstall fer facenet-pytorch -y --quiet 2>$null

# ── [3/9] PyTorch cu128 ───────────────────────────────────────────────────────
Write-Host "  [3/9] Checking PyTorch + CUDA..." -ForegroundColor Yellow
$cudaOk = & $venvPy -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaOk -ne "True") {
    Write-Host "  CUDA not available — installing PyTorch nightly cu128..." -ForegroundColor Red
    & $venvPip uninstall torch torchvision torchaudio -y --quiet 2>$null
    & $venvPip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    $cudaOk2 = & $venvPy -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaOk2 -eq "True") {
        $gpuName = & $venvPy -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
        Write-Host "  [OK] CUDA ready: $gpuName" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] CUDA still not available. Ensure NVIDIA driver >= 570 + CUDA 12.8." -ForegroundColor Red
    }
} else {
    $tv      = & $venvPy -c "import torch; print(torch.__version__)" 2>$null
    $gpuName = & $venvPy -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
    Write-Host "  [SKIP] torch $tv | GPU: $gpuName" -ForegroundColor DarkGray
}

# ── [4/9] Core deps ───────────────────────────────────────────────────────────
Write-Host "  [4/9] Core dependencies..." -ForegroundColor Yellow
& $venvPip install --quiet `
    flask flask-socketio flask-cors eventlet `
    anthropic openai-whisper `
    sounddevice pyaudio `
    opencv-python `
    edge-tts pygame `
    scipy requests

$npv = & $venvPy -c "import numpy; print(numpy.__version__)" 2>$null
if (-not ($npv -match "^2\.")) {
    Write-Host "  Upgrading numpy to 2.x..." -ForegroundColor Cyan
    & $venvPip install "numpy>=2.0" --quiet
} else { Write-Host "  [SKIP] numpy $npv" -ForegroundColor DarkGray }

# ── [5/9] Vision: YOLOv8 + emotion backends ──────────────────────────────────
Write-Host "  [5/9] Vision deps (YOLOv8 + emotion backends)..." -ForegroundColor Yellow
$ultraOk = & $venvPy -c "import ultralytics; print('ok')" 2>$null
if ($ultraOk -ne "ok") {
    Write-Host "  Installing ultralytics (YOLOv8)..." -ForegroundColor Cyan
    & $venvPip install ultralytics
} else { Write-Host "  [SKIP] ultralytics ok" -ForegroundColor DarkGray }

$tfkOk = & $venvPy -c "import tf_keras; print('ok')" 2>$null
if ($tfkOk -ne "ok") {
    Write-Host "  Installing tf-keras..." -ForegroundColor Cyan
    & $venvPip install tf-keras --quiet
} else { Write-Host "  [SKIP] tf-keras ok" -ForegroundColor DarkGray }

# fer is tried first; deepface is the fallback — both installed, one must work
$ferOk = & $venvPy -c "from fer import FER; print('ok')" 2>$null
if ($ferOk -ne "ok") {
    Write-Host "  Installing fer..." -ForegroundColor Cyan
    & $venvPip install fer --quiet
} else { Write-Host "  [SKIP] fer ok" -ForegroundColor DarkGray }

$dfOk = & $venvPy -c "from deepface import DeepFace; print('ok')" 2>$null
if ($dfOk -ne "ok") {
    Write-Host "  Installing deepface..." -ForegroundColor Cyan
    & $venvPip install deepface
} else { Write-Host "  [SKIP] deepface ok" -ForegroundColor DarkGray }

# ── [6/9] Face recognition (dlib prebuilt wheel → face_recognition) ──────────
Write-Host "  [6/9] Face recognition (dlib + face_recognition)..." -ForegroundColor Yellow
$frOk = & $venvPy -c "import face_recognition; print('ok')" 2>$null
if ($frOk -ne "ok") {
    Write-Host "  Attempting dlib install (requires cmake)..." -ForegroundColor Cyan
    & $venvPip install dlib --quiet 2>$null
    $dlibOk = & $venvPy -c "import dlib; print('ok')" 2>$null
    if ($dlibOk -ne "ok") {
        Write-Host "  cmake not found — using prebuilt dlib wheel..." -ForegroundColor Cyan
        $dlibWheel = "https://github.com/z-mahmud22/Dlib_Windows_Python3.x/raw/main/dlib-19.24.1-cp312-cp312-win_amd64.whl"
        & $venvPip install $dlibWheel
    }
    $dlibOk2 = & $venvPy -c "import dlib; print('ok')" 2>$null
    if ($dlibOk2 -eq "ok") {
        & $venvPip install face_recognition --quiet
        $frCheck = & $venvPy -c "import face_recognition; print('ok')" 2>$null
        if ($frCheck -eq "ok") {
            Write-Host "  [OK] face_recognition ready" -ForegroundColor Green
        } else {
            Write-Host "  [WARN] face_recognition install failed — face identity disabled" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  [WARN] dlib install failed — face identity disabled" -ForegroundColor Yellow
    }
} else { Write-Host "  [SKIP] face_recognition ok" -ForegroundColor DarkGray }

# ── [7/9] Audio emotion (librosa + soundfile) ─────────────────────────────────
Write-Host "  [7/9] Audio emotion (librosa + soundfile)..." -ForegroundColor Yellow
$lbOk = & $venvPy -c "import librosa; print('ok')" 2>$null
if ($lbOk -ne "ok") {
    & $venvPip install librosa soundfile --quiet
    $lbCheck = & $venvPy -c "import librosa; print('ok')" 2>$null
    if ($lbCheck -eq "ok") {
        Write-Host "  [OK] librosa ready" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] librosa install failed — audio emotion falls back to basic energy analysis" -ForegroundColor Yellow
    }
} else { Write-Host "  [SKIP] librosa ok" -ForegroundColor DarkGray }
$sfOk = & $venvPy -c "import soundfile; print('ok')" 2>$null
if ($sfOk -ne "ok") { & $venvPip install soundfile --quiet }

# ── [8/9] Data directories ────────────────────────────────────────────────────
Write-Host "  [8/9] Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\memory" | Out-Null

# ── [9/9] Preflight check ─────────────────────────────────────────────────────
Write-Host ""
Write-Host "  [9/9] Running preflight check..." -ForegroundColor Yellow

$preflight_script = @"
import sys, importlib

# ── Required: hard block ──────────────────────────────────────────────────────
# Every entry here maps to real functionality. If it's missing, something
# breaks — no silent degradation for these.
REQUIRED = [
    # Neural engine
    ("torch",           "PyTorch",          "pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"),
    ("torchvision",     "torchvision",      "pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128"),
    ("numpy",           "NumPy",            "pip install numpy>=2.0"),
    ("scipy",           "SciPy",            "pip install scipy"),
    # Vision
    ("cv2",             "OpenCV",           "pip install opencv-python"),
    ("ultralytics",     "YOLOv8",           "pip install ultralytics"),
    # Web UI
    ("flask",           "Flask",            "pip install flask"),
    ("flask_socketio",  "Flask-SocketIO",   "pip install flask-socketio"),
    ("flask_cors",      "Flask-CORS",       "pip install flask-cors"),
    ("eventlet",        "eventlet",         "pip install eventlet"),
    # Voice input
    ("whisper",         "Whisper (STT)",    "pip install openai-whisper"),
    ("sounddevice",     "sounddevice",      "pip install sounddevice"),
    # Voice output
    ("edge_tts",        "edge-tts",         "pip install edge-tts"),
    ("pygame",          "pygame",           "pip install pygame"),
    # Storage
    ("sqlite3",         "sqlite3",          "(built into Python — reinstall Python 3.12)"),
]

# ── Soft-optional: warn only ──────────────────────────────────────────────────
# face_recognition: face identity panel disabled, everything else works fine.
# librosa:          audio emotion falls back to basic energy/ZCR analysis.
OPTIONAL = [
    ("face_recognition", "face_recognition (person identity)", "see README — needs dlib prebuilt wheel on Windows"),
    ("librosa",          "librosa (audio emotion)",            "pip install librosa soundfile"),
]

# ── Emotion backend: at least one of fer / deepface must be present ───────────
EMOTION_BACKENDS = [("fer", "FER"), ("deepface", "deepface")]

missing_required = []
missing_optional = []

for mod, label, fix in REQUIRED:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing_required.append((label, fix))

for mod, label, fix in OPTIONAL:
    try:
        importlib.import_module(mod)
    except ImportError:
        missing_optional.append((label, fix))

# torch CUDA check
try:
    import torch
    if not torch.cuda.is_available():
        missing_required.append((
            "PyTorch CUDA (GPU)",
            "reinstall: pip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128"
        ))
except Exception:
    pass

# Emotion backend: need at least one
emo_ok = False
for mod, label in EMOTION_BACKENDS:
    try:
        importlib.import_module(mod)
        emo_ok = True
        break
    except ImportError:
        pass
if not emo_ok:
    missing_required.append((
        "Emotion backend (fer or deepface)",
        "pip install fer   OR   pip install deepface"
    ))

if missing_optional:
    print("OPTIONAL_MISSING:" + "|".join(f"{l}::{f}" for l,f in missing_optional))

if missing_required:
    print("REQUIRED_MISSING:" + "|".join(f"{l}::{f}" for l,f in missing_required))
    sys.exit(1)

sys.exit(0)
"@

$preflight_out  = & $venvPy -c $preflight_script 2>&1
$preflight_exit = $LASTEXITCODE

# Print optional warnings
foreach ($line in $preflight_out) {
    if ($line -match "^OPTIONAL_MISSING:(.+)") {
        $items = $Matches[1] -split "\|"
        foreach ($item in $items) {
            $parts = $item -split "::"
            Write-Host ("  [WARN] " + $parts[0] + " not available") -ForegroundColor Yellow
            Write-Host ("         " + $parts[1]) -ForegroundColor DarkGray
        }
    }
}

# Hard stop if anything required is missing
if ($preflight_exit -ne 0) {
    Write-Host ""
    Write-Host "  ╔══════════════════════════════════════════════════════╗" -ForegroundColor Red
    Write-Host "  ║   AXON CANNOT START — required dependencies missing  ║" -ForegroundColor Red
    Write-Host "  ╚══════════════════════════════════════════════════════╝" -ForegroundColor Red
    Write-Host ""
    foreach ($line in $preflight_out) {
        if ($line -match "^REQUIRED_MISSING:(.+)") {
            $items = $Matches[1] -split "\|"
            foreach ($item in $items) {
                $parts = $item -split "::"
                Write-Host ("  ✗  " + $parts[0]) -ForegroundColor Red
                Write-Host ("     " + $parts[1]) -ForegroundColor DarkGray
                Write-Host ""
            }
        }
    }
    Write-Host "  Re-run launch.ps1 to retry installation, or fix manually and try again." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "  [OK] All required dependencies verified." -ForegroundColor Green

# ── Launch ────────────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  ┌──────────────────────────────────────┐" -ForegroundColor Cyan
Write-Host "  │  Dashboard: http://localhost:7777     │" -ForegroundColor Cyan
Write-Host "  │  Press Ctrl+C to stop                │" -ForegroundColor Cyan
Write-Host "  └──────────────────────────────────────┘" -ForegroundColor Cyan
Write-Host ""
Start-Process "http://localhost:7777"

$proc = Start-Process -FilePath $venvPy `
    -ArgumentList "-m", "axon.ui.app" `
    -NoNewWindow -PassThru

try {
    $proc.WaitForExit()
} finally {
    if (-not $proc.HasExited) {
        Write-Host ""
        Write-Host "  [AXON] Stopping..." -ForegroundColor Yellow
        $proc.CloseMainWindow() | Out-Null
        Start-Sleep -Milliseconds 800
        if (-not $proc.HasExited) { $proc.Kill() }
    }
    Write-Host "  [AXON] Exited." -ForegroundColor Cyan
}
pause
