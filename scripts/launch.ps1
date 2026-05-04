# AXON -- One-click install + launch
# ASCII-only - no Unicode box characters, no special symbols
$Host.UI.RawUI.WindowTitle = "AXON"
$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot
Write-Host ""
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host "       A X O N  --  Emerging Intelligence     " -ForegroundColor Cyan
Write-Host "  =============================================" -ForegroundColor Cyan
Write-Host ""

# --------------------------------------------------------------------------
# [0] Python 3.12
# --------------------------------------------------------------------------
$PY = $null
try { $v = & py -3.12 --version 2>$null; if ($v -match "3\.12") { $PY = "py -3.12" } } catch {}
if (-not $PY) { try { $v = & python --version 2>$null; if ($v -match "3\.12") { $PY = "python" } } catch {} }
if (-not $PY) {
    Write-Host "  [FATAL] Python 3.12 not found." -ForegroundColor Red
    Write-Host "          Download from https://python.org" -ForegroundColor DarkGray
    pause; exit 1
}
Write-Host "  Python: $PY" -ForegroundColor DarkGray

# --------------------------------------------------------------------------
# venv
# --------------------------------------------------------------------------
if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  Creating virtual environment..." -ForegroundColor Yellow
    Invoke-Expression "$PY -m venv .venv"
}
$venvPy  = "$projectRoot\.venv\Scripts\python.exe"
$venvPip = "$projectRoot\.venv\Scripts\pip.exe"

# --------------------------------------------------------------------------
# [1/10] pip + setuptools
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "  [1/10] Upgrading pip + setuptools..." -ForegroundColor Yellow
& $venvPip install --upgrade pip wheel --quiet
& $venvPip install "setuptools<82" --quiet

# --------------------------------------------------------------------------
# [2/10] Remove conflicting packages
# --------------------------------------------------------------------------
Write-Host "  [2/10] Removing conflicting packages (fer, facenet-pytorch)..." -ForegroundColor Yellow
& $venvPip uninstall fer facenet-pytorch -y --quiet 2>$null

# --------------------------------------------------------------------------
# [3/10] PyTorch cu128
# --------------------------------------------------------------------------
Write-Host "  [3/10] Checking PyTorch + CUDA..." -ForegroundColor Yellow
$cudaOk = & $venvPy -c "import torch; print(torch.cuda.is_available())" 2>$null
if ($cudaOk -ne "True") {
    Write-Host "  CUDA not available -- installing PyTorch nightly cu128..." -ForegroundColor Red
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

# --------------------------------------------------------------------------
# [4/10] Core deps
# --------------------------------------------------------------------------
Write-Host "  [4/10] Core dependencies..." -ForegroundColor Yellow
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
} else {
    Write-Host "  [SKIP] numpy $npv" -ForegroundColor DarkGray
}

# --------------------------------------------------------------------------
# [5/10] Vision: YOLOv8 + emotion backends
# --------------------------------------------------------------------------
Write-Host "  [5/10] Vision deps (YOLOv8 + emotion backends)..." -ForegroundColor Yellow
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

# --------------------------------------------------------------------------
# [6/10] Face recognition
# --------------------------------------------------------------------------
Write-Host "  [6/10] Face recognition (dlib + face_recognition)..." -ForegroundColor Yellow
$frOk = & $venvPy -c "import face_recognition; print('ok')" 2>$null
if ($frOk -ne "ok") {
    Write-Host "  Attempting dlib install (requires cmake)..." -ForegroundColor Cyan
    & $venvPip install dlib --quiet 2>$null
    $dlibOk = & $venvPy -c "import dlib; print('ok')" 2>$null
    if ($dlibOk -ne "ok") {
        Write-Host "  cmake not found -- using prebuilt dlib wheel..." -ForegroundColor Cyan
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
            Write-Host "  [WARN] face_recognition install failed -- face identity disabled" -ForegroundColor Yellow
        }
    } else {
        Write-Host "  [WARN] dlib install failed -- face identity disabled" -ForegroundColor Yellow
    }
} else { Write-Host "  [SKIP] face_recognition ok" -ForegroundColor DarkGray }

# --------------------------------------------------------------------------
# [7/10] Audio emotion (librosa + soundfile)
# --------------------------------------------------------------------------
Write-Host "  [7/10] Audio emotion (librosa + soundfile)..." -ForegroundColor Yellow
$lbOk = & $venvPy -c "import librosa; print('ok')" 2>$null
if ($lbOk -ne "ok") {
    & $venvPip install librosa soundfile --quiet
    $lbCheck = & $venvPy -c "import librosa; print('ok')" 2>$null
    if ($lbCheck -eq "ok") {
        Write-Host "  [OK] librosa ready" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] librosa failed -- audio emotion falls back to basic energy analysis" -ForegroundColor Yellow
    }
} else { Write-Host "  [SKIP] librosa ok" -ForegroundColor DarkGray }

$sfOk = & $venvPy -c "import soundfile; print('ok')" 2>$null
if ($sfOk -ne "ok") { & $venvPip install soundfile --quiet }

# --------------------------------------------------------------------------
# [8/10] Document parsing (PDF / DOCX / EPUB)
# --------------------------------------------------------------------------
Write-Host "  [8/10] Document parsing libraries..." -ForegroundColor Yellow
$pdfOk = & $venvPy -c "import pdfplumber; print('ok')" 2>$null
if ($pdfOk -ne "ok") {
    Write-Host "  Installing pdfplumber..." -ForegroundColor Cyan
    & $venvPip install pdfplumber --quiet
} else { Write-Host "  [SKIP] pdfplumber ok" -ForegroundColor DarkGray }

$docxOk = & $venvPy -c "import docx; print('ok')" 2>$null
if ($docxOk -ne "ok") {
    Write-Host "  Installing python-docx..." -ForegroundColor Cyan
    & $venvPip install python-docx --quiet
} else { Write-Host "  [SKIP] python-docx ok" -ForegroundColor DarkGray }

$epubOk = & $venvPy -c "import ebooklib; print('ok')" 2>$null
if ($epubOk -ne "ok") {
    Write-Host "  Installing EbookLib..." -ForegroundColor Cyan
    & $venvPip install EbookLib --quiet
} else { Write-Host "  [SKIP] EbookLib ok" -ForegroundColor DarkGray }

# --------------------------------------------------------------------------
# [9/10] Data directories
# --------------------------------------------------------------------------
Write-Host "  [9/10] Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\memory" | Out-Null

# --------------------------------------------------------------------------
# [10/10] Preflight check
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "  [10/10] Running preflight check..." -ForegroundColor Yellow

$preflight_script = @"
import sys

REQUIRED = [
    ("torch",           "PyTorch",          "pip install --pre torch --index-url https://download.pytorch.org/whl/nightly/cu128"),
    ("torchvision",     "torchvision",      "pip install --pre torchvision --index-url https://download.pytorch.org/whl/nightly/cu128"),
    ("numpy",           "NumPy",            "pip install numpy>=2.0"),
    ("scipy",           "SciPy",            "pip install scipy"),
    ("cv2",             "OpenCV",           "pip install opencv-python"),
    ("ultralytics",     "YOLOv8",           "pip install ultralytics"),
    ("flask",           "Flask",            "pip install flask"),
    ("flask_socketio",  "Flask-SocketIO",   "pip install flask-socketio"),
    ("flask_cors",      "Flask-CORS",       "pip install flask-cors"),
    ("eventlet",        "eventlet",         "pip install eventlet"),
    ("whisper",         "Whisper",          "pip install openai-whisper"),
    ("sounddevice",     "sounddevice",      "pip install sounddevice"),
    ("edge_tts",        "edge-tts",         "pip install edge-tts"),
    ("pygame",          "pygame",           "pip install pygame"),
]

OPTIONAL = [
    ("face_recognition","face_recognition"),
    ("librosa",         "librosa"),
    ("pdfplumber",      "pdfplumber"),
    ("docx",            "python-docx"),
    ("ebooklib",        "EbookLib"),
    ("fer",             "fer"),
    ("deepface",        "deepface"),
]

missing_required = []
for mod, name, fix in REQUIRED:
    try:
        __import__(mod)
    except ImportError:
        missing_required.append((name, fix))

missing_optional = []
for mod, name in OPTIONAL:
    try:
        __import__(mod)
    except ImportError:
        missing_optional.append(name)

if missing_required:
    items = "|".join(name + "::" + fix for name, fix in missing_required)
    print("REQUIRED_MISSING:" + items)
    sys.exit(1)

if missing_optional:
    print("OPTIONAL_MISSING:" + ",".join(missing_optional))

sys.exit(0)
"@

$preflight_out  = & $venvPy -c $preflight_script 2>&1
$preflight_exit = $LASTEXITCODE

if ($preflight_exit -ne 0) {
    Write-Host ""
    Write-Host "  +------------------------------------------------------+" -ForegroundColor Red
    Write-Host "  |  AXON CANNOT START -- required dependencies missing  |" -ForegroundColor Red
    Write-Host "  +------------------------------------------------------+" -ForegroundColor Red
    Write-Host ""
    foreach ($line in $preflight_out) {
        if ($line -match "^REQUIRED_MISSING:(.+)") {
            $items = $Matches[1] -split "\|"
            foreach ($item in $items) {
                $parts = $item -split "::"
                Write-Host ("  [X]  " + $parts[0]) -ForegroundColor Red
                if ($parts.Length -gt 1) {
                    Write-Host ("       " + $parts[1]) -ForegroundColor DarkGray
                }
                Write-Host ""
            }
        }
    }
    Write-Host "  Re-run launch.ps1 to retry, or fix manually and try again." -ForegroundColor Yellow
    pause
    exit 1
}

Write-Host "  [OK] All required dependencies verified." -ForegroundColor Green

foreach ($line in $preflight_out) {
    if ($line -match "^OPTIONAL_MISSING:(.+)") {
        $names = $Matches[1] -split ","
        Write-Host ""
        Write-Host "  Optional features unavailable (non-fatal):" -ForegroundColor DarkGray
        foreach ($n in $names) {
            Write-Host ("    [-] " + $n) -ForegroundColor DarkGray
        }
    }
}

# --------------------------------------------------------------------------
# Launch
# --------------------------------------------------------------------------
Write-Host ""
Write-Host "  +--------------------------------------+" -ForegroundColor Cyan
Write-Host "  |  Dashboard : http://localhost:7777   |" -ForegroundColor Cyan
Write-Host "  |  Press Ctrl+C to stop                |" -ForegroundColor Cyan
Write-Host "  +--------------------------------------+" -ForegroundColor Cyan
Write-Host ""
Start-Sleep -Milliseconds 800
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
