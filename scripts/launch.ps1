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
if (-not $PY) { Write-Host "  Python 3.12 required. Download from python.org" -ForegroundColor Red; pause; exit 1 }
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

# numpy 2.x required by opencv-python 4.13+
$npv = & $venvPy -c "import numpy; print(numpy.__version__)" 2>$null
if (-not ($npv -match "^2\.")) {
    Write-Host "  Upgrading numpy to 2.x..." -ForegroundColor Cyan
    & $venvPip install "numpy>=2.0" --quiet
} else {
    Write-Host "  [SKIP] numpy $npv" -ForegroundColor DarkGray
}

# ── [5/9] Vision: YOLOv8 + deepface ─────────────────────────────────────────
Write-Host "  [5/9] Vision deps (YOLOv8 + deepface)..." -ForegroundColor Yellow
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

$dfOk = & $venvPy -c "from deepface import DeepFace; print('ok')" 2>$null
if ($dfOk -ne "ok") {
    Write-Host "  Installing deepface..." -ForegroundColor Cyan
    & $venvPip install deepface
} else { Write-Host "  [SKIP] deepface ok" -ForegroundColor DarkGray }

# ── [6/9] Face recognition (dlib prebuilt wheel → face_recognition) ──────────
Write-Host "  [6/9] Face recognition (dlib + face_recognition)..." -ForegroundColor Yellow
$frOk = & $venvPy -c "import face_recognition; print('ok')" 2>$null
if ($frOk -ne "ok") {
    # Try pip dlib first (works if cmake is installed)
    Write-Host "  Attempting dlib install (requires cmake)..." -ForegroundColor Cyan
    & $venvPip install dlib --quiet 2>$null
    $dlibOk = & $venvPy -c "import dlib; print('ok')" 2>$null
    if ($dlibOk -ne "ok") {
        # Fall back to prebuilt wheel for Python 3.12 Windows x64
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
        Write-Host "         AXON will still run without person recognition." -ForegroundColor DarkGray
    }
} else {
    Write-Host "  [SKIP] face_recognition ok" -ForegroundColor DarkGray
}

# ── [7/9] Audio emotion (librosa + soundfile) ─────────────────────────────────
Write-Host "  [7/9] Audio emotion (librosa + soundfile)..." -ForegroundColor Yellow
$lbOk = & $venvPy -c "import librosa; print('ok')" 2>$null
if ($lbOk -ne "ok") {
    Write-Host "  Installing librosa..." -ForegroundColor Cyan
    & $venvPip install librosa soundfile --quiet
    $lbCheck = & $venvPy -c "import librosa; print('ok')" 2>$null
    if ($lbCheck -eq "ok") {
        Write-Host "  [OK] librosa ready" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] librosa install failed — audio emotion disabled" -ForegroundColor Yellow
    }
} else {
    Write-Host "  [SKIP] librosa ok" -ForegroundColor DarkGray
}
$sfOk = & $venvPy -c "import soundfile; print('ok')" 2>$null
if ($sfOk -ne "ok") {
    & $venvPip install soundfile --quiet
} else {
    Write-Host "  [SKIP] soundfile ok" -ForegroundColor DarkGray
}

# ── [8/9] Data directories ────────────────────────────────────────────────────
Write-Host "  [8/9] Creating data directories..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\memory" | Out-Null
Write-Host "  [OK] data/memory ready" -ForegroundColor DarkGray

# ── [9/9] Launch ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "  [9/9] Starting AXON..." -ForegroundColor Green
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
