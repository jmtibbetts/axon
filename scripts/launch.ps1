# AXON -- One-click launch
$Host.UI.RawUI.WindowTitle = "AXON"
$scriptDir   = Split-Path -Parent $MyInvocation.MyCommand.Path
$projectRoot = Split-Path -Parent $scriptDir
Set-Location $projectRoot
Write-Host ""
Write-Host "  ==============================================" -ForegroundColor Cyan
Write-Host "       A X O N  --  Emerging Intelligence      " -ForegroundColor Cyan
Write-Host "  ==============================================" -ForegroundColor Cyan
Write-Host ""

$PY = $null
try { $v = & py -3.12 --version 2>$null; if ($v -match "3\.12") { $PY = "py -3.12" } } catch {}
if (-not $PY) { try { $v = & python --version 2>$null; if ($v -match "3\.12") { $PY = "python" } } catch {} }
if (-not $PY) { Write-Host "  Python 3.12 required." -ForegroundColor Red; pause; exit 1 }

if (-not (Test-Path ".venv\Scripts\python.exe")) {
    Write-Host "  Creating venv..." -ForegroundColor Yellow
    Invoke-Expression "$PY -m venv .venv"
}

$venvPy  = "$projectRoot\.venv\Scripts\python.exe"
$venvPip = "$projectRoot\.venv\Scripts\pip.exe"

# ---------------------------------------------------------------------------
Write-Host "  [1/6] pip..." -ForegroundColor Yellow
& $venvPip install --upgrade pip setuptools wheel --quiet

# ---------------------------------------------------------------------------
# fer and facenet-pytorch conflict with modern torch + numpy.
# Remove them so they cannot drag torch back to 2.2.2 (CPU-only).
Write-Host "  [2/6] Removing conflicting packages (fer, facenet-pytorch)..." -ForegroundColor Yellow
& $venvPip uninstall fer facenet-pytorch -y --quiet 2>$null

# ---------------------------------------------------------------------------
Write-Host "  [3/6] PyTorch nightly cu128..." -ForegroundColor Yellow
$tv     = & $venvPy -c 'import torch; print(torch.__version__)' 2>$null
$cudaOk = & $venvPy -c 'import torch; print(torch.cuda.is_available())' 2>$null

if ($cudaOk -ne "True") {
    Write-Host "  CUDA not available (torch=$tv). Installing cu128 build..." -ForegroundColor Red
    & $venvPip uninstall torch torchvision torchaudio -y --quiet 2>$null
    & $venvPip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    $cudaOk2 = & $venvPy -c 'import torch; print(torch.cuda.is_available())' 2>$null
    if ($cudaOk2 -eq "True") {
        $gpuName = & $venvPy -c 'import torch; print(torch.cuda.get_device_name(0))' 2>$null
        Write-Host "  [OK] CUDA ready: $gpuName" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] CUDA still not available. Check NVIDIA driver >= 570 + CUDA 12.8." -ForegroundColor Red
    }
} else {
    $gpuName = & $venvPy -c 'import torch; print(torch.cuda.get_device_name(0))' 2>$null
    Write-Host "  [SKIP] torch ok ($tv) | GPU: $gpuName" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
Write-Host "  [4/6] Core deps..." -ForegroundColor Yellow
& $venvPip install --quiet flask flask-socketio flask-cors eventlet anthropic openai-whisper sounddevice opencv-python mediapipe edge-tts pygame scipy pyaudio

# numpy: must be >=2 for opencv-python 4.13, but we pin to avoid breakage
$npv = & $venvPy -c 'import numpy; print(numpy.__version__)' 2>$null
if (-not ($npv -match "^2\.")) {
    Write-Host "  Upgrading numpy to 2.x..." -ForegroundColor Cyan
    & $venvPip install "numpy>=2.0" --quiet
} else {
    Write-Host "  [SKIP] numpy ok ($npv)" -ForegroundColor DarkGray
}

# ---------------------------------------------------------------------------
Write-Host "  [5/6] Vision deps (YOLOv8 + deepface)..." -ForegroundColor Yellow

$ultraOk = & $venvPy -c 'import ultralytics' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing ultralytics (YOLOv8)..." -ForegroundColor Cyan
    & $venvPip install ultralytics
} else {
    Write-Host "  [SKIP] ultralytics ok" -ForegroundColor DarkGray
}

# deepface is our emotion backend (no torch version conflicts)
$dfOk = & $venvPy -c 'import deepface' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing deepface..." -ForegroundColor Cyan
    & $venvPip install deepface
} else {
    Write-Host "  [SKIP] deepface ok" -ForegroundColor DarkGray
}

# Verify deepface can actually run
$dfTest = & $venvPy -c 'from deepface import DeepFace; print("ok")' 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  [WARN] deepface import failed -- reinstalling..." -ForegroundColor Yellow
    & $venvPip install deepface --quiet
} else {
    Write-Host "  [OK] deepface ready" -ForegroundColor Green
}

# ---------------------------------------------------------------------------
Write-Host "  [6/6] Creating data dirs..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\memory" | Out-Null

# ---------------------------------------------------------------------------
Write-Host "  Launching AXON..." -ForegroundColor Green
Write-Host ""
Write-Host "  Open: http://localhost:7777" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C here to stop AXON" -ForegroundColor DarkGray
Write-Host ""
Start-Process "http://localhost:7777"

[Console]::TreatControlCAsInput = $false

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
