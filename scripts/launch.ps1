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

function Is-Installed($p) {
    $r = & $venvPip show $p 2>$null
    return ($null -ne $r -and $r -ne "")
}

Write-Host "  [1/5] pip..." -ForegroundColor Yellow
& $venvPip install --upgrade pip setuptools wheel --quiet

Write-Host "  [2/5] PyTorch nightly cu128..." -ForegroundColor Yellow
$tv      = & $venvPy -c "import torch; print(torch.__version__)"        2>$null
$cudaOk  = & $venvPy -c "import torch; print(torch.cuda.is_available())" 2>$null
$isNightly = $tv -match "dev"

if (-not $isNightly -or $cudaOk -ne "True") {
    if ($cudaOk -ne "True") {
        Write-Host "  WARNING: PyTorch installed ($tv) but CUDA not available — reinstalling cu128 build..." -ForegroundColor Red
    } else {
        Write-Host "  Installing PyTorch nightly (cu128)..." -ForegroundColor Yellow
    }
    & $venvPip uninstall torch torchvision torchaudio -y --quiet 2>$null
    & $venvPip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
    $cudaOk2 = & $venvPy -c "import torch; print(torch.cuda.is_available())" 2>$null
    if ($cudaOk2 -eq "True") {
        $gpuName = & $venvPy -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
        Write-Host "  [OK] CUDA available: $gpuName" -ForegroundColor Green
    } else {
        Write-Host "  [WARN] CUDA still not available after reinstall — check CUDA 12.8 drivers" -ForegroundColor Red
        Write-Host "         Make sure NVIDIA driver >= 570 and CUDA toolkit 12.8 are installed." -ForegroundColor DarkGray
    }
} else {
    $gpuName = & $venvPy -c "import torch; print(torch.cuda.get_device_name(0))" 2>$null
    Write-Host "  [SKIP] PyTorch nightly ok ($tv) | GPU: $gpuName" -ForegroundColor DarkGray
}

Write-Host "  [3/5] Core deps..." -ForegroundColor Yellow
& $venvPip install --quiet flask flask-socketio flask-cors eventlet anthropic openai-whisper sounddevice opencv-python mediapipe edge-tts pygame numpy scipy pyaudio

Write-Host "  [3b/5] Vision deps (YOLOv8 + FER)..." -ForegroundColor Yellow
$ultraOk = & $venvPy -c "import ultralytics" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing ultralytics (YOLOv8)..." -ForegroundColor Cyan
    & $venvPip install ultralytics
} else {
    Write-Host "  [SKIP] ultralytics already installed" -ForegroundColor DarkGray
}

# Test whether FER actually works (package may be installed but broken)
$ferOk = & $venvPy -c "from fer import FER" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  fer broken or missing — trying fer==22.5.1 + tensorflow..." -ForegroundColor Cyan
    & $venvPip uninstall fer -y --quiet 2>$null
    & $venvPip install "fer==22.5.1" tensorflow
    # Re-test; if still broken install deepface as fallback
    $ferOk2 = & $venvPy -c "from fer import FER" 2>$null
    if ($LASTEXITCODE -ne 0) {
        Write-Host "  fer still broken — installing deepface as emotion backend..." -ForegroundColor Yellow
        & $venvPip install deepface
    }
} else {
    Write-Host "  [SKIP] fer working" -ForegroundColor DarkGray
}

# Ensure deepface is available as backup regardless
$dfOk = & $venvPy -c "import deepface" 2>$null
if ($LASTEXITCODE -ne 0) {
    Write-Host "  Installing deepface (emotion fallback)..." -ForegroundColor Cyan
    & $venvPip install deepface
} else {
    Write-Host "  [SKIP] deepface already installed" -ForegroundColor DarkGray
}

Write-Host "  [4/5] Creating data dirs..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\memory" | Out-Null

Write-Host "  [5/5] Launching AXON..." -ForegroundColor Green
Write-Host ""
Write-Host "  Open: http://localhost:7777" -ForegroundColor Cyan
Write-Host "  Press Ctrl+C here to stop AXON" -ForegroundColor DarkGray
Write-Host ""
Start-Process "http://localhost:7777"

# Register Ctrl+C handler so it cleanly stops AXON
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
