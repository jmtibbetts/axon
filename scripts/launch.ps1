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
$tv = & $venvPy -c "import torch; print(torch.__version__)" 2>$null
$isNightly = $tv -match "dev"
if (-not $isNightly) {
    & $venvPip uninstall torch torchvision torchaudio -y --quiet 2>$null
    & $venvPip install --pre torch torchvision torchaudio --index-url https://download.pytorch.org/whl/nightly/cu128
} else {
    Write-Host "  [SKIP] PyTorch nightly ok ($tv)" -ForegroundColor DarkGray
}

Write-Host "  [3/5] Core deps..." -ForegroundColor Yellow
& $venvPip install --quiet flask flask-socketio flask-cors eventlet anthropic openai-whisper sounddevice opencv-python mediapipe edge-tts pygame numpy scipy pyaudio

Write-Host "  [4/5] Creating data dirs..." -ForegroundColor Yellow
New-Item -ItemType Directory -Force -Path "data\memory" | Out-Null

# Load SERPER_API_KEY from axon.env if it exists
$envFile = Join-Path $projectRoot "axon.env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^([^#][^=]+)=(.+)$") {
            [System.Environment]::SetEnvironmentVariable($matches[1].Trim(), $matches[2].Trim(), "Process")
        }
    }
    Write-Host "  [ENV] Loaded axon.env" -ForegroundColor DarkGray
} else {
    Write-Host "  [ENV] No axon.env found — web search uses Wikipedia fallback" -ForegroundColor DarkYellow
    Write-Host "        Create axon.env with SERPER_API_KEY=your_key for Google Search" -ForegroundColor DarkYellow
}

Write-Host "  [5/5] Launching AXON..." -ForegroundColor Green
Write-Host ""
Write-Host "  Open: http://localhost:7777" -ForegroundColor Cyan
Write-Host ""
Start-Process "http://localhost:7777"
& $venvPy -m axon.ui.app
pause
