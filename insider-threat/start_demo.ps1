# Insider Threat Detection - Complete Demo Launcher
# This script starts both the FastAPI backend and Streamlit frontend

Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Insider Threat Detection Demo" -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""

# Get the script directory
$scriptDir = Split-Path -Parent $MyInvocation.MyCommand.Path
Set-Location $scriptDir

# Check if API is already running
$apiRunning = $false
try {
    $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 2 -ErrorAction Stop
    $apiRunning = $true
    Write-Host "[INFO] FastAPI backend is already running" -ForegroundColor Green
} catch {
    Write-Host "[INFO] Starting FastAPI backend..." -ForegroundColor Yellow
    
    # Start API in background
    $apiJob = Start-Job -ScriptBlock {
        Set-Location $using:scriptDir
        python -m uvicorn app.inference_api:app --host 0.0.0.0 --port 8000
    }
    
    Write-Host "[INFO] FastAPI backend starting (Job ID: $($apiJob.Id))" -ForegroundColor Green
    Write-Host "       API will be available at: http://localhost:8000" -ForegroundColor Gray
    Write-Host "       API docs: http://localhost:8000/docs" -ForegroundColor Gray
    
    # Wait for API to be ready
    Write-Host ""
    Write-Host "Waiting for API to start..." -ForegroundColor Yellow
    $maxAttempts = 30
    $attempt = 0
    while ($attempt -lt $maxAttempts) {
        Start-Sleep -Seconds 1
        try {
            $response = Invoke-WebRequest -Uri "http://localhost:8000/health" -UseBasicParsing -TimeoutSec 1 -ErrorAction Stop
            Write-Host "[OK] API is ready!" -ForegroundColor Green
            $apiRunning = $true
            break
        } catch {
            $attempt++
            Write-Host "." -NoNewline -ForegroundColor Gray
        }
    }
    Write-Host ""
}

if (-not $apiRunning) {
    Write-Host "[WARNING] API may not be ready. Continuing anyway..." -ForegroundColor Yellow
}

# Start Streamlit
Write-Host ""
Write-Host "========================================" -ForegroundColor Cyan
Write-Host "  Starting Streamlit Frontend..." -ForegroundColor Cyan
Write-Host "========================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "The demo will open in your browser automatically." -ForegroundColor Green
Write-Host ""
Write-Host "Frontend URL: http://localhost:8501" -ForegroundColor Yellow
Write-Host "API URL:      http://localhost:8000/docs" -ForegroundColor Yellow
Write-Host ""
Write-Host "Press Ctrl+C to stop both services." -ForegroundColor Gray
Write-Host ""

# Start Streamlit (this will block)
# Use python -m streamlit for better compatibility (works even if streamlit.exe not in PATH)
Write-Host "Starting Streamlit..." -ForegroundColor Yellow
python -m streamlit run demo_app.py --server.headless false

# Cleanup: Stop API job if we started it
if (-not $apiRunning -and $apiJob) {
    Write-Host ""
    Write-Host "Stopping FastAPI backend..." -ForegroundColor Yellow
    Stop-Job -Job $apiJob -ErrorAction SilentlyContinue
    Remove-Job -Job $apiJob -ErrorAction SilentlyContinue
}

Write-Host ""
Write-Host "Demo stopped." -ForegroundColor Green

