# Multi-LLM Development Framework - Service Starter
# Usage: .\scripts\start_services.ps1 [-All] [-LiteLLM] [-Router] [-vLLM] [-LlamaCpp]

param(
    [switch]$All,
    [switch]$LiteLLM,
    [switch]$Router,
    [switch]$vLLM,
    [switch]$LlamaCpp
)

$ErrorActionPreference = "Stop"
$PROJECT_ROOT = Split-Path -Parent $PSScriptRoot

Write-Host "=== Multi-LLM Development Framework ===" -ForegroundColor Cyan
Write-Host "Project Root: $PROJECT_ROOT"

# Load environment variables
$envFile = Join-Path $PROJECT_ROOT ".env"
if (Test-Path $envFile) {
    Get-Content $envFile | ForEach-Object {
        if ($_ -match "^([^=]+)=(.*)$") {
            [Environment]::SetEnvironmentVariable($matches[1], $matches[2], "Process")
        }
    }
    Write-Host "Loaded .env file" -ForegroundColor Green
}

function Start-LiteLLM {
    Write-Host "`n[LiteLLM Proxy] Starting on port 4000..." -ForegroundColor Yellow
    $configPath = Join-Path $PROJECT_ROOT "config\litellm.yaml"
    if (-not (Test-Path $configPath)) {
        $configPath = Join-Path $PROJECT_ROOT "config\litellm.example.yaml"
    }
    Start-Process -NoNewWindow -FilePath "litellm" -ArgumentList "--config", $configPath, "--port", "4000"
    Write-Host "[LiteLLM Proxy] Started" -ForegroundColor Green
}

function Start-PolicyRouter {
    Write-Host "`n[Policy Router] Starting on port 5000..." -ForegroundColor Yellow
    $routerPath = Join-Path $PROJECT_ROOT "src\router\policy_router.py"
    Start-Process -NoNewWindow -FilePath "uvicorn" -ArgumentList "src.router.policy_router:app", "--host", "0.0.0.0", "--port", "5000"
    Write-Host "[Policy Router] Started" -ForegroundColor Green
}

function Start-vLLM {
    param([string]$Model = "Qwen/Qwen2.5-14B-Instruct")
    Write-Host "`n[vLLM] Starting on port 8000..." -ForegroundColor Yellow
    Start-Process -NoNewWindow -FilePath "vllm" -ArgumentList "serve", $Model, "--port", "8000"
    Write-Host "[vLLM] Started with model: $Model" -ForegroundColor Green
}

function Start-LlamaCpp {
    param(
        [string]$ModelPath = "models/qwen2.5-7b-q4_k_m.gguf",
        [int]$ContextSize = 8192,
        [int]$NGPULayers = 24
    )
    Write-Host "`n[llama.cpp] Starting on port 8080..." -ForegroundColor Yellow

    $llamaPath = $env:LLAMA_CPP_PATH
    if (-not $llamaPath) {
        $llamaPath = "llama-server"
    }

    Start-Process -NoNewWindow -FilePath $llamaPath -ArgumentList `
        "-m", $ModelPath, `
        "-c", $ContextSize, `
        "-ngl", $NGPULayers, `
        "--port", "8080"
    Write-Host "[llama.cpp] Started" -ForegroundColor Green
}

# Start services based on flags
if ($All -or (-not $LiteLLM -and -not $Router -and -not $vLLM -and -not $LlamaCpp)) {
    Write-Host "`nStarting all services..." -ForegroundColor Cyan
    Start-LlamaCpp
    Start-Sleep -Seconds 2
    Start-LiteLLM
    Start-Sleep -Seconds 2
    Start-PolicyRouter
    Write-Host "`nvLLM not started by default (requires explicit -vLLM flag)" -ForegroundColor Yellow
}
else {
    if ($LlamaCpp) { Start-LlamaCpp }
    if ($vLLM) { Start-vLLM }
    if ($LiteLLM) { Start-LiteLLM }
    if ($Router) { Start-PolicyRouter }
}

Write-Host "`n=== Services Started ===" -ForegroundColor Cyan
Write-Host "LiteLLM Proxy: http://localhost:4000"
Write-Host "Policy Router: http://localhost:5000"
Write-Host "llama.cpp:     http://localhost:8080"
Write-Host "vLLM:          http://localhost:8000 (if started)"
Write-Host "`nHealth checks:"
Write-Host "  curl http://localhost:5000/health"
Write-Host "  curl http://localhost:4000/health"
