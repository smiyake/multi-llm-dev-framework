# Multi-LLM Development Framework - Nightly Job Template
# Customize this script for your project's nightly automation needs
#
# To schedule with Windows Task Scheduler:
# schtasks /Create /SC DAILY /TN "MultiLLMNightly" /TR "powershell -ExecutionPolicy Bypass -File D:\path\to\scripts\nightly.ps1" /ST 02:00

param(
    [string]$WorkspaceDir = (Split-Path -Parent $PSScriptRoot)
)

$ErrorActionPreference = "Stop"
$LogFile = Join-Path $WorkspaceDir ".multi-llm\logs\nightly-$(Get-Date -Format 'yyyy-MM-dd').log"

function Write-Log {
    param([string]$Message, [string]$Level = "INFO")
    $timestamp = Get-Date -Format "yyyy-MM-dd HH:mm:ss"
    $logMessage = "[$timestamp] [$Level] $Message"
    Write-Host $logMessage
    Add-Content -Path $LogFile -Value $logMessage
}

# Initialize
Write-Log "=== Nightly Job Started ==="
Write-Log "Workspace: $WorkspaceDir"

try {
    # Step 1: Health check
    Write-Log "Step 1: Service Health Check"
    $services = @(
        @{Name="Policy Router"; Url="http://localhost:5000/health"},
        @{Name="LiteLLM"; Url="http://localhost:4000/health"}
    )
    foreach ($svc in $services) {
        try {
            $response = Invoke-RestMethod -Uri $svc.Url -TimeoutSec 5
            Write-Log "  $($svc.Name): OK"
        }
        catch {
            Write-Log "  $($svc.Name): OFFLINE" "WARN"
        }
    }

    # Step 2: Process pending tasks
    Write-Log "Step 2: Process Pending Tasks"
    $pendingDir = Join-Path $WorkspaceDir ".multi-llm\tasks\pending"
    $pendingTasks = Get-ChildItem -Path $pendingDir -Filter "*.json" -ErrorAction SilentlyContinue
    Write-Log "  Pending tasks: $($pendingTasks.Count)"

    # TODO: Add your task processing logic here
    # Example: Run OpenHands for large tasks
    # foreach ($task in $pendingTasks) {
    #     $taskData = Get-Content $task.FullName | ConvertFrom-Json
    #     if ($taskData.priority -le 3) {
    #         # High priority - process with OpenHands
    #         Write-Log "  Processing: $($taskData.task_id)"
    #     }
    # }

    # Step 3: Generate reports
    Write-Log "Step 3: Generate Reports"
    # TODO: Add report generation logic
    # Example: Use local LLM to summarize logs
    # python scripts/summarize_logs.py

    # Step 4: Cleanup old files
    Write-Log "Step 4: Cleanup"
    $completedDir = Join-Path $WorkspaceDir ".multi-llm\tasks\completed"
    $oldTasks = Get-ChildItem -Path $completedDir -Filter "*.json" |
        Where-Object { $_.LastWriteTime -lt (Get-Date).AddDays(-30) }
    if ($oldTasks) {
        Write-Log "  Archiving $($oldTasks.Count) old tasks"
        $archiveDir = Join-Path $WorkspaceDir ".multi-llm\archive"
        New-Item -ItemType Directory -Path $archiveDir -Force | Out-Null
        $oldTasks | Move-Item -Destination $archiveDir
    }

    # Step 5: Update status
    Write-Log "Step 5: Update System Status"
    $statusFile = Join-Path $WorkspaceDir ".multi-llm\shared\status.json"
    $status = @{
        last_nightly = Get-Date -Format "yyyy-MM-ddTHH:mm:ssZ"
        status = "healthy"
    }
    $status | ConvertTo-Json | Set-Content -Path $statusFile

    Write-Log "=== Nightly Job Completed Successfully ==="
}
catch {
    Write-Log "ERROR: $($_.Exception.Message)" "ERROR"
    Write-Log "=== Nightly Job Failed ===" "ERROR"
    exit 1
}
