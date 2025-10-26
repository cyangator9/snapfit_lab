# Snap-Fit RL Training - PowerShell Script for Windows
# =====================================================

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  Snap-Fit Assembly RL Training (Windows PowerShell)" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""

# Configuration
$ISAACLAB_PATH = "C:\IsaacLab"
$NUM_ENVS = 128
$DEVICE = "cuda:0"
$HEADLESS = $true

# Check if Isaac Lab exists
if (-not (Test-Path "$ISAACLAB_PATH\isaaclab.bat")) {
    Write-Host "[ERROR] Isaac Lab not found at: $ISAACLAB_PATH" -ForegroundColor Red
    Write-Host ""
    Write-Host "Please install Isaac Lab or update the ISAACLAB_PATH variable in this script." -ForegroundColor Yellow
    Write-Host "Edit this file and change line 9 to your Isaac Lab installation path." -ForegroundColor Yellow
    Write-Host ""
    Write-Host "Example: `$ISAACLAB_PATH = `"C:\YourPath\IsaacLab`"" -ForegroundColor Yellow
    Write-Host ""
    Read-Host "Press Enter to exit"
    exit 1
}

Write-Host "[INFO] Isaac Lab found: $ISAACLAB_PATH" -ForegroundColor Green
Write-Host "[INFO] Project directory: $PSScriptRoot" -ForegroundColor Green
Write-Host "[INFO] Number of environments: $NUM_ENVS" -ForegroundColor Green
Write-Host "[INFO] Device: $DEVICE" -ForegroundColor Green
Write-Host "[INFO] Headless mode: $HEADLESS" -ForegroundColor Green
Write-Host ""

# Build command
$args_list = @(
    "-p",
    "scripts\skrl\train.py",
    "--task=Template-Snapfit-Lab-Direct-v0",
    "--num_envs=$NUM_ENVS",
    "--device=$DEVICE"
)

if ($HEADLESS) {
    $args_list += "--headless"
}

# Run training
Write-Host "[INFO] Starting training..." -ForegroundColor Green
Write-Host ""

& "$ISAACLAB_PATH\isaaclab.bat" $args_list

Write-Host ""
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host "  Training completed or interrupted" -ForegroundColor Cyan
Write-Host "================================================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "Logs saved to: $PSScriptRoot\logs\metrics\" -ForegroundColor Yellow
Write-Host ""
Read-Host "Press Enter to exit"
