@echo off
REM ========================================
REM Snap-Fit RL Training - Windows Runner
REM ========================================

echo.
echo ================================================================================
echo   Starting Snap-Fit Assembly RL Training (Windows)
echo ================================================================================
echo.

REM Set Isaac Lab path (MODIFY THIS to your Isaac Lab installation)
set ISAACLAB_PATH=C:\IsaacLab

REM Check if Isaac Lab exists
if not exist "%ISAACLAB_PATH%\isaaclab.bat" (
    echo ERROR: Isaac Lab not found at %ISAACLAB_PATH%
    echo Please edit this script and set ISAACLAB_PATH to your Isaac Lab installation
    echo Example: set ISAACLAB_PATH=C:\IsaacLab
    pause
    exit /b 1
)

echo [INFO] Using Isaac Lab: %ISAACLAB_PATH%
echo [INFO] Project directory: %~dp0
echo.

REM Run training with Isaac Lab's Python environment (optimized for RTX 4050)
"%ISAACLAB_PATH%\isaaclab.bat" -p scripts\skrl\train.py ^
  --task=Template-Snapfit-Lab-Direct-v0 ^
  --num_envs=64 ^
  --device=cuda:0 ^
  --headless

echo.
echo ================================================================================
echo   Training completed or interrupted
echo ================================================================================
pause
