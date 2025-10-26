@echo off
REM ========================================
REM Snap-Fit RL Training - Windows Runner (WITH GUI)
REM ========================================

echo.
echo ================================================================================
echo   Starting Snap-Fit Assembly RL Training (Windows - WITH VISUALIZATION)
echo ================================================================================
echo.

REM Set Isaac Lab path (MODIFY THIS to your Isaac Lab installation)
set ISAACLAB_PATH=C:\IsaacLab

REM Check if Isaac Lab exists
if not exist "%ISAACLAB_PATH%\isaaclab.bat" (
    echo ERROR: Isaac Lab not found at %ISAACLAB_PATH%
    echo Please edit this script and set ISAACLAB_PATH to your Isaac Lab installation
    pause
    exit /b 1
)

echo [INFO] Using Isaac Lab: %ISAACLAB_PATH%
echo [INFO] Running with GUI (slower but you can see the robot!)
echo [INFO] Fewer environments for better visualization
echo.

REM Run training with visualization (fewer envs for performance)
"%ISAACLAB_PATH%\isaaclab.bat" -p scripts\skrl\train.py ^
  --task=Template-Snapfit-Lab-Direct-v0 ^
  --num_envs=16 ^
  --device=cuda:0

echo.
echo ================================================================================
echo   Training completed or interrupted
echo ================================================================================
pause
