@echo off
REM ========================================
REM Isaac Lab Installation Script for Windows
REM ========================================

echo.
echo ================================================================================
echo   Isaac Lab Installation for Windows
echo ================================================================================
echo.

REM Check if running as Administrator
net session >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: This script must be run as Administrator!
    echo Right-click this file and select "Run as administrator"
    pause
    exit /b 1
)

echo [INFO] Administrator privileges confirmed
echo.

REM Check if Git is installed
where git >nul 2>&1
if %errorLevel% neq 0 (
    echo ERROR: Git is not installed!
    echo.
    echo Please install Git first:
    echo   1. Go to: https://git-scm.com/download/win
    echo   2. Download and install
    echo   3. Re-run this script
    pause
    exit /b 1
)

echo [INFO] Git found: OK
echo.

REM Set installation directory
set INSTALL_DIR=C:\IsaacLab

REM Check if Isaac Lab already exists
if exist "%INSTALL_DIR%" (
    echo WARNING: Isaac Lab directory already exists at %INSTALL_DIR%
    echo.
    choice /C YN /M "Do you want to reinstall (this will DELETE existing installation)?"
    if errorlevel 2 goto :skip_clone
    if errorlevel 1 (
        echo [INFO] Removing existing installation...
        rmdir /s /q "%INSTALL_DIR%"
    )
)

echo.
echo [INFO] Cloning Isaac Lab repository...
echo [INFO] This may take 5-10 minutes depending on your internet speed...
echo.

cd C:\
git clone https://github.com/isaac-sim/IsaacLab.git

if %errorLevel% neq 0 (
    echo ERROR: Failed to clone Isaac Lab repository
    pause
    exit /b 1
)

:skip_clone
echo.
echo [INFO] Isaac Lab cloned successfully
echo.

REM Install Isaac Lab
echo [INFO] Installing Isaac Lab...
echo [INFO] This will take 5-10 minutes...
echo.

cd "%INSTALL_DIR%"
call isaaclab.bat --install

if %errorLevel% neq 0 (
    echo ERROR: Isaac Lab installation failed!
    pause
    exit /b 1
)

echo.
echo ================================================================================
echo   Isaac Lab Installation Complete!
echo ================================================================================
echo.
echo Isaac Lab installed at: %INSTALL_DIR%
echo.
echo Next steps:
echo   1. Install your snap-fit project:
echo      cd C:\snapfit_lab
echo      %INSTALL_DIR%\isaaclab.bat -p -m pip install -e source\snapfit_lab
echo.
echo   2. Run training:
echo      Double-click run_training_windows.bat
echo.
pause
