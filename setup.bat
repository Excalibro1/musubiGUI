@echo off
setlocal EnableExtensions EnableDelayedExpansion
title MusubiGUI Setup

echo ==========================================
echo MusubiGUI Setup
echo ==========================================
echo.

cd /d "%~dp0"

where git >nul 2>nul
if errorlevel 1 (
  echo ERROR: Git is not installed or not in PATH.
  echo Install Git from https://git-scm.com/download/win and run setup.bat again.
  pause
  exit /b 1
)

where py >nul 2>nul
if errorlevel 1 (
  where python >nul 2>nul
  if errorlevel 1 (
    echo ERROR: Python 3.10-3.12 is not installed or not in PATH.
    echo Install Python from https://www.python.org/downloads/windows/ and check "Add Python to PATH".
    pause
    exit /b 1
  )
  set "PYTHON=python"
) else (
  set "PYTHON=py -3.11"
)

if not exist "musubi-tuner" (
  echo Cloning kohya-ss/musubi-tuner...
  git clone https://github.com/kohya-ss/musubi-tuner.git musubi-tuner
  if errorlevel 1 (
    echo ERROR: Failed to clone musubi-tuner.
    pause
    exit /b 1
  )
) else (
  echo musubi-tuner folder already exists. Pulling latest changes...
  pushd musubi-tuner
  git pull
  popd
)

if not exist "musubi-tuner\.venv-sage\Scripts\python.exe" (
  echo Creating Python virtual environment: musubi-tuner\.venv-sage
  %PYTHON% -m venv "musubi-tuner\.venv-sage"
  if errorlevel 1 (
    echo ERROR: Failed to create virtual environment.
    echo If py -3.11 failed, install Python 3.11 or edit setup.bat to use your Python.
    pause
    exit /b 1
  )
) else (
  echo Virtual environment already exists.
)

set "VENV_PY=%~dp0musubi-tuner\.venv-sage\Scripts\python.exe"

echo.
echo Choose PyTorch CUDA build:
echo   1^) CUDA 12.8  ^(recommended for current NVIDIA drivers^)
echo   2^) CUDA 12.4
echo   3^) CPU only  ^(not recommended for training^)
echo.
set /p CUDA_CHOICE="Enter choice [1]: "
if "%CUDA_CHOICE%"=="" set "CUDA_CHOICE=1"

if "%CUDA_CHOICE%"=="2" (
  set "TORCH_INDEX=https://download.pytorch.org/whl/cu124"
) else if "%CUDA_CHOICE%"=="3" (
  set "TORCH_INDEX=https://download.pytorch.org/whl/cpu"
) else (
  set "TORCH_INDEX=https://download.pytorch.org/whl/cu128"
)

echo.
echo Upgrading pip/wheel/setuptools...
"%VENV_PY%" -m ensurepip --upgrade
"%VENV_PY%" -m pip install -U pip wheel
REM TensorBoard currently imports pkg_resources. setuptools 81+ may not provide it.
"%VENV_PY%" -m pip install "setuptools<81"
if errorlevel 1 (
  echo ERROR: Failed to prepare pip/setuptools.
  pause
  exit /b 1
)

echo.
echo Installing PyTorch from %TORCH_INDEX% ...
"%VENV_PY%" -m pip install torch torchvision --index-url %TORCH_INDEX%
if errorlevel 1 (
  echo ERROR: Failed to install PyTorch.
  pause
  exit /b 1
)

echo.
echo Installing musubi-tuner and GUI dependencies...
pushd musubi-tuner
"%VENV_PY%" -m pip install -e .
"%VENV_PY%" -m pip install tensorboard prompt-toolkit "setuptools<81"
popd
if errorlevel 1 (
  echo ERROR: Failed to install musubi-tuner dependencies.
  pause
  exit /b 1
)

echo.
echo Optional: install SageAttention?
echo This can improve speed but may require a compatible compiler/CUDA setup.
set /p INSTALL_SAGE="Install SageAttention now? [y/N]: "
if /I "%INSTALL_SAGE%"=="Y" (
  "%VENV_PY%" -m pip install sageattention
  if errorlevel 1 echo WARNING: SageAttention install failed. You can still use SDPA.
)

if not exist "configs" mkdir configs
if not exist "models" mkdir models
if not exist "train_images" mkdir train_images
if not exist "cache" mkdir cache
if not exist "output" mkdir output
if not exist "samples" mkdir samples
if not exist "logs" mkdir logs

echo.
echo Verifying install...
"%VENV_PY%" -c "import torch; print('Torch:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('GPU:', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'none')"
"%VENV_PY%" -c "import tkinter, PIL, accelerate, transformers; print('GUI dependencies OK')"
if errorlevel 1 (
  echo WARNING: Verification reported an issue. Check the messages above.
)

echo.
echo ==========================================
echo Setup complete.
echo Run gui.bat to start MusubiGUI.
echo ==========================================
pause
