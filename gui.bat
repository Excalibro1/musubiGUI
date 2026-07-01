@echo off
title Musubi GUI - Sage Env
echo Starting Musubi GUI using .venv-sage...
echo.
set "CUDA_PATH=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.6"
set "PATH=%CUDA_PATH%\bin;%~dp0musubi-tuner\.venv-sage\Lib\site-packages\torch\lib;%PATH%"
cd /d "%~dp0musubi-tuner"
"%~dp0musubi-tuner\.venv-sage\Scripts\python.exe" "%~dp0musubi_gui.py"
pause
