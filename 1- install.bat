@echo off
setlocal EnableExtensions
cd /d "%~dp0"
title MOSS-TTS Easy GUI - Fixed CUDA Installer

set "UV_VERSION=0.10.0"
set "UV_URL=https://github.com/astral-sh/uv/releases/download/%UV_VERSION%/uv-x86_64-pc-windows-msvc.zip"
set "UV_ZIP=%CD%\.runtime\temp\uv-%UV_VERSION%-windows-x64.zip"
set "UV_EXTRACT_DIR=%CD%\.runtime\temp\uv-%UV_VERSION%-extract"

set "PYTHON_VERSION=3.12.10"
set "TORCH_VERSION=2.9.0"
set "CUDA_RUNTIME=12.8"
set "TRITON_VERSION=3.5.1.post24"
set "FA_VERSION=2.8.3"

set "FA_WHEEL_NAME=flash_attn-2.8.3+cu128torch2.9.0cxx11abiTRUE-cp312-cp312-win_amd64.whl"
set "FA_WHEEL_URL=https://huggingface.co/Wildminder/AI-windows-whl/resolve/main/%FA_WHEEL_NAME%?download=true"

set "LLAMA_LIB_REPO=Blakus/llama-cpp-custom-libs"
set "LLAMA_BIN=%CD%\.runtime\llama-cpp\bin"

set "UV_DIR=%CD%\.runtime\uv"
set "UV_EXE=%UV_DIR%\uv.exe"
set "PY_EXE=%CD%\.venv\Scripts\python.exe"

set "UV_PYTHON_INSTALL_DIR=%CD%\.runtime\python"
set "UV_PROJECT_ENVIRONMENT=%CD%\.venv"
set "UV_CACHE_DIR=%CD%\.runtime\uv-cache"

set "HF_HOME=%CD%\.runtime\hf-cache"
set "HUGGINGFACE_HUB_CACHE=%HF_HOME%\hub"
set "HF_HUB_CACHE=%HF_HOME%\hub"
set "HF_XET_CACHE=%HF_HOME%\xet"

set "TMP=%CD%\.runtime\temp"
set "TEMP=%CD%\.runtime\temp"

set "UV_NO_CACHE=1"
set "UV_LINK_MODE=copy"
set "PIP_NO_CACHE_DIR=1"
set "PYTHONDONTWRITEBYTECODE=1"

set "TRITON_CACHE_DIR=%CD%\.runtime\triton-cache"
set "TORCHINDUCTOR_CACHE_DIR=%CD%\.runtime\torchinductor-cache"
set "TORCH_EXTENSIONS_DIR=%CD%\.runtime\torch-extensions"

for %%D in (
  ".runtime"
  ".runtime\temp"
  "%UV_DIR%"
  "models"
  "models\asr"
  "outputs"
  "voices"
  "training\datasets"
  "training\outputs"
  "accelerator_wheels"
) do if not exist "%%~D" mkdir "%%~D"

where nvidia-smi >nul 2>nul
if errorlevel 1 (
  echo [ERROR] NVIDIA driver / nvidia-smi was not detected.
  goto :fail
)

echo [1/10] NVIDIA GPU detected:
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader

rem ============================================================
rem 2/10 - uv
rem ============================================================

if exist "%UV_EXE%" (
  echo [2/10] Project-local uv already present.
) else (
  echo [2/10] Downloading project-local uv %UV_VERSION%...

  where curl.exe >nul 2>&1 || (
    echo [ERROR] curl.exe is required to bootstrap uv.
    goto :fail
  )

  where tar.exe >nul 2>&1 || (
    echo [ERROR] tar.exe is required to bootstrap uv.
    goto :fail
  )

  if exist "%UV_ZIP%" del /q "%UV_ZIP%" >nul 2>&1
  if exist "%UV_EXTRACT_DIR%" rmdir /s /q "%UV_EXTRACT_DIR%" >nul 2>&1
  mkdir "%UV_EXTRACT_DIR%" >nul 2>&1

  curl.exe -L --fail --retry 3 --retry-delay 2 ^
    -o "%UV_ZIP%" ^
    "%UV_URL%"
  if errorlevel 1 (
    echo [ERROR] Failed to download uv from:
    echo         %UV_URL%
    goto :fail
  )

  tar.exe -xf "%UV_ZIP%" -C "%UV_EXTRACT_DIR%"
  if errorlevel 1 (
    echo [ERROR] Failed to extract uv archive.
    goto :fail
  )

  for /r "%UV_EXTRACT_DIR%" %%F in (uv.exe) do (
    if exist "%%~fF" if not exist "%UV_EXE%" copy /y "%%~fF" "%UV_EXE%" >nul
  )

  if not exist "%UV_EXE%" (
    echo [ERROR] uv.exe was not found inside the downloaded archive.
    goto :fail
  )

  del /q "%UV_ZIP%" >nul 2>&1
  rmdir /s /q "%UV_EXTRACT_DIR%" >nul 2>&1
)

rem ============================================================
rem 3/10 - Python
rem ============================================================

if exist "%PY_EXE%" (
  for /f "delims=" %%V in ('"%PY_EXE%" -c "import sys; print('.'.join(map(str,sys.version_info[:3])))" 2^>nul') do set "CURRENT_PYTHON=%%V"
)

if /I "%CURRENT_PYTHON%"=="%PYTHON_VERSION%" (
  echo [3/10] Project-local Python %PYTHON_VERSION% already present.
) else (
  echo [3/10] Installing project-local Python %PYTHON_VERSION%...
  "%UV_EXE%" python install %PYTHON_VERSION% --no-cache --no-bin --no-registry
  if errorlevel 1 goto :fail
)

rem ============================================================
rem 4/10 - Main environment
rem Skip uv sync when the existing environment already satisfies
rem the frozen runtime used by the application.
rem ============================================================

set "MAIN_ENV_READY="
if exist "%PY_EXE%" (
  "%PY_EXE%" -c "import importlib.metadata as m, sentencepiece; assert m.version('torch').startswith('2.9.0'); assert m.version('torchaudio').startswith('2.9.0'); assert m.version('transformers')=='5.0.0'; assert m.version('gradio')=='6.11.0'; assert m.version('huggingface-hub')=='1.3.0'; assert m.version('hf-xet')=='1.6.0'; assert m.version('peft')=='0.18.1'; assert m.version('faster-whisper')=='1.2.1'; assert m.version('ctranslate2')=='4.8.1'; assert m.version('onnxruntime-gpu')=='1.28.0'" >nul 2>&1
  if not errorlevel 1 set "MAIN_ENV_READY=1"
)

if defined MAIN_ENV_READY (
  echo [4/10] Frozen MOSS-TTS environment already satisfied. Skipping uv sync.
) else (
  echo [4/10] Synchronizing frozen MOSS-TTS environment...
  "%UV_EXE%" sync --no-cache --python %PYTHON_VERSION%
  if errorlevel 1 goto :fail
)

if not exist "%PY_EXE%" goto :fail

rem ============================================================
rem 5/10 - Triton Windows
rem ============================================================

set "TRITON_READY="
"%PY_EXE%" -c "import importlib.metadata as m; assert m.version('triton-windows')=='%TRITON_VERSION%'" >nul 2>&1
if not errorlevel 1 set "TRITON_READY=1"

if defined TRITON_READY (
  echo [5/10] Triton Windows %TRITON_VERSION% already installed. Skipping.
) else (
  echo [5/10] Installing fixed Triton Windows runtime...
  "%UV_EXE%" pip install --python "%PY_EXE%" --no-cache "triton-windows==%TRITON_VERSION%"
  if errorlevel 1 goto :fail
)

rem ============================================================
rem 6/10 - FlashAttention
rem ============================================================

set "FLASH_READY="
"%PY_EXE%" -c "import importlib.metadata as m; assert m.version('flash-attn').split('+')[0]=='%FA_VERSION%'; import flash_attn" >nul 2>&1
if not errorlevel 1 set "FLASH_READY=1"

if defined FLASH_READY (
  echo [6/10] FlashAttention %FA_VERSION% already installed. Skipping.
) else (
  echo [6/10] Installing fixed FlashAttention Windows wheel...
  call :install_flash
  if errorlevel 1 goto :fail
)

rem ============================================================
rem 7/10 - Runtime verification
rem ============================================================

echo [7/10] Verifying frozen CUDA and accelerator runtime...
"%PY_EXE%" "%CD%\tools\verify_runtime.py"
if errorlevel 1 goto :fail

rem ============================================================
rem 8/10 - MOSS-SoundEffect v2.0
rem ============================================================

set "SFX_VENV=%CD%\.runtime\sfx-v2-venv"
set "SFX_PY=%SFX_VENV%\Scripts\python.exe"
set "SFX_READY="

if exist "%SFX_PY%" (
  "%SFX_PY%" -c "import importlib.metadata as m; assert m.version('moss-soundeffect-v2')=='0.1.0'; assert m.version('torch').startswith('2.9.0'); assert m.version('transformers')=='4.57.1'; assert m.version('numpy').startswith('1.26.4')" >nul 2>&1
  if not errorlevel 1 set "SFX_READY=1"
)

if defined SFX_READY (
  echo [8/10] MOSS-SoundEffect v2.0 runtime already installed. Skipping.
) else (
  echo [8/10] Installing isolated MOSS-SoundEffect v2.0 runtime...

  if not exist "%SFX_PY%" (
    "%UV_EXE%" venv "%SFX_VENV%" --python %PYTHON_VERSION%
    if errorlevel 1 goto :fail
  )

  "%UV_EXE%" pip install ^
    --python "%SFX_PY%" ^
    --no-cache ^
    --index-strategy unsafe-best-match ^
    --extra-index-url https://download.pytorch.org/whl/cu128 ^
    -e "%CD%\moss_tts_upstream\moss_soundeffect_v2[torch-cu128]"
  if errorlevel 1 goto :fail
)

rem ============================================================
rem 9/10 - Precompiled llama.cpp runtime
rem ============================================================

echo [9/10] Checking precompiled llama.cpp CUDA runtime...
call :install_llama_runtime
if errorlevel 1 goto :fail

rem ============================================================
rem 10/10 - Complete
rem ============================================================

echo [10/10] Installation complete.
echo.
echo ============================================================
echo  MOSS-TTS Easy GUI is ready
echo ============================================================
echo Python: %PYTHON_VERSION%
echo PyTorch: %TORCH_VERSION% + CUDA %CUDA_RUNTIME%
echo Triton Windows: %TRITON_VERSION%
echo FlashAttention: %FA_VERSION%
echo llama.cpp CUDA runtime: Hugging Face %LLAMA_LIB_REPO%
echo.
echo Run:
echo   2- run.bat
echo ============================================================
echo.
pause
exit /b 0


rem ============================================================
rem Subroutines
rem ============================================================

:install_llama_runtime
if not exist "%CD%\.runtime\llama-cpp" mkdir "%CD%\.runtime\llama-cpp" >nul 2>&1
if not exist "%LLAMA_BIN%" mkdir "%LLAMA_BIN%" >nul 2>&1

set "LLAMA_READY="
if exist "%LLAMA_BIN%\llama.dll" ^
if exist "%LLAMA_BIN%\ggml.dll" ^
if exist "%LLAMA_BIN%\ggml-base.dll" ^
if exist "%LLAMA_BIN%\ggml-cpu.dll" ^
if exist "%LLAMA_BIN%\ggml-cuda.dll" (
  set "LLAMA_CUBLAS="
  for /f "delims=" %%F in ('dir /b /a-d "%LLAMA_BIN%\cublas64_*.dll" 2^>nul') do if not defined LLAMA_CUBLAS set "LLAMA_CUBLAS=%%F"

  set "LLAMA_CUBLASLT="
  for /f "delims=" %%F in ('dir /b /a-d "%LLAMA_BIN%\cublasLt64_*.dll" 2^>nul') do if not defined LLAMA_CUBLASLT set "LLAMA_CUBLASLT=%%F"

  if defined LLAMA_CUBLAS if defined LLAMA_CUBLASLT set "LLAMA_READY=1"
)

if defined LLAMA_READY (
  echo [llama.cpp] Precompiled runtime already present. Skipping download.
) else (
  echo [llama.cpp] Downloading precompiled runtime from:
  echo             https://huggingface.co/%LLAMA_LIB_REPO%

  "%PY_EXE%" -c "from huggingface_hub import snapshot_download; snapshot_download(repo_id=r'%LLAMA_LIB_REPO%', repo_type='model', local_dir=r'%LLAMA_BIN%')"
  if errorlevel 1 (
    echo [ERROR] Failed to download the precompiled llama.cpp runtime.
    exit /b 1
  )
)

for %%F in (
  "llama.dll"
  "ggml.dll"
  "ggml-base.dll"
  "ggml-cpu.dll"
  "ggml-cuda.dll"
) do (
  if not exist "%LLAMA_BIN%\%%~F" (
    echo [ERROR] Required llama.cpp runtime file is missing: %%~F
    exit /b 1
  )
)

set "LLAMA_CUBLAS="
for /f "delims=" %%F in ('dir /b /a-d "%LLAMA_BIN%\cublas64_*.dll" 2^>nul') do if not defined LLAMA_CUBLAS set "LLAMA_CUBLAS=%%F"
if not defined LLAMA_CUBLAS (
  echo [ERROR] No cublas64_*.dll was found in the llama.cpp runtime.
  exit /b 1
)

set "LLAMA_CUBLASLT="
for /f "delims=" %%F in ('dir /b /a-d "%LLAMA_BIN%\cublasLt64_*.dll" 2^>nul') do if not defined LLAMA_CUBLASLT set "LLAMA_CUBLASLT=%%F"
if not defined LLAMA_CUBLASLT (
  echo [ERROR] No cublasLt64_*.dll was found in the llama.cpp runtime.
  exit /b 1
)

if not exist "%CD%\.runtime\llama-cpp\bridge\backbone_bridge.dll" (
  echo [ERROR] MOSS backbone bridge is missing:
  echo         .runtime\llama-cpp\bridge\backbone_bridge.dll
  exit /b 1
)

echo [llama.cpp] Runtime ready.
echo [llama.cpp] CUDA BLAS: %LLAMA_CUBLAS%
echo [llama.cpp] CUDA BLAS LT: %LLAMA_CUBLASLT%
exit /b 0


:install_flash
for %%F in ("accelerator_wheels\flash_attn*.whl") do if exist "%%~fF" (
  echo [FlashAttention] Using local wheel: %%~nxF
  "%UV_EXE%" pip install --python "%PY_EXE%" --no-cache --no-deps "%%~fF"
  exit /b %errorlevel%
)

"%UV_EXE%" pip install --python "%PY_EXE%" --no-cache "ninja==1.13.0" "packaging==25.0"
if errorlevel 1 exit /b 1

"%UV_EXE%" pip install --python "%PY_EXE%" --no-cache --no-deps "%FA_WHEEL_URL%"
exit /b %errorlevel%


:fail
if exist "%UV_CACHE_DIR%" rmdir /s /q "%UV_CACHE_DIR%"
echo.
echo INSTALLATION FAILED. Review the error above.
pause
exit /b 1
