@echo off
echo ========================================
echo JARVIS DEPENDENCY INSTALLER
echo ========================================
echo.

./venv/Scripts/activate

REM Check if virtual environment is activated
if "%VIRTUAL_ENV%"=="" (
    echo ERROR: Virtual environment is not activated!
    echo Please activate your virtual environment first:
    echo   venv\Scripts\activate
    echo Then run this script again.
    pause
    exit /b 1
)

echo Virtual environment detected: %VIRTUAL_ENV%
echo.

echo Installing Python dependencies...
echo.

REM Upgrade pip first
echo [1/4] Upgrading pip...
python -m pip install --upgrade pip

REM Install PyTorch with CUDA 12.1 support
echo.
echo [2/4] Installing PyTorch 2.5.1 with CUDA 12.1...
pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121

REM Install core dependencies
echo.
echo [3/4] Installing core dependencies...
pip install requests>=2.31.0
pip install numpy>=1.24.0
pip install sounddevice>=0.4.6
pip install scipy>=1.11.0
pip install psutil>=5.9.0

REM Install TTS/STT dependencies
echo.
echo [4/4] Installing TTS and STT dependencies...
pip install TTS>=0.22.0
pip install faster-whisper>=1.0.0
pip install pyttsx3>=2.90

echo.
echo ========================================
echo INSTALLATION COMPLETE!
echo ========================================
echo.

REM Test PyTorch CUDA availability
echo Testing PyTorch CUDA availability...
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}'); print(f'CUDA version: {torch.version.cuda if torch.cuda.is_available() else \"N/A\"}')"

echo.
echo Optional installations:
echo - For NVIDIA NeMo support: pip install nemo_toolkit[asr]
echo - For VITS support: Manual installation required
echo.

echo All dependencies installed successfully!
echo You can now run Jarvis with: python jarvis.py
echo.
pause