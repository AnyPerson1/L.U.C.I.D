#!/usr/bin/env python3
"""
JARVIS Dependency Installer
Automatically installs all required dependencies for Jarvis AI Assistant
"""

import subprocess
import sys
import os
import importlib.util

def check_virtual_env():
    """Check if virtual environment is activated"""
    if not hasattr(sys, 'real_prefix') and not (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix):
        print("❌ ERROR: Virtual environment is not activated!")
        print("Please activate your virtual environment first:")
        print("   venv\\Scripts\\activate  (Windows)")
        print("   source venv/bin/activate  (Linux/Mac)")
        print("Then run this script again.")
        return False
    
    print(f"✅ Virtual environment detected: {sys.prefix}")
    return True

def run_pip_command(command, description):
    """Run a pip command with error handling"""
    print(f"\n🔄 {description}...")
    try:
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        print(f"✅ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {description} failed:")
        print(f"Error: {e.stderr}")
        return False

def check_package_installed(package_name):
    """Check if a package is installed"""
    spec = importlib.util.find_spec(package_name)
    return spec is not None

def main():
    print("=" * 50)
    print("JARVIS DEPENDENCY INSTALLER")
    print("=" * 50)
    
    # Check virtual environment
    if not check_virtual_env():
        return False
    
    # Upgrade pip first
    if not run_pip_command("python -m pip install --upgrade pip", "Upgrading pip"):
        print("⚠️  Warning: Failed to upgrade pip, continuing anyway...")
    
    # Install PyTorch with CUDA 12.1 support
    pytorch_cmd = "pip install torch==2.5.1+cu121 torchaudio==2.5.1+cu121 --index-url https://download.pytorch.org/whl/cu121"
    if not run_pip_command(pytorch_cmd, "Installing PyTorch 2.5.1 with CUDA 12.1"):
        print("❌ Failed to install PyTorch. This is critical for Jarvis.")
        return False
    
    # Core dependencies
    core_deps = [
        "requests>=2.31.0",
        "numpy>=1.24.0", 
        "sounddevice>=0.4.6",
        "scipy>=1.11.0",
        "psutil>=5.9.0"
    ]
    
    print("\n🔄 Installing core dependencies...")
    for dep in core_deps:
        if not run_pip_command(f"pip install {dep}", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️  Warning: Failed to install {dep}")
    
    # TTS/STT dependencies
    tts_deps = [
        "TTS>=0.22.0",
        "faster-whisper>=1.0.0", 
        "pyttsx3>=2.90"
    ]
    
    print("\n🔄 Installing TTS and STT dependencies...")
    for dep in tts_deps:
        if not run_pip_command(f"pip install {dep}", f"Installing {dep.split('>=')[0]}"):
            print(f"⚠️  Warning: Failed to install {dep}")
    
    # Test PyTorch CUDA availability
    print("\n🧪 Testing PyTorch CUDA availability...")
    try:
        import torch
        print(f"✅ PyTorch version: {torch.__version__}")
        print(f"✅ CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"✅ CUDA version: {torch.version.cuda}")
            print(f"✅ GPU device: {torch.cuda.get_device_name(0)}")
        else:
            print("⚠️  CUDA not available - Jarvis will run on CPU (slower)")
    except ImportError:
        print("❌ Failed to import PyTorch")
        return False
    
    # Check critical dependencies
    print("\n🔍 Checking critical dependencies...")
    critical_deps = ['torch', 'requests', 'numpy', 'sounddevice', 'TTS', 'faster_whisper']
    all_good = True
    
    for dep in critical_deps:
        if check_package_installed(dep):
            print(f"✅ {dep} installed")
        else:
            print(f"❌ {dep} NOT installed")
            all_good = False
    
    print("\n" + "=" * 50)
    if all_good:
        print("🎉 INSTALLATION COMPLETE!")
        print("✅ All critical dependencies installed successfully!")
        print("\nYou can now run Jarvis with:")
        print("   python jarvis.py")
    else:
        print("⚠️  INSTALLATION COMPLETED WITH WARNINGS")
        print("Some dependencies failed to install. Jarvis may not work properly.")
    
    print("\n📝 Optional installations:")
    print("- For NVIDIA NeMo support: pip install nemo_toolkit[asr]")
    print("- For VITS support: Manual installation required")
    print("=" * 50)
    
    return all_good

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)