# Jarvis Dependencies Installation Guide

Bu rehber, Jarvis AI Assistant için gerekli tüm dependency'leri yüklemenize yardımcı olacaktır.

## Hızlı Kurulum

### Yöntem 1: Otomatik Kurulum (Önerilen)
```bash
# Virtual environment'ı aktifleştirin
venv\Scripts\activate

# Python installer'ı çalıştırın
python install_dependencies.py
```

### Yöntem 2: Batch Script (Windows)
```bash
# Virtual environment'ı aktifleştirin
venv\Scripts\activate

# Batch script'i çalıştırın
install_dependencies.bat
```

### Yöntem 3: Manuel Kurulum
```bash
# Virtual environment'ı aktifleştirin
venv\Scripts\activate

# Requirements dosyasından yükleyin
pip install -r requirements.txt
```

## Gerekli Dependencies

### Ana Bağımlılıklar
- **PyTorch 2.5.1+cu121**: CUDA 12.1 desteği ile derin öğrenme
- **TTS**: Coqui TTS kütüphanesi (metin-konuşma)
- **faster-whisper**: Hızlı konuşma tanıma
- **sounddevice**: Ses giriş/çıkış işlemleri
- **requests**: HTTP istekleri (Gemini API)
- **numpy**: Sayısal hesaplamalar
- **scipy**: Bilimsel hesaplamalar

### İsteğe Bağlı Dependencies
- **pyttsx3**: Alternatif TTS motoru
- **nemo_toolkit[asr]**: NVIDIA NeMo ASR (gelişmiş kullanıcılar için)
- **VITS**: Gelişmiş TTS modeli (manuel kurulum gerekli)

## Sistem Gereksinimleri

### Minimum Gereksinimler
- Python 3.8+
- 8GB RAM
- 2GB disk alanı
- Windows 10/11

### Önerilen Gereksinimler
- Python 3.9+
- 16GB RAM
- NVIDIA GPU (CUDA 12.1 desteği)
- 5GB disk alanı

## CUDA Kurulumu

Jarvis'in tam performansla çalışması için NVIDIA GPU ve CUDA gereklidir:

1. **NVIDIA GPU Driver**: En son sürümü yükleyin
2. **CUDA Toolkit 12.1**: [NVIDIA'dan indirin](https://developer.nvidia.com/cuda-12-1-0-download-archive)
3. **cuDNN**: CUDA ile uyumlu sürümü yükleyin

## Sorun Giderme

### PyTorch CUDA Sorunu
```bash
# CUDA versiyonunu kontrol edin
nvidia-smi

# PyTorch CUDA desteğini test edin
python -c "import torch; print(torch.cuda.is_available())"
```

### TTS Model İndirme Sorunu
```bash
# TTS cache'i temizleyin
python -c "from TTS.utils.manage import ModelManager; ModelManager().download_model('tts_models/en/ljspeech/tacotron2-DDC')"
```

### Ses Cihazı Sorunu
```bash
# Ses cihazlarını listeleyin
python -c "import sounddevice as sd; print(sd.query_devices())"
```

### Memory Hatası
- Daha küçük model kullanın (large-v2 yerine base)
- Batch size'ı azaltın
- GPU memory'yi temizleyin: `torch.cuda.empty_cache()`

## Performans Optimizasyonu

### GPU Kullanımı
- CUDA 12.1 ile PyTorch kullanın
- Mixed precision training aktifleştirin
- Model cache'i etkinleştirin

### CPU Kullanımı
- Daha küçük modeller seçin
- Thread sayısını optimize edin
- Quantized modeller kullanın

## Güncelleme

Dependencies'i güncellemek için:
```bash
pip install --upgrade -r requirements.txt
```

## Destek

Sorun yaşıyorsanız:
1. Virtual environment'ın aktif olduğundan emin olun
2. Python versiyonunu kontrol edin (3.8+)
3. CUDA kurulumunu doğrulayın
4. Log dosyalarını kontrol edin

## Lisanslar

Bu proje aşağıdaki açık kaynak kütüphaneleri kullanır:
- PyTorch (BSD License)
- Coqui TTS (MPL 2.0)
- Faster Whisper (MIT License)
- SoundDevice (MIT License)

Tüm dependency'ler kendi lisansları altında dağıtılır.