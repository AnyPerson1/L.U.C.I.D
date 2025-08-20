# Advanced Setup Guide - VITS, NeMo ve Ollama

Bu rehber, Jarvis için gelişmiş TTS/STT motorları ve lokal AI modellerinin kurulumunu açıklar.

## 🎯 NVIDIA NeMo ASR Kurulumu

### Sistem Gereksinimleri
- NVIDIA GPU (CUDA desteği)
- CUDA 11.8+ veya 12.x
- Python 3.8-3.10
- 8GB+ GPU RAM (önerilen)

### Kurulum Adımları

```bash
# 1. NeMo Toolkit kurulumu
pip install nemo_toolkit[asr]

# 2. Ek bağımlılıklar
pip install omegaconf hydra-core
pip install pytorch-lightning>=1.9.0

# 3. CUDA uyumluluğu için
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121
```

### Test Etme
```python
import nemo.collections.asr as nemo_asr
model = nemo_asr.models.EncDecCTCModel.from_pretrained("nvidia/stt_en_conformer_ctc_large")
print("NeMo ASR başarıyla yüklendi!")
```

## 🎵 VITS TTS Kurulumu

### Manuel Kurulum (Gelişmiş)

```bash
# 1. VITS repository'sini klonla
git clone https://github.com/jaywalnut310/vits.git
cd vits

# 2. Gerekli paketleri yükle
pip install torch torchaudio
pip install scipy matplotlib
pip install phonemizer
pip install librosa
pip install unidecode

# 3. Monotonic Alignment Search derle
cd monotonic_align
python setup.py build_ext --inplace
cd ..
```

### Pre-trained Model İndirme
```bash
# LJ Speech modeli (İngilizce)
wget https://drive.google.com/uc?id=1q86w74Ygw2hNzYP9cWkeClGT5X25PvBT -O vits_ljs_base.pth

# Model dosyalarını Jarvis klasörüne taşı
mkdir -p "C:\Users\LUCID\Desktop\Jarvis V1\vits_models"
move vits_ljs_base.pth "C:\Users\LUCID\Desktop\Jarvis V1\vits_models\"
```

### VITS Konfigürasyonu
```json
{
  "model_path": "./vits_models/vits_ljs_base.pth",
  "config_path": "./vits_models/config.json",
  "vocab_path": "./vits_models/vocab.txt"
}
```

## 🤖 Ollama Phi-4-Mini Kurulumu

### 1. Ollama Kurulumu
```bash
# Windows için Ollama indirin:
# https://ollama.ai/download/windows

# Kurulum sonrası terminal'de:
ollama --version
```

### 2. Phi-4-Mini Model İndirme
```bash
# Phi-4-Mini modelini çek (yaklaşık 2.4GB)
ollama pull phi3:mini

# Alternatif olarak daha büyük model:
ollama pull phi3:medium
```

### 3. Model Test Etme
```bash
# Modeli test et
ollama run phi3:mini "Hello, how are you?"
```

### 4. API Endpoint Test
```bash
# REST API test
curl http://localhost:11434/api/generate -d '{
  "model": "phi3:mini",
  "prompt": "Why is the sky blue?",
  "stream": false
}'
```

## ⚙️ Jarvis Konfigürasyonu

Jarvis'te bu motorları aktifleştirmek için `jarvis.py` dosyasındaki CONFIG bölümünü güncelleyin:

```python
CONFIG = {
    "TTS_ENGINE": "vits",  # "coqui", "pyttsx3", "mimic3", "vits"
    "STT_ENGINE": "nemo",  # "whisper", "nemo"
    "USE_OLLAMA": True,    # Ollama kullanımı
    "OLLAMA_MODEL": "phi3:mini",  # Ollama model adı
    "OLLAMA_URL": "http://localhost:11434",  # Ollama server URL
}
```

## 🔧 Sorun Giderme

### NeMo Sorunları
```bash
# CUDA uyumluluk sorunu
pip uninstall torch torchaudio
pip install torch==2.1.0+cu121 torchaudio==2.1.0+cu121 --index-url https://download.pytorch.org/whl/cu121

# Memory hatası
export CUDA_VISIBLE_DEVICES=0
```

### VITS Sorunları
```bash
# Phonemizer hatası (Windows)
# Espeak-ng yükleyin: https://github.com/espeak-ng/espeak-ng/releases

# Monotonic align hatası
cd monotonic_align
rm -rf build/
python setup.py clean --all
python setup.py build_ext --inplace
```

### Ollama Sorunları
```bash
# Service başlatma
ollama serve

# Port değiştirme
set OLLAMA_HOST=0.0.0.0:11435
ollama serve
```

## 📊 Performans Karşılaştırması

| Motor | Hız | Kalite | GPU Gereksinimi | Disk Alanı |
|-------|-----|--------|----------------|-------------|
| Whisper | Orta | Yüksek | Opsiyonel | 1-3GB |
| NeMo | Hızlı | Çok Yüksek | Gerekli | 500MB-2GB |
| Coqui TTS | Orta | İyi | Opsiyonel | 100-500MB |
| VITS | Hızlı | Çok Yüksek | Önerilen | 100-300MB |
| Ollama Phi-4 | Hızlı | Yüksek | Önerilen | 2.4GB |

## 🎯 Önerilen Konfigürasyonlar

### Yüksek Performans (GPU Gerekli)
```python
CONFIG = {
    "TTS_ENGINE": "vits",
    "STT_ENGINE": "nemo", 
    "USE_OLLAMA": True,
    "OLLAMA_MODEL": "phi3:mini"
}
```

### Orta Performans (GPU Opsiyonel)
```python
CONFIG = {
    "TTS_ENGINE": "coqui",
    "STT_ENGINE": "whisper",
    "USE_OLLAMA": True,
    "OLLAMA_MODEL": "phi3:mini"
}
```

### Düşük Kaynak (CPU Only)
```python
CONFIG = {
    "TTS_ENGINE": "pyttsx3",
    "STT_ENGINE": "whisper",
    "USE_OLLAMA": False  # Gemini kullan
}
```

## 📝 Notlar

- NeMo ve VITS GPU kullanımını önemli ölçüde artırır
- Ollama lokal çalıştığı için internet bağlantısı gerektirmez
- Phi-4-Mini modeli 2.4GB disk alanı kaplar
- Tüm motorlar aynı anda kullanılabilir (fallback sistemi ile)

Bu kurulumlar tamamlandıktan sonra Jarvis çok daha gelişmiş yeteneklere sahip olacak!