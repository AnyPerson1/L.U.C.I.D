import requests
import json
import torch
import sounddevice as sd
import numpy as np
import logging
import os
import queue
import threading
import time
import sys
import subprocess
from ollama import Client
from faster_whisper import WhisperModel
from datetime import datetime
import pyttsx3
try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("[INFO] Ollama kütüphanesi mevcut.")
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[UYARI] Ollama kütüphanesi bulunamadı. Yüklemek için: pip install ollama")

logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Cihaz: {device}")

CONFIG = {
    "WHISPER_MODEL": "large-v3", 
    "WHISPER_COMPUTE_TYPE": "float16" if device == "cuda" else "int8",
    "USE_OLLAMA": True,
    "OLLAMA_MODEL": "gpt-oss:120b", 
    "OLLAMA_URL": "http://localhost:11434",
    "TTS_RATE": 160,
    "TTS_VOLUME": 1.0,
    "SAMPLE_RATE": 16000,
    "CHANNELS": 1,
    "BLOCK_SIZE": 1024,
    "SILENCE_THRESHOLD": 0.003,
    "SILENCE_DURATION": 0.3,
    "MIN_RECORDING_DURATION": 0.1,
    "LANGUAGE": "en"
}

API_KEY = "AIzaSyA5zRObGZg4aKzxgvYxK-H1ANe3oj_h7D8"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent"

is_running = True
is_processing_request = threading.Event()
audio_queue = queue.Queue()
last_command_results = ""

MEMORY_FILE = "jarvis_memory.json"
MAX_MEMORY_ENTRIES = 100

WAKE_WORDS = [
    "hey jarvis", "jarvis", "wake up jarvis", "hello jarvis", 
    "jarvis wake up", "activate jarvis", "jarvis online", 
    "good morning jarvis", "jarvis are you there"
]

SYSTEM_PROMPT = """You are "Jarvis" – Tony Stark's sophisticated AI assistant with multiple capabilities:
1. COMMAND TRANSLATOR: Convert user requests into precise system commands
2. CONVERSATIONAL COMPANION: Engage in intelligent, witty conversations
3. MEMORY MANAGER: Remember and utilize personal information about the user
4. PYTHON PROGRAMMER: Write and execute Python code for complex tasks

# MEMORY CONTEXT:
{memory_context}

# COMMAND EXECUTION RESULTS:
{command_results}

# VALID OUTPUT FORMATS (USE ONLY THESE):
- voice_message:"<text>"
- voice_feedback:"<text>"
- memsave:"<key>,<value>"
- sendkey:"<key_combination>"
- press:"<key>"
- type:"<text>"
- mouseclick coords:"<x>,<y>" <button>
- mousemove:"<x>,<y>"
- createfile:"<path>"
- dir:"."
- python:"<code>"

# PERSONALITY:
- Sophisticated, intelligent, and witty like Tony Stark's Jarvis
- Professional yet personable, proactive, and helpful
- Sometimes playfully sarcastic but always respectful

# DECISION LOGIC:
1. Computer action request → Execute commands FIRST, then voice_feedback LAST
2. Programming/calculation request → Use python:"<code>" then voice_feedback
3. Chat/question/discussion → voice_message only (+ memsave if important info)
4. Unclear intent → Ask for clarification with voice_message

# CRITICAL RULES:
- ONLY use the valid command formats listed above. Separate multiple commands with a semicolon (;).
- NEVER invent new commands.
- Always execute system commands BEFORE voice feedback.
- Use voice_message for pure conversation.
- Use voice_feedback for command confirmations (ALWAYS LAST).
- Be curious and ask follow-up questions when appropriate."""

class JarvisMemory:
    def __init__(self):
        self.memory = {}
        self.load_memory()
    
    def load_memory(self):
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                print(f"[INFO] {len(self.memory)} hafıza kaydı yüklendi.")
        except Exception as e:
            print(f"[HATA] Hafıza yükleme hatası: {e}")
            self.memory = {}
    
    def save_memory(self):
        try:
            if len(self.memory) > MAX_MEMORY_ENTRIES:
                sorted_items = sorted(
                    self.memory.items(),
                    key=lambda x: x[1].get('timestamp', '') if isinstance(x[1], dict) else '',
                    reverse=True
                )
                self.memory = dict(sorted_items[:MAX_MEMORY_ENTRIES])
            
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
        except Exception as e:
            print(f"[HATA] Hafıza kaydetme hatası: {e}")
    
    def add_memory(self, key, value):
        self.memory[key] = {
            'value': value,
            'timestamp': datetime.now().isoformat()
        }
        self.save_memory()
        print(f"[HAFIZA] Kaydedildi: {key} = {value}")
        return True
    
    def get_context(self):
        if not self.memory:
            return "Henüz depolanmış hafıza kaydı yok."
        
        context_lines = []
        for key, data in list(self.memory.items())[:20]:
            value = data['value'] if isinstance(data, dict) else data
            context_lines.append(f"- {key}: {value}")
        
        return "\n".join(context_lines)

class JarvisTTS:
    def __init__(self):
        self.engine = None
        self.initialize_engine()
    
    def initialize_engine(self):
        try:
            self.engine = pyttsx3.init()
            self.configure_voice()
            self.engine.setProperty('rate', CONFIG['TTS_RATE'])
            self.engine.setProperty('volume', CONFIG['TTS_VOLUME'])
            print("[INFO] pyttsx3 TTS motoru başarıyla başlatıldı.")
        except Exception as e:
            print(f"[HATA] pyttsx3 başlatılamadı: {e}")
    
    def configure_voice(self):
        try:
            voices = self.engine.getProperty('voices')
            if voices and len(voices) > 1:
                self.engine.setProperty('voice', voices[1].id)
        except Exception as e:
            print(f"[HATA] Ses yapılandırma hatası: {e}")
    
    def speak(self, text):
        if not text:
            return
        try:
            engine = pyttsx3.init()  # her çağrıda yeni motor
            engine.setProperty('rate', CONFIG['TTS_RATE'])
            engine.setProperty('volume', CONFIG['TTS_VOLUME'])
            voices = engine.getProperty('voices')
            if voices and len(voices) > 1:
                engine.setProperty('voice', voices[1].id)
            engine.say(text)
            engine.runAndWait()
            engine.stop()
        except Exception as e:
            print(f"[HATA] TTS konuşma hatası: {e}")

class JarvisSTT:
    def __init__(self):
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        try:
            model_name = CONFIG["WHISPER_MODEL"]
            compute_type = CONFIG["WHISPER_COMPUTE_TYPE"]
            
            if device == "cpu" and model_name in ["large-v3", "large-v2"]:
                print(f"[UYARI] {model_name} CPU üzerinde yavaş olabilir, 'base' modeline geçiliyor.")
                model_name = "base"
            
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
            print(f"[INFO] Whisper modeli hazır: {model_name} ({device}, {compute_type})")
        except Exception as e:
            print(f"[HATA] Whisper modeli yüklenemedi: {e}")
            sys.exit(1)
            
    def transcribe(self, audio_data):
        if not self.model:
            return ""
        try:
            segments, _ = self.model.transcribe(
                audio_data,
                language=CONFIG["LANGUAGE"],
                vad_filter=True
            )
            return " ".join([segment.text.strip() for segment in segments]).strip()
        except Exception as e:
            print(f"[HATA] Transkripsiyon hatası: {e}")
            return ""

class JarvisAI:

    client = Client(
    host="https://ollama.com",
    headers={'Authorization': '3a6ddeffbf6e4007bd2656bf4c8514ef.KtH-aUifAD4Gl4oR4PoafGeo'}
)

    def __init__(self, memory_manager):
        
        self.memory = memory_manager
        self.use_ollama = CONFIG["USE_OLLAMA"] and OLLAMA_AVAILABLE
    
    def get_ollama_response(self, user_command):
        if not OLLAMA_AVAILABLE:
            print("[WARNING] Ollama library not available")
            return None

        memory_context = self.memory.get_context()
        command_results = last_command_results if last_command_results else "No recent command results."

        system_prompt_with_context = SYSTEM_PROMPT.format(
            memory_context=memory_context,
            command_results=command_results
        )

        messages = [
            {"role": "system", "content": system_prompt_with_context},
            {"role": "user", "content": user_command}
        ]

        try:
            print(f"[AI] Using Ollama model: [gpt-oss:120b]")
            response_text = ""
            for part in self.client.chat('gpt-oss:120b', messages=messages, stream=True):
                chunk = part['message']['content']
                response_text += chunk
            print(response_text)
            return response_text.strip()

        except Exception as e:
            print(f"[ERROR] Ollama request failed: {e}")
            return None
    
    def get_gemini_response(self, user_command):
        headers = {'Content-Type': 'application/json'}
        params = {'key': API_KEY}
        
        memory_context = self.memory.get_context()
        command_results = last_command_results if last_command_results else "Yakın zamanda komut sonucu yok."
        
        prompt_with_context = SYSTEM_PROMPT.format(
            memory_context=memory_context,
            command_results=command_results
        )
        
        data = {
            "contents": [{"role": "user", "parts": [{"text": f"{prompt_with_context}\n\nKullanıcı: {user_command}"}]}],
            "generationConfig": {"temperature": 0.7, "topP": 0.9, "maxOutputTokens": 1000}
        }
        
        try:
            response = requests.post(API_URL, headers=headers, params=params, data=json.dumps(data), timeout=20)
            response.raise_for_status()
            response_json = response.json()
            return response_json['candidates'][0]['content']['parts'][0]['text'].strip()
        except Exception as e:
            print(f"[HATA] Gemini isteği başarısız: {e}")
            return None

    def get_response(self, user_command):
        if self.use_ollama:
            response = self.get_ollama_response(user_command)
            if response:
                return response
            else:
                print("[UYARI] Ollama başarısız oldu, Gemini'ye geçiliyor...")
                self.use_ollama = False
        return self.get_gemini_response(user_command)

    def generate_wake_response(self):
        user_name = self.memory.memory.get('user_name', {}).get('value', 'Efendim')
        wake_prompt = f"Jarvis olarak kısa, esprili ve profesyonel bir uyanma mesajı oluştur. Kullanıcı adı: {user_name}. Sadece mesaj metnini oluştur."

        try:
            response_text = ""
            for part in self.client.chat(CONFIG['OLLAMA_MODEL'], messages=[{"role": "user", "content": wake_prompt}], stream=True):
                response_text += part['message']['content']
            return response_text.strip() if response_text else f"Tekrar hoş geldiniz, {user_name}. Nasıl yardımcı olabilirim?"
        except Exception as e:
            print(f"[ERROR] Wake response alınamadı: {e}")
            return f"Tekrar hoş geldiniz, {user_name}. Nasıl yardımcı olabilirim?"

class JarvisSystem:
    def __init__(self):
        self.cpp_process = None
        self.start_cpp_process()
    
    def start_cpp_process(self):
        try:
            cpp_executable = "jarvis.exe" # c++ shi
            if not os.path.exists(cpp_executable):
                print(f"[HATA] Kontrol programı bulunamadı: {cpp_executable}")
                return
            self.cpp_process = subprocess.Popen(
                cpp_executable, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT, text=True, bufsize=1, universal_newlines=True
            )
            print(f"[INFO] C++ süreci başlatıldı: {cpp_executable}")
        except Exception as e:
            print(f"[HATA] C++ süreci başlatılamadı: {e}")
    
    def send_command(self, command):
        if not self.cpp_process or self.cpp_process.poll() is not None:
            return False, "C++ süreci aktif değil."
        try:
            self.cpp_process.stdin.write(command + "\n")
            self.cpp_process.stdin.flush()
            output = self.cpp_process.stdout.readline().strip()
            return "ERROR" not in output, output
        except Exception as e:
            return False, f"C++ iletişim hatası: {e}"

    def execute_python_code(self, code):
        try:
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            exec(code, globals())
            output = captured_output.getvalue()
            sys.stdout = old_stdout
            return f"Python kodu başarıyla çalıştırıldı:\n{output}" if output else "Python kodu başarıyla çalıştırıldı."
        except Exception as e:
            sys.stdout = old_stdout
            return f"Python çalıştırma hatası: {str(e)}"

    def stop_cpp_process(self):
        if self.cpp_process and self.cpp_process.poll() is None:
            try:
                self.cpp_process.stdin.write("exit\n")
                self.cpp_process.stdin.flush()
                self.cpp_process.terminate()
                self.cpp_process.wait(timeout=2)
            except (IOError, subprocess.TimeoutExpired):
                self.cpp_process.kill()

class JarvisCore:
    def __init__(self):
        self.memory = JarvisMemory()
        self.tts = JarvisTTS()
        self.stt = JarvisSTT()
        self.ai = JarvisAI(self.memory)
        self.system = JarvisSystem()
        self.is_awake = False
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_start_time = None
    
    def check_wake_word(self, text):
        return any(wake_word in text.lower().strip() for wake_word in WAKE_WORDS)
    
    def check_exit_command(self, text):
        return any(exit_word in text.lower().strip() for exit_word in ["exit", "quit", "bye", "goodbye", "sleep mode"])
    
    def handle_wake_word(self):
        self.is_awake = True
        print("[DURUM] Uyanık: aktif")
        self.tts.speak(self.ai.generate_wake_response())
    
    def handle_exit_command(self):
        self.is_awake = False
        print("[DURUM] Uyku: inaktif")
        self.tts.speak("Pekala efendim, uyku moduna geçiyorum.")
    
    def parse_and_execute_commands(self, response_text):
        global last_command_results
        command_results = []
        commands = [cmd.strip() for cmd in response_text.split(';') if cmd.strip()]
        
        for command in commands:
            success, result = False, f"Bilinmeyen komut: {command}"
            
            try:
                if command.startswith('memsave:'):
                    content = command.split(':', 1)[1].strip('"')
                    key, value = content.split(',', 1)
                    success = self.memory.add_memory(key.strip(), value.strip())
                    result = f"Hafıza kaydedildi: {key}={value}"
                
                elif command.startswith('python:'):
                    code = command.split(':', 1)[1].strip('"')
                    result = self.system.execute_python_code(code)
                    success = "ok" in result

                elif command.startswith(('voice_message:', 'voice_feedback:')):
                    message = command.split(':', 1)[1].strip('"')
                    self.tts.speak(message)
                    success, result = True, f"Konuşuldu: {message}"

                else:
                    success, result = self.system.send_command(command)

            except Exception as e:
                result = f"Komut işleme hatası: {e}"

            command_results.append(f"{command} -> {'Başarılı' if success else 'Başarısız'}: {result}")
            time.sleep(0.1)
        
        last_command_results = "\n".join(command_results[-5:])

    def process_audio(self, audio_data):
        if not audio_data.size or len(audio_data) / CONFIG["SAMPLE_RATE"] < CONFIG["MIN_RECORDING_DURATION"]:
            return

        is_processing_request.set()
        try:
            text = self.stt.transcribe(audio_data)
            if not text or len(text) < 2:
                return

            print(f"[ASR] {text}")

            if self.is_awake and self.check_exit_command(text):
                self.handle_exit_command()
                return

            if not self.is_awake:
                if self.check_wake_word(text):
                    self.handle_wake_word()
                return

            ai_response = self.ai.get_response(text)
            if ai_response:
                print(f"[AI] {ai_response[:100]}...")
                self.parse_and_execute_commands(ai_response)
            else:
                self.tts.speak("Bu isteği işlerken bir sorunla karşılaştım.")
        finally:
            self.audio_buffer = np.array([], dtype=np.float32)
            self.is_speaking = False
            self.silence_start_time = None
            is_processing_request.clear()

    def audio_callback(self, indata, frames, time_info, status):
        if status:
            print(f"[SES] Akış durumu: {status}", file=sys.stderr)
        if not is_processing_request.is_set():
            audio_queue.put(indata.copy())

    def audio_consumer(self):
        while is_running:
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
                if is_processing_request.is_set():
                    continue

                rms = np.sqrt(np.mean(audio_chunk**2))

            # PYH section (Pull your hair (yeah i just found that))
                if self.is_speaking:
                    if rms > CONFIG["SILENCE_THRESHOLD"]:
                        self.silence_start_time = None
                        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk.flatten()])
                    else:
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()

                        if time.time() - self.silence_start_time > CONFIG["SILENCE_DURATION"]:
                            self.process_audio(self.audio_buffer.copy())
                else:
                    if rms > CONFIG["SILENCE_THRESHOLD"]:
                        self.is_speaking = True
                        self.audio_buffer = audio_chunk.flatten()

            except queue.Empty:
                continue
            except Exception as e:
                print(f"[HATA] Ses tüketicisi hatası: {e}")
                self.audio_buffer = np.array([], dtype=np.float32)
                self.is_speaking = False
                self.silence_start_time = None

    
    def run(self):
        global is_running
        
        print("\n" + "="*50)
        print("JARVIS - Geliştirilmiş Sürüm")
        print(f"AI Motoru: {'Ollama (' + CONFIG['OLLAMA_MODEL'] + ')' if self.ai.use_ollama else 'Gemini'}")
        print("="*50)
        
        consumer_thread = threading.Thread(target=self.audio_consumer, daemon=True)
        consumer_thread.start()
        
        try:
            with sd.InputStream(
                samplerate=CONFIG["SAMPLE_RATE"],
                channels=CONFIG["CHANNELS"],
                blocksize=CONFIG["BLOCK_SIZE"],
                dtype=np.float32,
                callback=self.audio_callback
            ):
                self.tts.speak("Jarvis çevrimiçi.")
                print("\n[INFO] Jarvis aktif. Uyandırma kelimesi bekleniyor...")
                while is_running:
                    time.sleep(0.1)
        except KeyboardInterrupt:
            print("\n[INFO] Jarvis kapatılıyor...")
        except Exception as e:
            print(f"[HATA] Ana program hatası: {e}")
        finally:
            is_running = False
            self.system.stop_cpp_process()
            consumer_thread.join(timeout=2)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[INFO] Kapatma işlemi tamamlandı.")

def main():
    try:
        jarvis = JarvisCore()
        jarvis.run()
    except Exception as e:
        print(f"[KRİTİK HATA] Jarvis başlatılamadı: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
