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
from faster_whisper import WhisperModel
from datetime import datetime
import pyttsx3

# Ollama import with fallback
try:
    import ollama
    OLLAMA_AVAILABLE = True
    print("[INFO] Ollama library available")
except ImportError:
    OLLAMA_AVAILABLE = False
    print("[WARNING] Ollama library not available. Install with: pip install ollama")

# Logging configuration
logging.getLogger("requests").setLevel(logging.ERROR)
logging.getLogger("urllib3").setLevel(logging.ERROR)

# Device configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"[INFO] Device: {device}")

# Configuration
CONFIG = {
    "WHISPER_MODEL": "base",  # Optimize edilmiş model
    "WHISPER_COMPUTE_TYPE": "float16" if device == "cuda" else "int8",
    "USE_OLLAMA": True,  # Ollama'yı varsayılan olarak kullan
    "OLLAMA_MODEL": "phi4-mini",  # Varsayılan Ollama modeli
    "OLLAMA_URL": "http://localhost:11434",
    "TTS_RATE": 160,  # Jarvis tarzı konuşma hızı
    "TTS_VOLUME": 1.0,
    "SAMPLE_RATE": 16000,
    "CHANNELS": 1,
    "BLOCK_SIZE": 1024,
    "SILENCE_THRESHOLD": 0.003,
    "SILENCE_DURATION": 0.3,
    "MIN_RECORDING_DURATION": 0.1,
    "LANGUAGE": "en"
}

# API Configuration
API_KEY = "AIzaSyA5zRObGZg4aKzxgvYxK-H1ANe3oj_h7D8"
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-exp:generateContent"

# Global variables
is_running = True
is_processing_request = threading.Event()
audio_queue = queue.Queue()
cpp_process = None
is_awake = False
whisper_model = None
tts_engine = None
jarvis_memory = {}
last_command_results = ""

# Memory configuration
MEMORY_FILE = "jarvis_memory.json"
MAX_MEMORY_ENTRIES = 100

# Wake words
WAKE_WORDS = [
    "hey jarvis", "jarvis", "wake up jarvis", "hello jarvis", 
    "jarvis wake up", "activate jarvis", "jarvis online", 
    "good morning jarvis", "jarvis are you there"
]

# System prompts
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
- voice_message "<text>" - For conversations, questions, or when no command needed
- voice_feedback "<text>" - For command confirmations (brief, action-focused)
- memsave:"<key>,<value>" - Save important information to memory
- sendkey:"<key_combination>" - Key combinations (e.g., sendkey:"ctrl+c")
- press:"<key>" - Single key presses (e.g., press:"enter")
- type:"<text>" - Type text (e.g., type:"hello world")
- mouseclick coords:"<x>,<y>" <button> - Mouse clicks
- mousemove:"<x>,<y>" - Move mouse
- createfile:"<path>" - Create files
- dir:"." - List current directory
- python:"<code>" - Execute Python code

# PERSONALITY:
- Sophisticated, intelligent, and witty like Tony Stark's Jarvis
- Professional yet personable
- Proactive and helpful
- Sometimes playfully sarcastic but always respectful

# DECISION LOGIC:
1. Computer action request → Execute commands FIRST, then voice_feedback LAST
2. Programming/calculation request → Use python:"<code>" then voice_feedback
3. Chat/question/discussion → voice_message only (+ memsave if important info)
4. Unclear intent → Ask for clarification with voice_message

# CRITICAL RULES:
- ONLY use the valid command formats listed above
- NEVER invent new commands
- Always execute system commands BEFORE voice feedback
- Use voice_message for pure conversation
- Use voice_feedback for command confirmations (ALWAYS LAST)
- Be curious and ask follow-up questions when appropriate"""

class JarvisMemory:
    """Memory management class for Jarvis"""
    
    def __init__(self):
        self.memory = {}
        self.load_memory()
    
    def load_memory(self):
        """Load memory from file"""
        try:
            if os.path.exists(MEMORY_FILE):
                with open(MEMORY_FILE, 'r', encoding='utf-8') as f:
                    self.memory = json.load(f)
                print(f"[INFO] Loaded {len(self.memory)} memories")
            else:
                self.memory = {}
        except Exception as e:
            print(f"[ERROR] Memory loading error: {e}")
            self.memory = {}
    
    def save_memory(self):
        """Save memory to file"""
        try:
            if len(self.memory) > MAX_MEMORY_ENTRIES:
                # Keep most recent memories
                sorted_items = sorted(
                    self.memory.items(),
                    key=lambda x: x[1].get('timestamp', '') if isinstance(x[1], dict) else '',
                    reverse=True
                )
                self.memory = dict(sorted_items[:MAX_MEMORY_ENTRIES])
            
            with open(MEMORY_FILE, 'w', encoding='utf-8') as f:
                json.dump(self.memory, f, indent=2, ensure_ascii=False)
            
            print(f"[DEBUG] Saved {len(self.memory)} memories")
        except Exception as e:
            print(f"[ERROR] Memory saving error: {e}")
    
    def add_memory(self, key, value):
        """Add new memory entry"""
        self.memory[key] = {
            'value': value,
            'timestamp': datetime.now().isoformat(),
            'access_count': 0
        }
        self.save_memory()
        print(f"[MEMORY] Saved: {key} = {value}")
        return True
    
    def get_context(self):
        """Get memory context for AI"""
        if not self.memory:
            return "No stored memories yet."
        
        context_lines = []
        for key, data in list(self.memory.items())[:20]:  # Limit context size
            value = data['value'] if isinstance(data, dict) else data
            context_lines.append(f"- {key}: {value}")
            
            # Update access count
            if isinstance(data, dict):
                self.memory[key]['access_count'] = data.get('access_count', 0) + 1
        
        return "\n".join(context_lines)

class JarvisTTS:
    """Optimized TTS class using only pyttsx3"""
    
    def __init__(self):
        self.engine = None
        self.initialize_engine()
    
    def initialize_engine(self):
        """Initialize pyttsx3 engine with optimized settings"""
        try:
            self.engine = pyttsx3.init()
            self.configure_voice()
            self.configure_speech_properties()
            print("[INFO] pyttsx3 TTS engine initialized successfully")
            return True
        except Exception as e:
            print(f"[ERROR] pyttsx3 initialization failed: {e}")
            return False
    
    def configure_voice(self):
        """Configure voice selection (prefer male English voice)"""
        try:
            voices = self.engine.getProperty('voices')
            if not voices:
                print("[WARNING] No voices found")
                return
            
            selected_voice = None
            
            # Priority 1: Male English voices
            for voice in voices:
                voice_name = voice.name.lower()
                voice_id = voice.id.lower()
                
                if any(male_indicator in voice_name for male_indicator in ['david', 'mark', 'male']):
                    if any(lang in voice_name or lang in voice_id for lang in ['english', 'en', 'us', 'uk']):
                        selected_voice = voice
                        break
            
            # Priority 2: Any English voice
            if not selected_voice:
                for voice in voices:
                    voice_name = voice.name.lower()
                    voice_id = voice.id.lower()
                    if any(lang in voice_name or lang in voice_id for lang in ['english', 'en', 'us', 'uk']):
                        selected_voice = voice
                        break
            
            # Set selected voice
            if selected_voice:
                self.engine.setProperty('voice', selected_voice.id)
                print(f"[INFO] Selected voice: {selected_voice.name}")
            else:
                print("[WARNING] No suitable English voice found, using default")
                
        except Exception as e:
            print(f"[ERROR] Voice configuration error: {e}")
    
    def configure_speech_properties(self):
        """Configure speech rate and volume"""
        try:
            self.engine.setProperty('rate', CONFIG['TTS_RATE'])
            self.engine.setProperty('volume', CONFIG['TTS_VOLUME'])
            print(f"[INFO] TTS configured - Rate: {CONFIG['TTS_RATE']}, Volume: {CONFIG['TTS_VOLUME']}")
        except Exception as e:
            print(f"[ERROR] Speech properties configuration error: {e}")
    
    def speak(self, text):
        try:
            self.engine.say(text)
            self.engine.runAndWait()
        except Exception as e:
            print(f"[ERROR] {e}")
        finally:
        # motoru yeniden başlat
            self.engine.stop()
            self.engine = pyttsx3.init()
            self.engine.setProperty('rate', self.engine.getProperty('rate') - 30)
            self.engine.setProperty('volume', 1)
            self.engine.setProperty('voice', self.engine.getProperty('voices')[0].id)

class JarvisSTT:
    """Speech-to-text class using Whisper"""
    
    def __init__(self):
        self.model = None
        self.initialize_model()
    
    def initialize_model(self):
        """Initialize Whisper model"""
        try:
            model_name = CONFIG["WHISPER_MODEL"]
            compute_type = CONFIG["WHISPER_COMPUTE_TYPE"]
            
            if device == "cpu" and model_name in ["large-v3", "large-v2"]:
                print(f"[WARNING] {model_name} may be slow on CPU, using 'base' instead")
                model_name = "base"
            
            self.model = WhisperModel(model_name, device=device, compute_type=compute_type)
            print(f"[INFO] Whisper model ready: {model_name} ({device}, {compute_type})")
            return True
        except Exception as e:
            print(f"[ERROR] Whisper model failed to load: {e}")
            # Try fallback to base model
            try:
                self.model = WhisperModel("base", device=device, compute_type="int8")
                print("[INFO] Whisper fallback model ready: base")
                return True
            except Exception as e2:
                print(f"[ERROR] Whisper fallback failed: {e2}")
                return False
    
    def transcribe(self, audio_data):
        """Transcribe audio data to text"""
        if not self.model:
            print("[ERROR] Whisper model not initialized")
            return ""
        
        try:
            segments, _ = self.model.transcribe(
                audio_data,
                language=CONFIG["LANGUAGE"],
                vad_filter=True,
                vad_parameters=dict(min_silence_duration_ms=300),
                beam_size=3,
                best_of=3,
                temperature=0.0,
            )
            
            text = " ".join([segment.text.strip() for segment in segments])
            return text.strip()
        except Exception as e:
            print(f"[ERROR] Transcription error: {e}")
            return ""

class JarvisAI:
    """AI response generation class - supports both Ollama and Gemini"""
    
    def __init__(self, memory_manager):
        self.memory = memory_manager
        self.use_ollama = CONFIG["USE_OLLAMA"] and OLLAMA_AVAILABLE
    
    def get_ollama_response(self, user_command):
        """Get AI response using Ollama"""
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
            print(f"[AI] Using Ollama model: {CONFIG['OLLAMA_MODEL']}")
            response = ollama.chat(
                model=CONFIG['OLLAMA_MODEL'],
                messages=messages,
            )
            
            return response['message']['content'].strip() if response and 'message' in response else None
            
        except Exception as e:
            print(f"[ERROR] Ollama request failed: {e}")
            return None
    
    def get_gemini_response(self, user_command):
        """Get AI response using Gemini API"""
        headers = {'Content-Type': 'application/json'}
        params = {'key': API_KEY}
        
        memory_context = self.memory.get_context()
        command_results = last_command_results if last_command_results else "No recent command results."
        
        prompt_with_context = SYSTEM_PROMPT.format(
            memory_context=memory_context,
            command_results=command_results
        )
        
        data = {
            "contents": [{"role": "user", "parts": [{"text": f"{prompt_with_context}\n\nUser: {user_command}"}]}],
            "generationConfig": {
                "temperature": 0.7,
                "topP": 0.9,
                "maxOutputTokens": 1000
            }
        }
        
        try:
            print("[AI] Using Gemini API")
            response = requests.post(API_URL, headers=headers, params=params, data=json.dumps(data), timeout=20)
            response.raise_for_status()
            response_json = response.json()
            return response_json['candidates'][0]['content']['parts'][0]['text'].strip()
        except requests.exceptions.RequestException as e:
            print(f"[ERROR] Gemini request failed: {e}")
            return None
        except (KeyError, IndexError) as e:
            print(f"[ERROR] Gemini response parsing failed: {e}")
            return None
    
    def get_response(self, user_command):
        """Get AI response - try Ollama first, then fallback to Gemini"""
        if self.use_ollama:
            response = self.get_ollama_response(user_command)
            if response:
                return response
            
            print("[WARNING] Ollama failed, falling back to Gemini...")
            self.use_ollama = False  # Disable Ollama for this session
        
        return self.get_gemini_response(user_command)
    
    def generate_wake_response(self):
        """Generate wake up response"""
        user_name = ""
        if 'user_name' in self.memory.memory:
            user_name_data = self.memory.memory['user_name']
            user_name = user_name_data['value'] if isinstance(user_name_data, dict) else user_name_data
        
        # Use AI to generate dynamic wake response
        if self.use_ollama or True:  # Allow wake responses even without Ollama
            wake_prompt = f"""Generate a brief, witty, and professional wake-up response for Jarvis (Tony Stark's AI assistant). 
User name: {user_name if user_name else 'Sir'}
Keep it 1-2 sentences max. Be sophisticated and ready to assist.
Generate ONLY the wake-up message text, no quotes or extra formatting:"""
            
            if self.use_ollama:
                try:
                    response = ollama.generate(
                        model=CONFIG['OLLAMA_MODEL'],
                        prompt=wake_prompt
                    )
                    if response and 'response' in response:
                        return response['response'].strip()
                except Exception as e:
                    print(f"[ERROR] Ollama wake response failed: {e}")
        
        # Fallback to static response
        if user_name:
            return f"Good to see you again, {user_name}. How may I assist you today?"
        else:
            return "Good to see you again, Sir. How may I assist you today?"

class JarvisSystem:
    """System command execution class"""
    
    def __init__(self):
        self.cpp_process = None
        self.start_cpp_process()
    
    def start_cpp_process(self):
        """Start C++ control process"""
        try:
            cwd = os.getcwd()
            candidates = [
                os.path.join(cwd, "jarvis_control", "jarvis_control", "jarvis.exe"),
                os.path.join(cwd, "jarvis.exe"),
                os.path.join(cwd, "jarvis_control.exe"),
                "jarvis.exe",
                "jarvis_control.exe",
            ]
            
            cpp_executable = None
            for candidate in candidates:
                if os.path.isfile(candidate):
                    cpp_executable = candidate
                    break
            
            if cpp_executable is None:
                cpp_executable = "jarvis.exe"
            
            self.cpp_process = subprocess.Popen(
                cpp_executable,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
                universal_newlines=True
            )
            print(f"[INFO] C++ process started: {cpp_executable}")
            return True
        except Exception as e:
            print(f"[ERROR] C++ process failed to start: {e}")
            return False
    
    def send_command(self, command):
        """Send command to C++ process"""
        if self.cpp_process is None:
            return False, "C++ process not active"
        
        try:
            self.cpp_process.stdin.write(command + "\n")
            self.cpp_process.stdin.flush()
            
            output_lines = []
            while True:
                line = self.cpp_process.stdout.readline()
                if line == '':
                    return False, "C++ process terminated unexpectedly"
                
                stripped = line.strip()
                if stripped == "":
                    continue
                if stripped == "OK":
                    result = "\n".join(output_lines) if output_lines else "Command executed successfully"
                    return True, result
                if stripped.startswith("ERROR"):
                    return False, stripped
                
                output_lines.append(stripped)
        except Exception as e:
            return False, f"C++ communication error: {e}"
    
    def execute_python_code(self, code):
        """Execute Python code safely"""
        try:
            safe_globals = {
                '__builtins__': {
                    'print': print, 'len': len, 'str': str, 'int': int, 'float': float,
                    'list': list, 'dict': dict, 'tuple': tuple, 'set': set, 'range': range,
                    'enumerate': enumerate, 'zip': zip, 'sum': sum, 'max': max, 'min': min,
                    'abs': abs, 'round': round, 'sorted': sorted, 'reversed': reversed,
                    'any': any, 'all': all, 'type': type, 'isinstance': isinstance,
                },
                'os': os, 'sys': sys, 'time': time, 'datetime': datetime, 
                'json': json, 'requests': requests, 'np': np,
            }
            
            from io import StringIO
            old_stdout = sys.stdout
            sys.stdout = captured_output = StringIO()
            
            exec(code, safe_globals)
            output = captured_output.getvalue()
            sys.stdout = old_stdout
            
            return f"✅ Python execution successful:\n{output}" if output else "✅ Python code executed successfully"
        except Exception as e:
            sys.stdout = old_stdout
            return f"❌ Python execution error: {str(e)}"
    
    def stop_cpp_process(self):
        """Stop C++ process"""
        if self.cpp_process:
            try:
                self.cpp_process.stdin.write("exit\n")
                self.cpp_process.stdin.flush()
                self.cpp_process.wait(timeout=3)
            except:
                self.cpp_process.terminate()
            finally:
                self.cpp_process = None

class JarvisCore:
    """Main Jarvis application class"""
    
    def __init__(self):
        self.memory = JarvisMemory()
        self.tts = JarvisTTS()
        self.stt = JarvisSTT()
        self.ai = JarvisAI(self.memory)
        self.system = JarvisSystem()
        self.is_awake = False
        
        # Audio processing variables
        self.audio_buffer = np.array([], dtype=np.float32)
        self.is_speaking = False
        self.silence_start_time = None
    
    def check_wake_word(self, text):
        """Check if text contains wake word"""
        text_lower = text.lower().strip()
        return any(wake_word.lower() in text_lower for wake_word in WAKE_WORDS)
    
    def check_exit_command(self, text):
        """Check if text contains exit command"""
        text_lower = text.lower().strip()
        exit_words = ["exit", "quit", "bye", "goodbye", "see you", "sleep mode"]
        return any(exit_word in text_lower for exit_word in exit_words)
    
    def handle_wake_word(self):
        """Handle wake word activation"""
        self.is_awake = True
        print("[STATE] WAKE: active")
        wake_message = self.ai.generate_wake_response()
        self.tts.speak(wake_message)
    
    def handle_exit_command(self):
        """Handle exit command"""
        self.is_awake = False
        print("[STATE] SLEEP: inactive")
        self.tts.speak("Very well sir, entering sleep mode. Use a wake word to reactivate me.")
    
    def parse_and_execute_commands(self, commands):
        """Parse and execute AI commands"""
        global last_command_results
        
        command_list = [cmd.strip() for cmd in commands.split(';') if cmd.strip()]
        valid_commands = ['voice_message', 'voice_feedback', 'memsave:', 'sendkey:', 'press:', 
                         'type:', 'mouseclick', 'mousemove:', 'createfile:', 'dir:', 'python:']
        
        filtered_commands = [cmd for cmd in command_list if any(cmd.startswith(vcmd) for vcmd in valid_commands)]
        
        # Organize commands by type
        voice_commands = []
        system_commands = []
        memory_commands = []
        python_commands = []
        
        for command in filtered_commands:
            if command.startswith(('voice_message', 'voice_feedback')):
                voice_commands.append(command)
            elif command.startswith('memsave:'):
                memory_commands.append(command)
            elif command.startswith('python:'):
                python_commands.append(command)
            else:
                system_commands.append(command)
        
        # Execute commands in order: memory, python, system, then voice
        all_commands = memory_commands + python_commands + system_commands + voice_commands
        command_results = []
        
        for command in all_commands:
            success = False
            result = ""
            
            if command.startswith('memsave:'):
                try:
                    content = command[8:].strip().strip('"')
                    if ',' in content:
                        key, value = content.split(',', 1)
                        key = key.strip()
                        value = value.strip()
                        success = self.memory.add_memory(key, value)
                        result = f"Memory saved: {key} = {value}" if success else f"Memory not saved"
                    else:
                        result = f"Invalid memsave format: {command}"
                except Exception as e:
                    result = f"Memory save error: {e}"
            
            elif command.startswith('python:'):
                try:
                    code = command[7:].strip().strip('"')
                    result = self.system.execute_python_code(code)
                    success = "successful" in result
                except Exception as e:
                    result = f"Python execution error: {e}"
            
            elif command.startswith(('voice_message', 'voice_feedback')):
                try:
                    message = command.split('"')[1]
                    self.tts.speak(message)
                    success = True
                    result = f"Spoke: {message}"
                except IndexError:
                    self.tts.speak("I encountered an error processing that message.")
                    result = f"Malformed voice command: {command}"
            
            elif command.startswith('type:'):
                try:
                    content = command[5:].strip().strip('"')
                    # Preserve original casing and characters
                    normalized_command = f'type:"{content}"'
                    print(f"[CMD] {normalized_command}")
                    success, result = self.system.send_command(normalized_command)
                    if not success:
                        self.tts.speak("Command execution failed.")
                except Exception as e:
                    result = f"Type command processing error: {e}"
            
            elif any(command.startswith(cmd) for cmd in ['sendkey:', 'press:', 'mouseclick', 'mousemove:', 'createfile:', 'dir:']):
                print(f"[CMD] {command}")
                success, result = self.system.send_command(command)
                if not success:
                    self.tts.speak("Command execution failed.")
            
            else:
                result = f"Unknown command: {command}"
            
            if result:
                command_results.append(f"{command} -> {result}")
            time.sleep(0.1)
        
        last_command_results = "\n".join(command_results[-5:])
    
    def process_audio(self, audio_data):
        """Process audio data and generate response"""
        if not audio_data.size:
            return
        
        # Transcribe audio
        text = self.stt.transcribe(audio_data)
        if not text or len(text) < 2:
            return
        
        print(f"[ASR] {text}")
        
        # Handle exit commands
        if self.is_awake and self.check_exit_command(text):
            self.handle_exit_command()
            return
        
        # Handle wake words
        if not self.is_awake and self.check_wake_word(text):
            self.handle_wake_word()
            return
        
        # If not awake, ignore other commands
        if not self.is_awake:
            return
        
        # Get AI response
        ai_response = self.ai.get_response(text.strip())
        
        if ai_response:
            print(f"[AI] {ai_response[:100]}...")
            self.parse_and_execute_commands(ai_response)
        else:
            self.tts.speak("I'm having trouble processing that request right now.")
    
    def audio_callback(self, indata, frames, time_info, status):
        """Audio input callback"""
        if status:
            print(f"[AUDIO] Stream status: {status}")
        if not is_processing_request.is_set():
            audio_queue.put(indata.copy())
    
    def audio_consumer(self):
        """Audio processing consumer thread"""
        print("[INFO] Audio consumer started")
        
        while is_running:
            try:
                audio_chunk = audio_queue.get(timeout=0.1)
                rms = np.sqrt(np.mean(audio_chunk**2))
                
                # Skip processing if TTS is currently speaking
                if is_processing_request.is_set():
                    continue
                
                if self.is_speaking:
                    if rms > CONFIG["SILENCE_THRESHOLD"]:
                        self.silence_start_time = None
                        self.audio_buffer = np.concatenate([self.audio_buffer, audio_chunk.flatten()])
                    else:
                        if self.silence_start_time is None:
                            self.silence_start_time = time.time()
                        
                        if time.time() - self.silence_start_time > CONFIG["SILENCE_DURATION"]:
                            if len(self.audio_buffer) / CONFIG["SAMPLE_RATE"] > CONFIG["MIN_RECORDING_DURATION"]:
                                print("[DEBUG] Processing audio buffer...")
                                self.process_audio(self.audio_buffer.copy())
                            
                            # Reset state
                            self.audio_buffer = np.array([], dtype=np.float32)
                            self.is_speaking = False
                            self.silence_start_time = None
                            
                            # Clear queue to prevent buildup
                            while not audio_queue.empty():
                                try:
                                    audio_queue.get_nowait()
                                except queue.Empty:
                                    break
                
                elif rms > CONFIG["SILENCE_THRESHOLD"]:
                    self.is_speaking = True
                    self.audio_buffer = audio_chunk.flatten()
                    print("[DEBUG] Started recording audio...")
                    
            except queue.Empty:
                continue
            except Exception as e:
                print(f"[ERROR] Audio consumer error: {e}")
                # Reset state on error
                self.audio_buffer = np.array([], dtype=np.float32)
                self.is_speaking = False
                self.silence_start_time = None
                continue
    
    def check_apis(self):
        """Check both Ollama and Gemini API connections"""
        print("[INFO] Checking API connections...")
        
        ollama_ok = False
        gemini_ok = False
        
        # Check Ollama if enabled
        if CONFIG["USE_OLLAMA"] and OLLAMA_AVAILABLE:
            try:
                print(f"[INFO] Testing Ollama connection with model: {CONFIG['OLLAMA_MODEL']}")
                result = ollama.generate(
                    model=CONFIG['OLLAMA_MODEL'],
                    prompt="Say hello briefly"
                )
                if result and 'response' in result:
                    print("[INFO] ✅ Ollama API OK")
                    print(f"[DEBUG] Ollama response: {result['response'][:50]}...")
                    ollama_ok = True
                else:
                    print("[WARNING] ❌ Ollama API returned empty response")
            except Exception as e:
                print(f"[WARNING] ❌ Ollama API check failed: {e}")
                print("[INFO] Will use Gemini as fallback...")
        
        # Check Gemini API
        if not API_KEY or "AIzaSy" not in API_KEY:
            print("[ERROR] ❌ Invalid Gemini API key")
        else:
            headers = {'Content-Type': 'application/json'}
            params = {'key': API_KEY}
            data = {"contents": [{"role": "user", "parts": [{"text": "Say hello"}]}]}
            
            try:
                response = requests.post(API_URL, headers=headers, params=params, 
                                       data=json.dumps(data), timeout=10)
                response.raise_for_status()
                response_json = response.json()
                result = response_json['candidates'][0]['content']['parts'][0]['text'].strip()
                
                if result:
                    print("[INFO] ✅ Gemini API OK")
                    gemini_ok = True
                else:
                    print("[ERROR] ❌ Gemini API empty response")
            except Exception as e:
                print(f"[ERROR] ❌ Gemini API check failed: {e}")
        
        # Determine which API to use
        if ollama_ok:
            self.ai.use_ollama = True
            print("[INFO] 🚀 Primary AI: Ollama")
            if gemini_ok:
                print("[INFO] 🔄 Backup AI: Gemini")
            return True
        elif gemini_ok:
            self.ai.use_ollama = False
            print("[INFO] 🚀 Primary AI: Gemini")
            return True
        else:
            print("[ERROR] ❌ No working AI API found!")
            return False
    
    def run(self):
        """Main application loop"""
        global is_running
        
        print("\n" + "="*50)
        print("JARVIS - OPTIMIZED PYTTSX3 + OLLAMA VERSION")
        print("="*50)
        print(f"TTS Engine: pyttsx3 (Rate: {CONFIG['TTS_RATE']})")
        print(f"STT Engine: Whisper ({CONFIG['WHISPER_MODEL']})")
        print(f"AI Engine: {'Ollama (' + CONFIG['OLLAMA_MODEL'] + ')' if CONFIG['USE_OLLAMA'] and OLLAMA_AVAILABLE else 'Gemini'} + Fallback")
        print(f"Device: {device}")
        if CONFIG['USE_OLLAMA']:
            print(f"Ollama URL: {CONFIG['OLLAMA_URL']}")
        print("="*50)
        
        # Check API connections
        if not self.check_apis():
            print("[ERROR] No working AI API found. Please check your configuration.")
            return
        
        try:
            # Start audio consumer thread
            consumer_thread = threading.Thread(target=self.audio_consumer, daemon=True)
            consumer_thread.start()
            
            # Start audio stream
            with sd.InputStream(
                channels=CONFIG["CHANNELS"],
                samplerate=CONFIG["SAMPLE_RATE"],
                blocksize=CONFIG["BLOCK_SIZE"],
                callback=self.audio_callback,
                dtype=np.float32
            ):
                # Initial greeting
                self.tts.speak("Jarvis online")
                print("\n[INFO] Jarvis active. Sleeping... Say a wake word:")
                for i, word in enumerate(WAKE_WORDS, 1):
                    print(f"  {i}. {word}")
                
                print(f"[AUDIO] SR={CONFIG['SAMPLE_RATE']}, block={CONFIG['BLOCK_SIZE']}")
                print("[INFO] Ready for voice commands...")
                
                # Main loop
                while is_running:
                    time.sleep(0.1)
                    
        except KeyboardInterrupt:
            print("\n[INFO] Shutting down Jarvis...")
        except Exception as e:
            print(f"[ERROR] Main program error: {e}")
        finally:
            is_running = False
            self.system.stop_cpp_process()
            if 'consumer_thread' in locals() and consumer_thread.is_alive():
                consumer_thread.join(timeout=1)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print("[INFO] Jarvis shutdown complete")

def main():
    """Main entry point"""
    try:
        jarvis = JarvisCore()
        jarvis.run()
    except Exception as e:
        print(f"[ERROR] Failed to start Jarvis: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()