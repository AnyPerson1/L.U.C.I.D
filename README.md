THIS PROJECT USES ONLY ONE GEMINI API, YOU MIGHT CONFRONT SOME ERRORS DURING USAGE. IF YOU WANT LOCAL AI ASSISTANT, SEE THE Lucid-LOCAL REPOSITORY.



# LucidAI

# Speak. Command. Control.

LucidAI is a voice-powered AI assistant that listens, understands, and controls your PC with nothing but your words.
Powered by Faster-Whisper, Gemini AI, and a C++ automation core, it bridges natural speech and low-level system commands — all in real time.

🚀 Features

🎙 Voice Recognition – Uses Faster-Whisper for fast & accurate transcription.

🧠 AI Understanding – Interprets commands with Gemini AI.

⌨ Full Automation – Executes keypresses, text typing, and mouse clicks with a C++ backend.

🔊 Voice Feedback – Responds in real time using TTS.

⚡ Real-Time Control – From launching apps to full desktop automation.

🛠 How It Works

Listen – LucidAI captures your voice via microphone.

Transcribe – Faster-Whisper converts speech to text.

Interpret – Gemini AI translates your words into predefined commands.

Execute – C++ engine triggers keyboard/mouse actions.

Respond – TTS delivers spoken feedback instantly.

Example:

You: "Open Spotify"
LucidAI → Presses Windows key → Types "Spotify" → Hits Enter

📦 Tech Stack

Python – Voice recognition, AI integration, TTS.

C++ – Direct Windows API calls for input simulation.

Faster-Whisper – High-speed speech-to-text engine.

Gemini AI – Natural language understanding.

⚙ Installation
# Clone the repository
git clone https://github.com/AnyPerson1/Lucid.git
cd LucidAI

# Install Python dependencies
pip install -r requirements.txt

# Build the C++ automation module
cd cpp_module
make  # or use your preferred compiler

▶ Usage
python main.py


Then simply speak your command, and LucidAI will handle the rest.

📜 License

MIT License – Feel free to use, modify, and share.
