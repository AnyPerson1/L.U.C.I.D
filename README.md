LucidAI
Speak. Command. Control.

LucidAI is a local AI assistant that listens, interprets, and controls your PC entirely via voice.
Designed for offline use, it bridges natural language and system commands in real time with precision.

Features

Voice Recognition – Converts speech to text using a fast, accurate STT engine.
Command Understanding – Interprets your words into executable actions.
Automation Core – Handles keypresses, text input, and mouse actions directly through a C++ backend.
Voice Feedback – Provides real-time responses via TTS.
Full Control – From launching applications to complete desktop automation.

How It Works

Listen – Captures your voice input through the microphone.

Transcribe – Converts speech to text using the local STT engine.

Interpret – AI engine translates text into predefined commands.

Execute – C++ automation module triggers the corresponding system actions.

Respond – TTS module delivers instant spoken feedback.

Example:

You: "Open Spotify"
LucidAI → Presses Windows key → Types "Spotify" → Hits Enter

Tech Stack

Python – Voice recognition, AI integration, TTS.

C++ – Direct system calls for input simulation.

Faster-Whisper – High-speed, local speech-to-text engine.

Local AI Model – Natural language understanding without internet dependency.

Installation
# Clone the repository
git clone https://github.com/AnyPerson1/Lucid.git
cd LucidAI

# Install Python dependencies
pip install -r requirements.txt

# Build the C++ automation module
cd cpp_module
make  # or use your preferred compiler

Usage
python main.py


Then speak your command, and LucidAI will execute actions and respond locally.

License

MIT License – Free to use, modify, and distribute.
