"""
Configuration file for AI Voice Control Assistant
"""
import os
from pathlib import Path

# Base directories
BASE_DIR = Path(__file__).parent.parent
DATA_DIR = BASE_DIR / "data"
LOGS_DIR = BASE_DIR / "logs"

# Ensure directories exist
DATA_DIR.mkdir(exist_ok=True)
LOGS_DIR.mkdir(exist_ok=True)

# AI Model Configuration
AI_PROVIDER = os.getenv("AI_PROVIDER", "openai")  # Options: openai, anthropic, ollama
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
OLLAMA_BASE_URL = os.getenv("OLLAMA_BASE_URL", "http://localhost:11434")

# Model settings
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-4-turbo-preview")  # or claude-3-opus-20240229
TEMPERATURE = float(os.getenv("TEMPERATURE", "0.7"))
MAX_TOKENS = int(os.getenv("MAX_TOKENS", "2000"))

# Speech Recognition
SPEECH_RECOGNITION_ENGINE = os.getenv("SPEECH_ENGINE", "google")  # Options: google, sphinx, vosk
MICROPHONE_INDEX = int(os.getenv("MICROPHONE_INDEX", "0"))
ENERGY_THRESHOLD = int(os.getenv("ENERGY_THRESHOLD", "4000"))
DYNAMIC_ENERGY_THRESHOLD = True
PAUSE_THRESHOLD = float(os.getenv("PAUSE_THRESHOLD", "0.8"))

# Wake word settings
USE_WAKE_WORD = os.getenv("USE_WAKE_WORD", "true").lower() == "true"
WAKE_WORD = os.getenv("WAKE_WORD", "jarvis")  # Can be customized

# Text-to-Speech
TTS_ENGINE = os.getenv("TTS_ENGINE", "pyttsx3")  # Options: pyttsx3, gtts
TTS_VOICE_RATE = int(os.getenv("TTS_VOICE_RATE", "175"))
TTS_VOLUME = float(os.getenv("TTS_VOLUME", "0.9"))

# Browser settings
DEFAULT_BROWSER = os.getenv("DEFAULT_BROWSER", "chrome")  # Options: chrome, firefox
HEADLESS_BROWSER = os.getenv("HEADLESS_BROWSER", "false").lower() == "true"

# Database
DB_PATH = DATA_DIR / "user_preferences.db"

# Logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
LOG_FILE = LOGS_DIR / "voice_assistant.log"

# System behavior
AUTO_CONFIRM_ACTIONS = os.getenv("AUTO_CONFIRM_ACTIONS", "false").lower() == "true"
LEARNING_MODE = os.getenv("LEARNING_MODE", "true").lower() == "true"
MAX_COMMAND_HISTORY = int(os.getenv("MAX_COMMAND_HISTORY", "100"))
