# AI Voice Control Assistant for Ubuntu

A powerful speech-to-speech AI assistant that controls your Ubuntu computer using voice commands. The assistant uses advanced AI models to understand natural language commands and execute complex tasks on your computer.

## Features

### 🎤 Voice Control
- **Speech Recognition**: Converts your voice commands to text using Google Speech Recognition or other engines
- **Wake Word Detection**: Optional wake word ("Jarvis" by default) to activate the assistant
- **Natural Language Processing**: Uses AI (OpenAI GPT-4, Claude, or local Ollama) to understand intent

### 🤖 AI-Powered Intelligence
- **Intent Understanding**: Comprehends complex commands like "look at prices of RayBan Meta glasses"
- **Action Planning**: Breaks down complex tasks into sequential actions
- **Context Awareness**: Remembers conversation history and user preferences
- **Adaptive Learning**: Learns from your usage patterns and adapts over time

### 🖥️ System Control
- **Application Control**: Open, close, and manage applications
- **Browser Automation**: Control Chrome/Firefox, search the web, navigate websites
- **Keyboard & Mouse**: Automated typing, clicking, and keyboard shortcuts
- **Information Extraction**: Extract prices, text, and data from web pages
- **Shell Commands**: Execute system commands safely

### 🧠 Learning & Adaptation
- **Usage Tracking**: Tracks frequently used apps and commands
- **Preference Learning**: Learns your preferences over time
- **Command History**: Maintains history of commands and results
- **Pattern Recognition**: Recognizes similar commands for faster processing

### 🔊 Text-to-Speech
- **Natural Voice**: Speaks responses using pyttsx3 TTS engine
- **Configurable**: Adjust voice rate, volume, and voice type
- **Context-Aware Responses**: Provides relevant verbal feedback

## Architecture

```
ai_voice_control/
├── voice_assistant.py          # Main orchestrator
├── config/
│   └── config.py              # Configuration management
├── modules/
│   ├── speech_recognition_module.py   # Voice input
│   ├── text_to_speech_module.py      # Voice output
│   ├── ai_processor.py               # AI/LLM integration
│   ├── system_controller.py          # System automation
│   └── preference_manager.py         # Learning & preferences
├── data/                      # User data and preferences
└── logs/                      # Application logs
```

## Installation

### Prerequisites
- Ubuntu 20.04+ or similar Debian-based Linux
- Python 3.8+
- Microphone and speakers
- Internet connection (for cloud AI models)
- API key for OpenAI or Anthropic (or local Ollama installation)

### Quick Install

1. **Clone or navigate to the directory:**
   ```bash
   cd ai_voice_control
   ```

2. **Run the installation script:**
   ```bash
   sudo ./install.sh
   ```

3. **Configure your API keys:**
   ```bash
   nano .env
   ```

   Add your API key:
   ```
   AI_PROVIDER=openai
   OPENAI_API_KEY=your_api_key_here
   ```

4. **Run the assistant:**
   ```bash
   ./run.sh
   ```

### Manual Installation

If you prefer manual installation:

```bash
# Install system dependencies
sudo apt-get update
sudo apt-get install -y python3 python3-pip python3-venv portaudio19-dev \
                        libasound2-dev libespeak1 espeak ffmpeg xdotool \
                        wmctrl python3-pyaudio flac chromium-browser

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install Python packages
pip install -r requirements.txt

# Configure environment
cp .env.example .env
nano .env  # Add your API keys

# Run
python3 voice_assistant.py
```

## Configuration

Edit the `.env` file to configure the assistant:

### AI Provider Options

**OpenAI (Recommended for best results):**
```env
AI_PROVIDER=openai
OPENAI_API_KEY=sk-your-key-here
MODEL_NAME=gpt-4-turbo-preview
```

**Anthropic Claude:**
```env
AI_PROVIDER=anthropic
ANTHROPIC_API_KEY=your-key-here
MODEL_NAME=claude-3-opus-20240229
```

**Local Ollama (Free, runs locally):**
```env
AI_PROVIDER=ollama
OLLAMA_BASE_URL=http://localhost:11434
MODEL_NAME=llama2
```

### Other Settings

```env
# Wake word (set to false for always-listening mode)
USE_WAKE_WORD=true
WAKE_WORD=jarvis

# Speech recognition
SPEECH_ENGINE=google  # Options: google, sphinx, vosk

# Text-to-speech
TTS_VOICE_RATE=175
TTS_VOLUME=0.9

# Browser
DEFAULT_BROWSER=chrome
HEADLESS_BROWSER=false

# Learning
LEARNING_MODE=true
AUTO_CONFIRM_ACTIONS=false
```

## Usage

### Starting the Assistant

```bash
cd ai_voice_control
./run.sh
```

Or with Python:
```bash
source venv/bin/activate
python3 voice_assistant.py
```

### Voice Commands

#### Application Control
- "Open Chrome"
- "Open Firefox and go to GitHub"
- "Close all browser windows"
- "Open terminal"
- "Launch calculator"

#### Web Automation
- "Search for Python tutorials"
- "Look at prices of RayBan Meta glasses"
- "Navigate to YouTube"
- "Show me the news"

#### Information Extraction
- "What's on this page?"
- "Extract prices from this page"
- "Read the title"

#### System Commands
- "Execute ls command"
- "Show my files"
- "Open settings"

#### Special Commands
- "Help" - Show help information
- "History" - Show recent commands
- "Preferences" - Show your preferences
- "Clear history" - Clear command history
- "Exit" / "Quit" - Stop the assistant

### Example Scenarios

**Scenario 1: Research Product Prices**
```
You: "Look at prices of RayBan Meta glasses"
Assistant: "I'll search for RayBan Meta glasses prices for you."
[Opens Chrome → Searches Google → Extracts prices]
Assistant: "I found these prices: $299, $329, $349"
```

**Scenario 2: Open Multiple Apps**
```
You: "Open Chrome and terminal"
Assistant: "Opening Chrome and terminal."
[Opens both applications]
Assistant: "Done!"
```

**Scenario 3: Web Research**
```
You: "Search for Python asyncio tutorial and open the first result"
Assistant: "Searching for Python asyncio tutorial."
[Opens browser, searches, clicks first result]
Assistant: "Opened the first result."
```

## How It Works

1. **Voice Input**: Your speech is captured via microphone and converted to text
2. **AI Processing**: The text is sent to an AI model (GPT-4/Claude/Ollama) which:
   - Understands your intent
   - Generates a structured action plan
   - Creates a spoken response
3. **Action Execution**: The system controller executes each action:
   - Opens applications
   - Controls the browser
   - Extracts information
   - Performs system operations
4. **Feedback**: Results are spoken back to you
5. **Learning**: The command and results are logged for future adaptation

## Troubleshooting

### Microphone Not Working
```bash
# Test your microphone
arecord -l

# Adjust microphone settings
alsamixer

# Check microphone index
python3 -c "import speech_recognition as sr; print(sr.Microphone.list_microphone_names())"
```

Update `MICROPHONE_INDEX` in `.env` if needed.

### Speech Recognition Issues
- **"Could not understand audio"**: Speak more clearly or reduce background noise
- **Timeout errors**: Increase timeout in config or speak more quickly
- **Wrong recognition**: Try adjusting `ENERGY_THRESHOLD` in `.env`

### Browser Automation Issues
```bash
# If Chrome driver fails, update it manually
pip install --upgrade webdriver-manager

# For Chromium instead of Chrome
sudo apt-get install chromium-browser chromium-chromedriver
```

### AI Model Issues
- **OpenAI errors**: Check your API key and billing status
- **Rate limits**: Add delays or upgrade your API plan
- **Ollama not responding**: Ensure Ollama is running (`ollama serve`)

### Audio Output Issues
```bash
# Test TTS
espeak "Hello world"

# If pyttsx3 doesn't work, install espeak
sudo apt-get install espeak espeak-data libespeak-dev

# Check available voices
python3 -c "import pyttsx3; engine = pyttsx3.init(); print(engine.getProperty('voices'))"
```

## Advanced Usage

### Using Local Ollama (No API Key Required)

1. **Install Ollama:**
   ```bash
   curl https://ollama.ai/install.sh | sh
   ```

2. **Pull a model:**
   ```bash
   ollama pull llama2
   # or
   ollama pull mistral
   ```

3. **Start Ollama server:**
   ```bash
   ollama serve
   ```

4. **Configure assistant:**
   ```env
   AI_PROVIDER=ollama
   MODEL_NAME=llama2
   OLLAMA_BASE_URL=http://localhost:11434
   ```

### Custom Wake Word

You can change the wake word in `.env`:
```env
WAKE_WORD=computer
# or
WAKE_WORD=hey assistant
```

### Disable Wake Word (Always Listen)

```env
USE_WAKE_WORD=false
```

### Auto-Confirm Actions

Skip confirmation prompts (use carefully):
```env
AUTO_CONFIRM_ACTIONS=true
```

## Security Considerations

⚠️ **Important Security Notes:**

1. **API Keys**: Never commit your `.env` file with API keys to version control
2. **Command Execution**: The assistant can execute shell commands - use `AUTO_CONFIRM_ACTIONS=false` for safety
3. **Browser Data**: Browser automation may access sensitive data - use a separate profile if needed
4. **Network**: Speech recognition may send audio to Google servers (use offline engines for privacy)
5. **Permissions**: The assistant needs microphone, system control, and network permissions

## Customization

### Adding New Action Types

Edit `modules/system_controller.py` and add new action handlers in the `execute_action` method.

### Changing Voice Settings

```python
# In modules/text_to_speech_module.py
tts.set_rate(200)  # Faster speech
tts.set_volume(1.0)  # Max volume
```

### Custom Commands

Teach the AI by using commands consistently. The learning system will adapt over time.

## Performance Tips

1. **Use Ollama locally** for faster response times (no API calls)
2. **Reduce `MAX_TOKENS`** in `.env` for faster AI responses
3. **Disable `LEARNING_MODE`** if you don't need preference tracking
4. **Use `HEADLESS_BROWSER=true`** for faster browser automation
5. **Adjust `PAUSE_THRESHOLD`** for quicker speech detection

## Project Structure

```
ai_voice_control/
├── voice_assistant.py              # Main entry point
├── install.sh                      # Installation script
├── run.sh                         # Launcher script
├── requirements.txt               # Python dependencies
├── .env.example                   # Configuration template
├── .env                          # Your configuration (not in git)
├── README.md                     # This file
│
├── config/
│   ├── __init__.py
│   └── config.py                 # Configuration management
│
├── modules/
│   ├── __init__.py
│   ├── speech_recognition_module.py    # Voice → Text
│   ├── text_to_speech_module.py       # Text → Voice
│   ├── ai_processor.py                # AI/LLM integration
│   ├── system_controller.py           # System automation
│   └── preference_manager.py          # Learning & DB
│
├── data/
│   └── user_preferences.db       # SQLite database
│
└── logs/
    └── voice_assistant.log       # Application logs
```

## Contributing

This is a proof-of-concept voice assistant. Areas for improvement:

- [ ] Better wake word detection (Porcupine, Snowboy)
- [ ] Offline speech recognition (Vosk, Whisper)
- [ ] Multi-language support
- [ ] Better web scraping
- [ ] Plugin system for custom actions
- [ ] GUI for settings
- [ ] Mobile app integration

## License

MIT License - Feel free to modify and extend!

## Credits

Built with:
- **SpeechRecognition** - Voice input
- **pyttsx3** - Text-to-speech
- **OpenAI/Anthropic** - AI models
- **Selenium** - Browser automation
- **PyAutoGUI** - System control
- **SQLite** - Data persistence

## Support

For issues, questions, or contributions, please open an issue on the GitHub repository.

---

**Enjoy your AI-powered voice control assistant!** 🎤🤖