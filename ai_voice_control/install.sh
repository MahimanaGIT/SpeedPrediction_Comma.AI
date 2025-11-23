#!/bin/bash

# AI Voice Control Assistant - Installation Script for Ubuntu
# This script installs all necessary dependencies and sets up the assistant

set -e  # Exit on error

echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   AI Voice Control Assistant - Installation               ║"
echo "║   Ubuntu Setup Script                                     ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""

# Check if running on Ubuntu/Debian
if ! command -v apt-get &> /dev/null; then
    echo "Error: This script is designed for Ubuntu/Debian systems"
    exit 1
fi

# Check for root/sudo
if [ "$EUID" -ne 0 ]; then
    echo "This script requires sudo privileges for system package installation."
    echo "Please run with sudo or as root."
    exit 1
fi

echo "[1/8] Updating system packages..."
apt-get update -qq

echo "[2/8] Installing system dependencies..."
apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    python3-venv \
    portaudio19-dev \
    libasound2-dev \
    libespeak1 \
    espeak \
    ffmpeg \
    xdotool \
    wmctrl \
    python3-pyaudio \
    flac \
    > /dev/null 2>&1

echo "[3/8] Installing Chrome/Chromium (for browser automation)..."
if ! command -v google-chrome &> /dev/null; then
    if ! command -v chromium-browser &> /dev/null; then
        apt-get install -y chromium-browser > /dev/null 2>&1 || \
        apt-get install -y chromium > /dev/null 2>&1 || \
        echo "Warning: Could not install Chrome/Chromium. Please install manually."
    fi
fi

echo "[4/8] Creating Python virtual environment..."
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"

# Remove old venv if exists
if [ -d "venv" ]; then
    rm -rf venv
fi

python3 -m venv venv
source venv/bin/activate

echo "[5/8] Upgrading pip..."
pip install --upgrade pip > /dev/null 2>&1

echo "[6/8] Installing Python dependencies..."
pip install -r requirements.txt > /dev/null 2>&1

echo "[7/8] Setting up environment configuration..."
if [ ! -f ".env" ]; then
    cp .env.example .env
    echo "Created .env file. Please edit it with your API keys:"
    echo "  nano .env"
else
    echo ".env file already exists"
fi

echo "[8/8] Creating launcher script..."
cat > run.sh << 'EOF'
#!/bin/bash
SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"
cd "$SCRIPT_DIR"
source venv/bin/activate
python3 voice_assistant.py "$@"
EOF

chmod +x run.sh

echo ""
echo "╔═══════════════════════════════════════════════════════════╗"
echo "║   Installation Complete!                                  ║"
echo "╚═══════════════════════════════════════════════════════════╝"
echo ""
echo "Next steps:"
echo "1. Edit the .env file with your API keys:"
echo "   nano $SCRIPT_DIR/.env"
echo ""
echo "2. Add your OpenAI or Anthropic API key to .env"
echo ""
echo "3. Run the assistant:"
echo "   cd $SCRIPT_DIR"
echo "   ./run.sh"
echo ""
echo "For more information, see README.md"
echo ""
