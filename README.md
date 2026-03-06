1. Install Homebrew and dependencies for Python venv

# If Homebrew not installed:
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"

# Basic tooling
brew update
brew install python ffmpeg

# Create project folder
mkdir -p ~/tello-test && cd ~/tello-test

# Python venv
python3 -m venv .venv
source .venv/bin/activate

# Install Python packages
pip install --upgrade pip
pip install djitellopy opencv-python

2) Connect to drone Wi‑Fi

1. Power on Tello
2. On Mac Wi‑Fi, connect to SSID like TELLO-XXXXXX
3. Wait until connected (no internet is normal)
