# Olympe API Server for Parrot Anafi

## Setting Up the Olympe Environment

### Requirements
- **Operating System:** Ubuntu 18.04 or later (64-bit)
- **Python Version:** 3.7 or 3.8 recommended
- **Olympe SDK**: Installed from Parrot's official sources

### Installation Steps

1. **Install System Dependencies**
   ```bash
   sudo apt update
   sudo apt install -y python3-pip python3-venv
   sudo apt install -y ffmpeg v4l-utils
   sudo apt install -y libavcodec-extra
   sudo apt install -y libavdevice-dev libavfilter-dev libavformat-dev
   sudo apt install -y libavutil-dev libswscale-dev libswresample-dev
   sudo apt install -y libvideo-dev libx264-dev libx265-dev
   ```

2. **Clone Olympe Source**
   ```bash
   git clone https://github.com/Parrot-Developers/olympe.git
   cd olympe
   
3. **Setup Python Environment**
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   cd src/olympe
   
4. **Install Olympe SDK**
   ```bash
   python3 setup.py install
   
5. **Verify Installation**
   ```bash
   python3 -m pip list | grep olympe
   
6. **Run API Server**
   Make sure you are in the `controller_olympe` directory:
   ```bash
   python3 olympe_pi_api_server.py
   
### Usage
- **Start Server**: Run the API server using the Olympe SDK to control the Anafi drone.
- **API Endpoints**: Utilize endpoints like `/api/takeoff`, `/api/land`, `/api/flip` to control the drone.
- **Test Commands**: Verify commands using the JSON HTTP client or similar tools.

Ensure you have internet access for the initial SDK and package installations.