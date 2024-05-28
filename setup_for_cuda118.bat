@echo off

:: Install Chocolatey (Windows package manager)
echo Installing Chocolatey...
@powershell -NoProfile -ExecutionPolicy Bypass -Command "iex ((New-Object System.Net.WebClient).DownloadString('https://chocolatey.org/install.ps1'))"

:: Install Python, pip, and Node.js using Chocolatey
echo Installing Python, pip, and Node.js...
choco install -y python
choco install -y nodejs

:: Add Python and pip to PATH (may require restart or re-login to take effect)
setx PATH "%PATH%;C:\Python39;C:\Python39\Scripts"

:: Clone the YOLOv5 repository
echo Cloning the YOLOv5 repository...
git clone https://github.com/ultralytics/yolov5
cd yolov5

:: Install required Python packages
echo Installing required Python packages...
pip install -r requirements.txt

:: Install additional libraries
echo Installing additional libraries...
pip install numpy pandas requests flask quart serial mysql opencv-python mysql-connector-python

:: Install torch and CUDA 11.8
echo Installing torch and CUDA 11.8...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo Setup complete!
