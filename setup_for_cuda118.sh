#!/bin/bash

# Update Homebrew
echo "Updating Homebrew..."
brew update

# Install Python, pip, and Node.js using Homebrew
echo "Installing Python, pip, and Node.js..."
brew install python
brew install node

# Ensure pip is up-to-date
echo "Updating pip..."
python3 -m pip install --upgrade pip

# Clone the YOLOv5 repository
echo "Cloning the YOLOv5 repository..."
git clone https://github.com/ultralytics/yolov5
cd yolov5

# Install required Python packages
echo "Installing required Python packages..."
pip3 install -r requirements.txt

# Install additional libraries
echo "Installing additional libraries..."
pip3 install numpy pandas requests flask quart serial mysql opencv-python mysql-connector-python

# Install torch and CUDA 11.8
echo "Installing torch and CUDA 11.8..."
pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

echo "Setup complete!"
