#!/bin/bash

# Clone the YOLOv5 repository
git clone https://github.com/ultralytics/yolov5
cd yolov5 || exit

# Install required Python packages
pip install -r requirements.txt

# Install additional libraries
pip install numpy pandas requests flask quart serial mysql  opencv-python mysql-connector-python

# Install torch and cuda 11.8

pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
