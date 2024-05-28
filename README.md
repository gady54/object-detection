# object-detection - Instruction
1. 
***for Windows user:***
run in PowerShell: .\setup_for_cuda118.bat ***change the number of cuda that feet  with your GPU***
***for macOS/Linux:***
run in Terminal: ./setup_for_cuda181.sh ***change the number of cuda that feet  with your GPU***

2. copy "main.py" to tour project directory.
3. import a dataset of what you want train the module to detect like "Drone".
4. Create Directories: Create directories for train, val, and test sets. For example:
dataset/
    images/
        train/
        val/
        test/
    labels/
        train/
        val/
        test/

5. copy to the project file "data.yaml" put it in the location of your dataset.
6. train the model:
  run in terminal: python train.py --img 640 --batch 16 --epochs 30 --data data.yaml --cfg models/yolov5s.yaml --weights weights/yolov5m.pt --device 0
7. change in "main.py" the location of "best.pt" in variable "weightsPath".
8. run "main.py".





