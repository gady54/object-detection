import asyncio
import threading
import time

import cv2
import mysql.connector
import numpy as np
import pandas as pd
import torch
import winsound
from flask import Flask, Response, jsonify
from flask_cors import CORS
# Global variables for drone counts
num_drones1 = 0
num_drones2 = 0
last_update_time = time.time()

# Calculate the direction and angle of the detected object
async def get_direction_and_angle(x1, x2, y1, y2, cap):
    frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
    frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
    center_x = frameWidth / 2
    center_y = frameHeight / 2
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    dx = x - center_x
    dy = center_y - y
    angles = np.degrees(np.arctan2(dy, dx))
    distances = np.sqrt(dx ** 2 + dy ** 2)
    angles = np.where(angles < 0, angles + 360, angles)
    return angles, distances

# Play a beep sound
def play_beep(frequency, duration):
    winsound.Beep(frequency, duration)

# Check if enough time has passed since the last update
def TimePassed(last_update_time):
    current_time = time.time()
    return int((current_time - last_update_time)) >= 10


# Generate a video stream from the camera feed
def generate_video_stream(cap):
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        results = model(frame)
        frame = results.render()[0]
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')

# Generate data for drone detection
async def generate_data(cap, camera_id):
    global num_drones1, num_drones2, direction, angle, last_update_time

    # Initialize variables
    frame_count = 0
    maxDrones = 0

    # Initialize database connection and cursor
    mydb, mycursor = InitDatabse()

    # Loop to process frames from the camera
    while cap.isOpened():
        # Read a frame from the camera
        ret, frame = cap.read()
        if not ret:
            break

        # Process every third frame
        frame_count += 1
        if frame_count % 3 != 0:
            continue
        # Perform object detection on the frame
        results = model(frame)
        frame = results.render()[0]
        results = results.pandas().xyxy[0]

        # Filter detected objects to get drones
        drones = results[(results['name'] == 'drone') & (results['confidence'] >= 0.5)]
        HowmanyDrones = len(drones)

        # Update global drone count variables
        if camera_id == 1:
            num_drones1 = HowmanyDrones
        else:
            num_drones2 = HowmanyDrones

        # Initialize or update the maximum number of drones detected
        if HowmanyDrones > maxDrones:
            maxDrones = HowmanyDrones

        # Check if enough time has passed since the last update
        if TimePassed(last_update_time):
            # Insert data into the database
            sql = "INSERT INTO DataDrone2Cam (DronesDetected, DateTime, `Time passed`, Camera) VALUES (%s, NOW(), %s, %s)"
            val = (maxDrones, int(time.time() - last_update_time), camera_id)
            mycursor.execute(sql, val)
            mydb.commit()
            maxDrones = 0
            last_update_time = time.time()

        # Calculate direction and angle of drones
        angle, direction = await get_direction_and_angle(drones['xmin'].values, drones['xmax'].values,
                                                         drones['ymin'].values, drones['ymax'].values, cap)
        directionSTR = ''.join(np.char.mod('%.2f', direction))
        angleSTR = ''.join(np.char.mod('%.2f', angle))

        # Play a beep if drones are detected
        if HowmanyDrones > 0:
            threading.Thread(target=play_beep, args=(1000, 500)).start()

        # Insert direction and angle data into the database
        sql = "INSERT INTO DataDrone2Cam (Direction, Angle, DateTime, `Time passed`, Camera) VALUES (%s, %s, NOW(), %s, %s)"
        val = (directionSTR, angleSTR, int(time.time() - last_update_time), camera_id)
        mycursor.execute(sql, val)
        mydb.commit()

        # Display the frame with detection results
        #cv2.imshow(f'Drone Detection - Camera {camera_id}', frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Release the camera and close all OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

# Run the drone detection process
def run_detection(cap, camera_id):
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_data(cap, camera_id))

# Initialize the database for drone detection
def InitDatabse():
    mydb = mysql.connector.connect(
        host="localhost",
        user="gady",
        password="Gad554007@",
        database="Drone_Detection_Data"
    )
    mycursor = mydb.cursor()
    mycursor.execute("SHOW TABLES LIKE 'DataDrone2Cam'")
    table_exists = mycursor.fetchone()
    if not table_exists:
        mycursor.execute(
            "CREATE TABLE DataDrone2Cam ( No INT AUTO_INCREMENT PRIMARY KEY, Camera INT, Direction VARCHAR(255), Angle VARCHAR(255), DronesDetected INT DEFAULT 0 , DateTime DATETIME, `Time passed` INT)"
        )
        mydb.commit()
    else:
        print("Table 'DataDrone2Cam' already exists")

    return mydb, mycursor

# Load the YOLOv5 model
weightsPath = 'C:/Users/gadyy/PycharmProjects/AI/yolov5/runs/train/exp/weights/best.pt'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = torch.hub.load('C:/Users/gadyy/PycharmProjects/AI/yolov5', 'custom', path=weightsPath, source='local').to(device)
cap1 = cv2.VideoCapture(1)
cap2 = cv2.VideoCapture(0)
direction = None
angle = None

app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Route to retrieve data from the SQL database
@app.route('/sql_data')
async def sql_data():
    mydb, mycursor = InitDatabse()
    sql = "SELECT * FROM DataDrone2Cam"
    df = pd.read_sql(sql, mydb)
    detection_data = df.to_dict(orient='records')
    return jsonify(detection_data)

# Route to retrieve live data
@app.route('/live_data')
async def live_data():
    global num_drones1, num_drones2, direction, angle
    drone_data = {
        "Angle": angle.tolist() if angle is not None else [],
        "Direction": direction.tolist() if direction is not None else [],
        "DronesDetectedCam1": num_drones1,
        "DronesDetectedCam2": num_drones2
    }
    return jsonify(drone_data)

# Route to stream live video from camera 1
@app.route('/live_stream_0')
async def video_feed1():
    return Response(generate_video_stream(cap1), mimetype='multipart/x-mixed-replace; boundary=frame')

# Route to stream live video from camera 2
@app.route('/live_stream_1')
async def video_feed2():
    return Response(generate_video_stream(cap2), mimetype='multipart/x-mixed-replace; boundary=frame')

# Start the drone detection process and run the web server
if __name__ == '__main__':
    dataThread1 = threading.Thread(target=run_detection, args=(cap1, 1))
    dataThread2 = threading.Thread(target=run_detection, args=(cap2, 2))
    dataThread1.start()
    dataThread2.start()
    app.run(host='0.0.0.0', port=5000)
