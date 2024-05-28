import torch
import cv2
import winsound
import numpy as np
import time
import asyncio
from quart import Quart, request, jsonify,Response
import threading
import serial
import mysql.connector
import pandas as pd


async def get_direction_and_angle(x1, x2, y1, y2, center_x, center_y):
    x = (x1 + x2) / 2
    y = (y1 + y2) / 2
    dx = x - center_x
    dy = center_y - y  # y axis is inverted in image coordinates
    angles = np.degrees(np.arctan2(dy, dx))
    distances = np.sqrt(dx ** 2 + dy ** 2)

    # Adjust angles to be in the range [0, 360)
    angles = np.where(angles < 0, angles + 360, angles)

    return angles, distances


def play_beep(frequency, duration):
    winsound.Beep(frequency, duration)


def TimePassed(last_update_time):
    current_time = time.time()
    return int((current_time - last_update_time)) >= 10  # you can change how much time you want herre is every minute


# initialize database for detection drones
mydb = mysql.connector.connect(
    host="localhost",
    user="gady",
    password="Gad554007@",
    database="Drone_Detection_Data"
)
mycursor = mydb.cursor()
# Check if the table exists
mycursor.execute("SHOW TABLES LIKE 'Data'")
table_exists = mycursor.fetchone()
if not table_exists:
    mycursor.execute(
        "CREATE TABLE Data (DroneNumber INT AUTO_INCREMENT PRIMARY KEY, Direction VARCHAR(255), Angle VARCHAR(255), DronesDetected INT, DateTime DATETIME, `Time passed` INT)")
    mydb.commit()
else:
    print("Table 'Data' already exists")

# Path to your trained model weights
weightsPath = 'C:/Users/gadyy/PycharmProjects/AI/yolov5/runs/train/exp4/weights/best.pt'

# Load the trained model and move it to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = torch.hub.load('C:/Users/gadyy/PycharmProjects/AI/yolov5', 'custom', path=weightsPath, source='local').to(
    device)
# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize serial communication
# ser = serial.Serial('COM5', 115200)  # Replace 'COM5' with your actual COM port
# Frame dimensions
frameWidth = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
frameHeight = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
frameCenterX = frameWidth / 2
frameCenterY = frameHeight / 2
last_update_time = time.time()  # Update the last update time
app = Quart(__name__)


@app.route('/sql_data')
async def sql_data():
    sql = "SELECT * FROM Data"
    df = pd.read_sql(sql, mydb)
    detection_data = df.to_dict(orient='records')
    return jsonify(detection_data)

@app.route('/live_data')
async def live_data():
    global HowmanyDrones,direction,angle
    drone_data = {
        "Angle": angle.tolist(),  # Convert NumPy array to list
        "Direction": direction.tolist(),  # Convert NumPy array to list
        "DronesDetected": HowmanyDrones
    }
    return jsonify(drone_data)

@app.route('/live_stream')
async def video_feed():
    return Response(generate_video_stream(), mimetype='multipart/x-mixed-replace; boundary=frame')

def generate_video_stream():
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Perform object detection on the frame from the camera
        results = model(frame)

        # Draw the detection results on the frame
        frame = results.render()[0]

        # Encode the frame in JPEG format
        ret, jpeg = cv2.imencode('.jpg', frame)
        if not ret:
            continue

        # Convert the JPEG frame to bytes and yield it
        frame_bytes = jpeg.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n\r\n')
async def generate_data():
    frame_count = 0
    maxDrones = 0
    global HowmanyDrones,direction,angle
    global last_update_time  # Use the global variable to track the last update time
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Draw results
        frame_count += 1
        if frame_count % 3 != 0:  # Process every 3rd frame
            continue

        # Performing object detection on the frame from the camera
        results = model(frame)

        # Drawing the detection results on the frame
        frame = results.render()[0]

        # Extract results for drones only
        results = results.pandas().xyxy[0]  # Convert to Pandas DataFrame
        drones = results[results['name'] == 'drone']

        # Count the number of drones detected
        HowmanyDrones = len(drones)
        if HowmanyDrones > maxDrones:
            maxDrones = HowmanyDrones

        if TimePassed(last_update_time):  # check if 10 seconds passed
            # Insert the max number of drones detected in the last 10 seconds into the database
            sql = "INSERT INTO Data (DronesDetected, DateTime, `Time passed`) VALUES (%s, NOW(), %s)"
            val = (maxDrones, int(time.time() - last_update_time))
            mycursor.execute(sql, val)
            mydb.commit()
            maxDrones = 0
            last_update_time = time.time()  # Update the last update time

        # get direction array and angle array
        angle, direction = await get_direction_and_angle(drones['xmin'].values, drones['xmax'].values,
                                                         drones['ymin'].values, drones['ymax'].values, frameCenterX,
                                                         frameCenterY)


        # convert angle and direction to str
        directionSTR = ''.join(np.char.mod('%.2f', direction))
        angleSTR = ''.join(np.char.mod('%.2f', angle))

        if HowmanyDrones > 0:
            # Play a beep sound with thread
            threading.Thread(target=play_beep, args=(1000, 500)).start()
            # Send serial data with thread
            # threading.Thread(target= sendSerialData, args=(HowmanyDrones, angleSTR, directionSTR)).start()

        # Insert the detection data into the table
        sql = "INSERT INTO Data (Direction, Angle, DateTime, `Time passed`) VALUES (%s, %s, NOW(), %s)"
        val = (directionSTR, angleSTR, int(time.time() - last_update_time))
        mycursor.execute(sql, val)
        mydb.commit()

        cv2.imshow('Drone Detection', frame)

        # Break loop with 'q' key
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()


def run_detection():
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    loop.run_until_complete(generate_data())


if __name__ == '__main__':
    # Start a separate thread for generating and sending data
    dataThread = threading.Thread(target=run_detection)
    dataThread.start()
    # Start the Quart app
    app.run(debug=True)
