import cv2
import numpy as np
from cv2 import aruco
import pyrealsense2
from realsense_depth import *

# Load the dictionary for ArUco markers
aruco_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)

# Create parameters for ArUco detection
parameters = aruco.DetectorParameters()

# Create a VideoCapture object to capture video from the camera
# cap = cv2.VideoCapture(0)
dc = DepthCamera()

while True:
    # ret, frame = cap.read()
    ret, depth_frame, frame = dc.get_frame()

    if not ret:
        break
   
    # Convert the frame to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
   
    # Detect ArUco markers
    corners, ids, _ = cv2.aruco.detectMarkers(gray, aruco_dict, parameters=parameters)
   
    if ids is not None:
        for i in range(len(ids)):
            # Draw bounding box around the detected marker
            cv2.aruco.drawDetectedMarkers(frame, corners)
           
            # Calculate and display the coordinates of the marker
            c = corners[i][0]
            cx = int((c[0][0] + c[1][0] + c[2][0] + c[3][0]) / 4)
            cy = int((c[0][1] + c[1][1] + c[2][1] + c[3][1]) / 4)
            cv2.putText(frame, f"ID: {ids[i][0]}", (cx, cy), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
   
    # Display the frame
    cv2.imshow('ArUco Marker Detection', frame)
   
    # Check for the 'q' key to quit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the VideoCapture object and close all OpenCV windows
cap.release()
cv2.destroyAllWindows()