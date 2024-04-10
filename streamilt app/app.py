import streamlit as st
import cv2 as cv
from cv2 import aruco
import numpy as np
import argparse

# Load in the calibration data
calib_data_path = "./calib_data/MultiMatrix.npz"

def reshape_np(arr_n):
    arr_new = np.zeros((1, 1, 3), dtype=np.float64)
    arr_new[0][0][0] = arr_n[0][0]
    arr_new[0][0][1] = arr_n[1][0]
    arr_new[0][0][2] = arr_n[2][0]
    return arr_new

def my_estimatePoseSingleMarkers(corners, marker_size, mtx, distortion):
    marker_points = np.array([[-marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, marker_size / 2, 0],
                              [marker_size / 2, -marker_size / 2, 0],
                              [-marker_size / 2, -marker_size / 2, 0]], dtype=np.float32)
    
    for c in corners:
        nada, R, t = cv.solvePnP(marker_points, c, mtx, distortion, False, cv.SOLVEPNP_IPPE_SQUARE)
    
    rvecs = reshape_np(R)
    tvecs = reshape_np(t)

    return rvecs, tvecs, nada

# Load calibration data
calib_data = np.load(calib_data_path)
cam_mat = calib_data["camMatrix"]
dist_coef = calib_data["distCoef"]
ap = argparse.ArgumentParser()
ap.add_argument("-t", "--size", type=float, default=14.2, help="Minimum confidence level of detection")
args = vars(ap.parse_args())
MARKER_SIZE = args["size"]

marker_dict = aruco.getPredefinedDictionary(aruco.DICT_5X5_1000)
param_markers = aruco.DetectorParameters()

@st.cache_resource
def init_camera():
    return cv.VideoCapture(0)

cap = init_camera()

st.title("ArUco Detection")

# Define the layout with columns
col1, col2 = st.columns([4, 1])

# Video placeholder
with col1:
    video_placeholder = st.empty()

# Marker info placeholder
with col2:
    marker_info_placeholder = st.empty()

# Loop to continuously update the video frame
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_width = 720
    frame_height = int((frame_width / frame.shape[1]) * frame.shape[0])
    frame = cv.resize(frame, (frame_width, frame_height))
    gray_frame = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)
    dt = aruco.ArucoDetector(marker_dict, param_markers)
    marker_corners, marker_IDs, reject = dt.detectMarkers(gray_frame)

    if marker_corners:
        id=marker_IDs[0][0]
        if(id==300 or id==301 or id==302):
            MARKER_SIZE=10
        else:
            MARKER_SIZE=15
        rVec, tVec, _ = my_estimatePoseSingleMarkers(marker_corners, MARKER_SIZE, cam_mat, dist_coef)
        
        total_markers = range(0, marker_IDs.size)
        for ids, corners, i in zip(marker_IDs, marker_corners, total_markers):
            
            cv.polylines(
            frame, [corners.astype(np.int32)], True, (0, 255, 255), 4, cv.LINE_AA
            )
            corners = corners.reshape(4, 2)
            corners = corners.astype(int)
            top_right = corners[0].ravel()
            top_left = corners[1].ravel()
            bottom_right = corners[2].ravel()
            bottom_left = corners[3].ravel()
            
            # Draw the pose of the marker2
            try:
                point = cv.drawFrameAxes(frame, cam_mat, dist_coef, rVec[i], tVec[i], 4, 4)
            except Exception as e:
                print(e)

            # Calculating the distance
            try:
                distance = np.sqrt(tVec[i][0][0] ** 2 + tVec[i][0][2] ** 2 + tVec[i][0][1] ** 2)
            except Exception as e:
                print(e)
            
            try:   
                if marker_corners: 
                    table_html = f"""
                    <div style='border: 0px solid #e6e6e6; padding: 0px; border-radius: 5px;'>
                        <table>
                            <tr>
                                <td >ID</td>
                                <td style='color: #e71d36'>{ids[0]}</td>
                            </tr>
                            <tr>
                                <td >X</td>
                                <td style='color: #2ec4b6'>{tVec[i][0][0]:.2f}</td>
                            </tr>
                            <tr>
                                <td >Y</td>
                                <td style='color: #2ec4b6'>{tVec[i][0][1]:.2f}</td>
                            </tr>
                            <tr>
                                <td >Z</td>
                                <td style='color: #2ec4b6'>{tVec[i][0][2]:.2f}</td>
                            </tr>
                            <tr>
                                <td >Size</td>
                                <td style='color: #ff9f1c'>{MARKER_SIZE} cm</td>
                            </tr>
                        </table>
                    </div>
                    """

                    marker_info_placeholder.markdown(table_html, unsafe_allow_html=True)

                else:
                    marker_info_placeholder.empty()
            except Exception as e:
                print(e)
            cv.putText(frame, f"id: {ids[0]} Dist: {distance}", top_right, cv.FONT_HERSHEY_PLAIN, 1.3, (0, 0, 255), 2, cv.LINE_AA)
            try:
                cv.putText(frame, f"x: {round(tVec[i][0][0], 1)} y: {round(tVec[i][0][1], 1)}", bottom_right, cv.FONT_HERSHEY_PLAIN, 1.0, (0, 0, 255), 2, cv.LINE_AA)
            except:
                pass

    video_placeholder.image(frame, channels="BGR", use_column_width=True)

# Release the camera and close the OpenCV windows
cap.release()
cv.destroyAllWindows()
