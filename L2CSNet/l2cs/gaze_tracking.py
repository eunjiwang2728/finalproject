from l2cs import Pipeline, render
import cv2
import torch
from collections import deque
import numpy as np

def gaze_tracking(callback, camera_index=0):
    
    gaze_pipeline = Pipeline(
        weights='L2CSNet_gaze360.pkl',
        arch='ResNet50',
        device=torch.device('cpu')  # or 'cuda' if available
    )

    stability_pitch_threshold = 0.3
    stability_yaw_threshold = 0.2
    gaze_pitch_threshold_max = 0.15
    gaze_pitch_threshold_min = -0.15
    gaze_yaw_threshold_max = 0.25
    gaze_yaw_threshold_min = -0.20
    window_size = 10
    pitch_deque = deque(maxlen=window_size)
    yaw_deque = deque(maxlen=window_size)

    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print("Cannot open camera")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Can't receive frame (stream end?). Exiting ...")
                break

           
            #small_frame = cv2.resize(frame, (0, 0), fx=0.5, fy=0.5)
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            gray_frame = cv2.cvtColor(gray_frame, cv2.COLOR_GRAY2BGR)  

            results = gaze_pipeline.step(gray_frame)

            pitch = results.pitch if isinstance(results.pitch, (int, float)) else results.pitch[0] if len(results.pitch) > 0 else 0
            yaw = results.yaw if isinstance(results.yaw, (int, float)) else results.yaw[0] if len(results.yaw) > 0 else 0
            pitch_deque.append(results.pitch)
            yaw_deque.append(results.yaw)
            pitch_std = np.std(pitch_deque)
            yaw_std = np.std(yaw_deque)
            stability_status = "Stable" if pitch_std < stability_pitch_threshold and yaw_std < stability_yaw_threshold else "Unstable"
            looking_status = "Looking" if gaze_pitch_threshold_min < abs(pitch) < gaze_pitch_threshold_max and gaze_yaw_threshold_min < abs(yaw) < gaze_yaw_threshold_max else "Not Looking"

      
            callback(frame, stability_status, looking_status)

            if cv2.waitKey(1) == ord('q'):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()
