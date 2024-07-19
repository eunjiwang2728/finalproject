import cv2
import time
from fer import FER

def expression_recognition(input_queue, output_queue):
    detector = FER(mtcnn=True)

    start_time = None
    current_emotion = None
    alert_start_time = None
    non_target_start_time = None

    while True:
        frame = input_queue.get()
        if frame is None:
            break  

        emotions = detector.detect_emotions(frame)
        message = "No face detected"
        alert = False

        if emotions:
            top_emotion, top_score = max(emotions[0]['emotions'].items(), key=lambda item: item[1])
            if top_emotion in ['disgust', 'fear', 'angry', 'sad']:
                if current_emotion not in ['disgust', 'fear', 'angry', 'sad']:
                    start_time = time.time()
                current_emotion = top_emotion
                elapsed_time = time.time() - start_time
                message = f"Detected {current_emotion} for {elapsed_time:.2f} seconds"
                if elapsed_time > 1:
                    alert = True
            else:
                if current_emotion in ['disgust', 'fear', 'angry', 'sad']:
                    alert_start_time = time.time()
                if top_emotion in ['happy', 'neutral', 'surprise']:
                    if non_target_start_time is None:
                        non_target_start_time = time.time()
                    else:
                        non_target_elapsed_time = time.time() - non_target_start_time
                        if non_target_elapsed_time > 2:
                            alert = False
                current_emotion = None
                start_time = None
                message = f"Detected {top_emotion}"

        output_queue.put((message, alert))
