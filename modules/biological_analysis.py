import cv2
import numpy as np
from scipy.signal import find_peaks

def extract_skin_signal(video_path=None):

    if video_path is None or video_path == "":
        cap = cv2.VideoCapture(0)
        print("Using webcam...")
    else:
        cap = cv2.VideoCapture(video_path)
        print("Using video file...")

    green_values = []

    frame_count = 0
    max_frames = 300   # limit for webcam capture

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Convert frame to RGB
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Extract green channel
        green_channel = rgb[:, :, 1]

        # Average green intensity
        avg_green = np.mean(green_channel)

        green_values.append(avg_green)

        frame_count += 1

        # Stop after some frames for webcam
        if video_path is None and frame_count >= max_frames:
            break

    cap.release()

    return np.array(green_values)


def biological_score(video_path=None):

    signal = extract_skin_signal(video_path)

    # Normalize signal safely
    signal = (signal - np.mean(signal)) / (np.std(signal) + 1e-8)

    # Detect peaks
    peaks, _ = find_peaks(signal, distance=5)

    heartbeat_strength = len(peaks) / len(signal)

    return heartbeat_strength