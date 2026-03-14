import cv2
import os

def extract_frames(video_path, output_folder):

    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Cannot open video")
        return

    count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame_path = os.path.join(output_folder, f"frame_{count}.jpg")
        cv2.imwrite(frame_path, frame)

        count += 1

    cap.release()

    print(f"{count} frames extracted")
    print(f"Video path: {video_path}")


if __name__ == "__main__":

    video_path = "input/videos/Sample.mp4"
    output_folder = "output/frames"

    os.makedirs(output_folder, exist_ok=True)

    extract_frames(video_path, output_folder)