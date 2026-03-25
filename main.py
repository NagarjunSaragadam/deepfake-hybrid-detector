from modules.face_detector import detect_face
from modules.frame_extractor import extract_frames
from modules.spatial_analysis import spatial_score
from modules.frequency_analysis import frequency_analysis
from modules.biological_analysis import biological_score
from modules.spatial_analysis import spatial_score_1
from modules.fusion_detector import fusion_score

import cv2
import os
import uuid

image_path = "input/images/test.jpg"
video_path = "input/videos/Sample.mp4"
frames_folder = "output/frames/"+str(uuid.uuid4())

def classify(score, threshold=0.5):
    return "Deepfake" if score >= threshold else "Real"

def process_video(video_path):
    os.makedirs(frames_folder, exist_ok=True)

    # Step 1: Extract frames
    print("Extracting frames...")
    extract_frames(video_path, frames_folder)

    # Step 2: Loop through frames
    spatial_scores = []
    frequency_scores = []
    
    print("Analyzing frames...")

    for file in sorted(os.listdir(frames_folder)):

        if not file.endswith(".jpg"):
            continue

        frame_path = os.path.join(frames_folder, file)

        # Step 3: Detect face
        face = detect_face(frame_path)

        if face is None:
            continue

        # Step 4: Run analyses on each frame
        spatial = spatial_score(face)
        frequency = frequency_analysis(face)

        spatial_scores.append(spatial)
        frequency_scores.append(frequency)

    # Step 5: Handle no-face case
    if len(spatial_scores) == 0:
        return {
            "error": "No faces detected in video"
        }

    # Step 6: Aggregate frame scores
    spatial_avg = sum(spatial_scores) / len(spatial_scores)
    frequency_avg = sum(frequency_scores) / len(frequency_scores)

    # Step 7: Biological analysis (full video)
    biological = biological_score(video_path)

    # Step 8: Fusion
    final_score = fusion_score(spatial_avg, frequency_avg, biological)    

    result = classify(final_score)
    print(result)
    print(final_score)

    return {
        "spatial_score": spatial_avg,
        "frequency_score": frequency_avg,
        "biological_score": biological,
        "final_score": final_score,
        "prediction": result
    }

def process_image(image_path):

    # Step 1: Detect face
    face = detect_face(image_path)

    if face is None:
        return {
            "error": "No face detected in image"
        }

    # Step 2: Run analyses
    spatial = spatial_score_1(face)
    frequency = frequency_analysis(face)

    # Step 3: Normalize (if not already done inside functions)
    spatial = float(spatial)
    frequency = float(frequency)

    # Step 4: Fusion (no biological)
    final_score =  (0.7 * spatial) + (0.3 * frequency)

    #final_score = round(final_score, 3)

    # Step 5: Classification
    if final_score > 0.5:
        result = "Real"
    else:
        result = "Deepfake"

    print(result)
    print(final_score)

    return {
        "spatial_score": spatial,
        "frequency_score": frequency,
        "final_score": final_score,
        "prediction": result
    }

print(process_image(image_path))