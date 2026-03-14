from modules.face_detector import detect_face
from modules.frequency_analysis import frequency_analysis
# from modules.spatial_analysis import spatial_score
from modules.biological_analysis import biological_score

image = "input/images/test.jpg"

face = detect_face(image)

# spatial = spatial_score(face)
spatial = 0.7  # Placeholder value for spatial score
frequency = frequency_analysis(face)
biological = biological_score()

final_score = (spatial + frequency + biological) / 3

if final_score > 0.6:
    result = "FAKE"
else:
    result = "REAL"

print("Spatial Score:", spatial)
print("Frequency Score:", frequency)
print("Biological Score:", biological)
print("Final Result:", result)