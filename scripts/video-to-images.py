
from scripts.functions import make_folder
import os
import cv2

# Test video location
video_folder = "../ANADROM/Video/"
files = os.listdir(video_folder)

output_folder = "data/frames"

# Create the directory if it does not already exist
make_folder(output_folder)

# Number of videos extracted
video_count = 0
total_frames = 0

for file in files:

    # Retrieve the video
    video = cv2.VideoCapture(os.path.join(video_folder, file))

    # FPS
    framerate = 10

    # Number of frames saved so far
    frames_saved = 0

    # Number of frames checked
    frame_count = 0


    while True:
        ret, frame = video.read()
        if not ret:
            break

        frame_count += 1

        if frame_count % int(video.get(5) / framerate) == 0:
            output_path = os.path.join(output_folder, f"{video_count}_frame{frames_saved}.jpg")
            cv2.imwrite(output_path, frame)
            frames_saved += 1
            total_frames += 1

    video_count += 1
    video.release()


print(f"Succesfully extracted {total_frames} images from {video_count} videos")