import os.path
from ultralytics import YOLO
import subprocess



# Run the video-to-images file
filename = "video-to-images.py"
try:
    subprocess.run(['python', f"{filename}"], check=True)
    print(f"{filename} file executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the {filename} file: {e}")





# Load a model
model = YOLO("runs/detect/blurry_dataset/weights/best.pt")

# Run batched inference on a list of images
results = model(source="data/frames", stream=True)  # return a generator of Results objects

output_folder = "runs/predict"
if not os.path.exists(output_folder):
    os.makedirs(output_folder)

# Process results generator
for result in results:

    boxes = result.boxes  # Boxes object for bounding box outputs
    masks = result.masks  # Masks object for segmentation masks outputs
    keypoints = result.keypoints  # Keypoints object for pose outputs
    probs = result.probs  # Probs object for classification outputs
    result.save(filename=os.path.join(result.save_dir, os.path.basename(result.path)))  # save to disk

print(f"Done with saving the prediction images")




# Run the images-to-video file
filename = "images-to-video.py"
try:
    subprocess.run(['python', f"{filename}"], check=True)
    print(f"{filename} file executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the {filename} file: {e}")



