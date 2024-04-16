import os.path
import shutil

from ultralytics import YOLO
from scripts.functions import make_folder
os.environ['KMP_DUPLICATE_LIB_OK']='True'



# Load a model
model = YOLO("runs/detect/dataset0_blurry/weights/best.pt")


vid_path = 'runs/detect/predict'
new_path = 'runs/prediction-videos'
source_path = '../ANADROM/video'
only_temp = 'temp'

make_folder(vid_path)
make_folder(new_path)
make_folder(only_temp)

for file in os.listdir(source_path):

    old_path = os.path.join(source_path, file)
    temp_path = os.path.join(only_temp, file)

    shutil.copy(old_path, temp_path)




dir_count = 0
files = os.listdir(only_temp)


count = 0

# Separate the source videos into different directory to avoid memory problems
while len(files) > 0:

    # Filter out any directories
    files = [file for file in files if os.path.isfile(os.path.join(only_temp, file))]

    for file in files:

        if count < 15*(dir_count+1):

            old_path = os.path.join(only_temp, file)

            new_dir = f'{vid_path}_{dir_count}'
            make_folder(new_dir)

            shutil.copy(old_path, os.path.join(new_dir, file))

            count += 1

            os.remove(old_path)

        else:
            break

    dir_count += 1



for i in range(dir_count - 1, -1, -1):
    if i > 0:

        path = f'{vid_path}_{i-1}'

        # Run batched inference on a list of images
        results = model(source=path, save=True, tracker='bytetrack.yaml')

        n_path = f'{vid_path}2'


        videos = os.listdir(n_path)

        # Move to save location
        while len(videos) > 0:

            vid = videos.pop()
            old_path = os.path.join(n_path, vid)

            shutil.copy(old_path, os.path.join(new_path, f'{count}.mp4'))

            count -= 1

            os.remove(old_path)

#        shutil.rmtree(path)
        os.rmdir(n_path)
