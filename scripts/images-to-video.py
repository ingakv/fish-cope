
from scripts.functions import make_folder
from itertools import groupby

import cv2
import os

image_folder = '../runs/predict/'
output_folder = '../runs/predict_videos/'

make_folder(output_folder)

temp_group = []
groups = []

images = [img for img in os.listdir(image_folder) if img.endswith(".jpg")]


# Group together frames that belong to the same video
for _, group in groupby(images, lambda x: x.split("_")[0]):
    temp_group.append(list(group))


# Sort alphabetically, make sure frame 1 follows frame 0, and so on
while len(temp_group) > 0:

    group = temp_group.pop()
    number = 0
    sorted_group = []

    while len(group) > 0:
        for image in group:
            if image.find(f"frame{number}.jpg") != -1:
                group.remove(image)
                sorted_group.append(image)
                number += 1

    groups.append(sorted_group)




temp_group = groups
temp_group.reverse()
groups = []

number = 0

# Make sure frames from video 1 follow from video 0, and so on
while len(temp_group) > 0:

    for group in temp_group:
        if group[0].startswith(f"{number}_frame"):
            temp_group.remove(group)
            groups.append(group)
            number += 1


for group in groups:

    video_name = group[0].split("_")[0] + ".mp4"

    frame = cv2.imread(os.path.join(image_folder, group[0]))

    height, width, layers = frame.shape

    # Create video
    video = cv2.VideoWriter(os.path.join(output_folder, video_name), 0, 10, (width, height))

    for image in group:
        path = (os.path.join(image_folder, image))
        video.write(cv2.imread(path))
    video.release()
