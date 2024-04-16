from multiprocessing import freeze_support

import comet_ml
import torch
from ultralytics import YOLO
import subprocess



# Run the main file
try:
    subprocess.run(['python', "main.py"], check=True)
    print("Main file executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the main file: {e}")




if __name__ == '__main__':
    try:
        # Call freeze_support() here
        freeze_support()

        # Load a model
        model = YOLO("yolov8s.pt")


        # Train the model
        results = model.train(
            data="fish.yaml",
            epochs=5,
            device="0",
            imgsz=640,
            plots=True,
            name="killer-fish",
            amp=False,
            batch=-1)
    except Exception as e:
        print(f"An error occurred while running the train file: {e}")
