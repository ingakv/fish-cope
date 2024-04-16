
import subprocess

# Run the images-to-video file
filename = "predict.py"
try:
    subprocess.run(['python', f"scripts/{filename}"], check=True)
    print(f"{filename} file executed successfully.")
except subprocess.CalledProcessError as e:
    print(f"An error occurred while running the {filename} file: {e}")



