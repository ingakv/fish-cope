import os
import xml.etree.ElementTree as ET
import zipfile
from PIL import Image, ImageDraw

annotation_folder = "../ANADROM/Annotation/"

annotations_data = []

temp_path = 'temp/'
modified_image_path = 'modified_images/'




# Extract Data from XML Files
for zip_file in os.listdir(annotation_folder):
    if zip_file.endswith(".zip"):

        try:
            # Open the zip file in read mode
            with zipfile.ZipFile(os.path.join(annotation_folder, zip_file), 'r') as zip_ref:

                # Filter file_list to include only files named "annotation.xml"
                file = [file for file in zip_ref.namelist() if os.path.basename(file) == "annotations.xml"][0]

                # Extract the file to a temporary location
                zip_ref.extract(file, temp_path)

                tree = ET.parse(os.path.join(temp_path, file))
                root = tree.getroot()
                for track in root.findall("./track"):
                    species = track.get("label")
                    for box in track.findall("./box"):
                        frame_number = int(box.get("frame"))
                        coordinates = [float(box.get(attr)) for attr in ["xtl", "ytl", "xbr", "ybr"]]

                        # Save data about frames which contains fish
                        annotations_data.append((zip_file, frame_number, species, coordinates))

                # Remove the extracted file after processing
                os.remove(temp_path + file)




        except Exception as e:
            print(f"An error occurred: {e}")







# Read Image Data and Overlay Annotations
for annotation in annotations_data:

    zip_path, frame_number, species, (xtl, ytl, xbr, ybr) = annotation

    image_path = os.path.join(annotation_folder, zip_path)

    try:
        # Open the zip file in read mode
        with zipfile.ZipFile(image_path, 'r') as zip_ref:

            # Get the file from the zip
            file_list = [file for file in zip_ref.namelist() if os.path.basename(file) == f"frame_{frame_number:06d}.PNG"]

            if len(file_list) > 1:
                print(f"Error: More than one file found for {image_path}")

            image_file = file_list[0]


            # Extract the file to a temporary location
            zip_ref.extract(image_file, temp_path)

            if os.path.exists(temp_path + image_file):

                # Draw the box around the fish
                image = Image.open(temp_path + image_file)
                draw = ImageDraw.Draw(image)
                draw.rectangle([(xtl, ytl), (xbr, ybr)], outline="red")

                # Save the modified image with annotations
                slay = zip_path.removeprefix("task_fish_detection-")
                slay2 = zip_path.removesuffix("-")


                annotated_image_path = modified_image_path + f"frame_{frame_number:06d}-" + zip_path.split('-')[1] + ".png"

                image.save(annotated_image_path)

            # Remove the extracted file after processing
            os.remove(temp_path + image_file)


    except Exception as e:
        print(f"An error occurred: {e}")



# Step 3: Analysis and Insights
# You can analyze patterns, behavior, and other insights based on the visualized annotations.
