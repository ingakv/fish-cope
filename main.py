import itertools
import os
from xml.etree.ElementTree import parse
import zipfile
from PIL import Image

annotation_folder = "../ANADROM/Annotation/"

# Test folder with a small subset of the original footage
#annotation_folder = "test_images/"

annotations_data = []
species_data = []

temp_path = 'temp/'
modified_image_path = 'modified_images/'




temp_annotations_data = []

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

                tree = parse(os.path.join(temp_path, file))
                root = tree.getroot()

                labels = root.find(".//labels")


                for label in labels.findall("./label"):
                    species = label.find("name")
                    color = label.find("color")

                    # Save data about the fish species and box color
                    species_data.append((zip_file, species, color))



                for track in root.findall("./track"):
                    species = track.get("label")
                    for box in track.findall("./box"):
                        frame_number = int(box.get("frame"))
                        coordinates = [float(box.get(attr)) for attr in ["xtl", "ytl", "xbr", "ybr"]]

                        # Save data about frames which contains fish
                        temp_annotations_data.append((frame_number, species, coordinates))

                # Remove the extracted file after processing
                os.remove(temp_path + file)

                # Sort the data based on frame number
                temp_annotations_data = sorted(temp_annotations_data, key=lambda x: x[0])

                # Group annotations_data by frame number, and add it to annotations_data
                for (frame, group) in itertools.groupby(temp_annotations_data, key=lambda x: x[0]):
                    annotations_data.append((zip_file, frame, list(group)))

                temp_annotations_data.clear()





        except Exception as e:
            print(f"Error: {e}")







# Read Image Data and Overlay Annotations
for annotation in annotations_data:

    all_species = []

    zip_path, frame_number, fish_group = annotation

    # Add all the species and their box color in the current xml
    for species_zip_path, fish_species, _ in species_data:
        if species_zip_path == zip_path:
            all_species.append(fish_species)


    image_path = os.path.join(annotation_folder, zip_path)

    try:
        # Open the zip file in read mode
        with (zipfile.ZipFile(image_path, 'r') as zip_ref):

            # Get the file from the zip
            file_list = [file for file in zip_ref.namelist() if os.path.basename(file) == f"frame_{frame_number:06d}.PNG"]

            # Only one file should be found
            if len(file_list) > 1:
                print(f"Error: More than one file found for {image_path}")

            image_file = file_list[0]


            # Extract the file to a temporary location
            zip_ref.extract(image_file, temp_path)

            if os.path.exists(temp_path + image_file):
                image = Image.open(temp_path + image_file)


                for _, species, coordinates in fish_group:

                    cropped_image = image.crop(coordinates)

                    # Save the modified image with annotations
                    folder_path = modified_image_path + f"{species}/"

                    if not os.path.exists(folder_path):
                        os.makedirs(folder_path)


                    cropped_image_path = folder_path + f"/frame_{frame_number:06d}-{zip_path.split('-')[1]}.png"

                    modifier = 0
                    while os.path.exists(cropped_image_path):
                        modifier += 1

                        cropped_image_path = cropped_image_path.removesuffix(".png")

                        cropped_image_path = cropped_image_path.removesuffix(f"({modifier - 1})")

                        cropped_image_path += f"({modifier}).png"


                    cropped_image.save(cropped_image_path)

#                if os.path.exists(annotated_image_path):
#                    print(f"Replaced {annotated_image_path}")


            # Remove the extracted file after processing
            os.remove(temp_path + image_file)


    except Exception as e:
        print(f"An error occurred: {e}")



total_files = 0
for _, _, filenames in os.walk(modified_image_path):
     total_files += len(filenames)

print(f"Done with {total_files} images")

# Step 3: Analysis and Insights
# You can analyze patterns, behavior, and other insights based on the visualized annotations.
