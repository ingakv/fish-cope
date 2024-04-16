import math
import os
import shutil
import zipfile
import xml.etree.ElementTree as ET
from multiprocessing import freeze_support
from xml.etree.ElementTree import parse
from scripts.functions import make_folder
import random

import cv2

zip_folder = "../ANADROM/Annotation/"

# Test folder with a small subset of the original footage
#zip_folder = "test_images/"


files = os.listdir(zip_folder)

species_data = []

temp_path = 'temp/'
output_folder = "data/"


out_path_train = os.path.join(output_folder, "train/")

out_path_val = os.path.join(output_folder, "val/")

def define_dirs(st):
    return [os.path.join(path, f"{st}/") for path in [out_path_train, out_path_val]]

train_img, val_img = define_dirs("images")
train_label, val_label = define_dirs("labels")




xml_file = f"{temp_path}/annotations.xml"
temp_images_folder = f"{temp_path}/images"

image_count = 0
label_count = 0


def add_species(xml_path):
    tree = parse(xml_path)
    root = tree.getroot()

    labels = root.find(".//labels")

    for label in labels.findall("./label"):
        species = label.find("name")
        color = label.find("color")

        already_added = False
        for s in species_data:
            if s[0].text == species.text:
                already_added = True
                break

        # Save data about the fish species and box color
        if not already_added:
            species_data.append((species, color))



def label_to_int(str):
    if str == "SEACHAR":
        return 0
    elif str == "SALMON":
        return 1
    elif str == "SMOLT":
        return 2
    elif str == "SEATROUT":
        return 3
    elif str == "Pink Salmon":
        return 4
    return 5  # Other types of fish that are not labeled


def randomize_vals(count):

    # Get the list of files in the directory
    label = os.listdir(train_label)
    img = [elem.replace("txt", "PNG") for elem in label]

    indices = []

    # Generate random indexes
    while len(indices) < count:
        index = random.randint(0, label_count-1)
        if index not in indices:
            indices.append(index)

            # Moves the file from the train subfolder to the validation subfolder
            for path, old, new in [[img, train_img, val_img], [label, train_label, val_label]]:
                old_path = os.path.join(old, path[index])
                new_path = os.path.join(new, path[index])

                shutil.copy(old_path, new_path)

    # Deletes the files from the training set
    for index in indices:
        os.remove(os.path.join(train_img, img[index]))
        os.remove(os.path.join(train_label, label[index]))


def unzip_files(zip_file, extract_folder):
    # Create extract folder if it doesn't exist
    make_folder(extract_folder)

    # Extract files
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)


def convert_to_yolo(xml_file, output_folder, index):
    global label_count

    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Iterate through each track in XML
    for track in root.findall('track'):
        label = label_to_int(track.attrib['label'])

        # Iterate through each box in the track
        for box in track.findall('box'):
            frame = int(box.attrib['frame'])
            xtl = float(box.attrib['xtl'])
            ytl = float(box.attrib['ytl'])
            xbr = float(box.attrib['xbr'])
            ybr = float(box.attrib['ybr'])

            # Calculate YOLO-NAS format coordinates
            image_width = 1920  # Assuming image width is fixed
            image_height = 1080  # Assuming image height is fixed
            x_center = (xtl + xbr) / (2 * image_width)
            y_center = (ytl + ybr) / (2 * image_height)
            width = (xbr - xtl) / image_width
            height = (ybr - ytl) / image_height

            # Write YOLO-NAS format to text file
            txt_filename = f"{output_folder}/{index}_frame_{frame:06d}.txt"

            if not os.path.exists(txt_filename):
                label_count += 1

            with open(txt_filename, 'a') as txt_file:
                txt_file.write(f"{label} {x_center} {y_center} {width} {height}\n")


def add_gaussian_blur(im):
    return cv2.GaussianBlur(im, (9, 9), 500)



if __name__ == '__main__':

    freeze_support()

    blur = True

    # Create the directories if they do not already exist
    dirs = {
        train_img,
        train_label,
        val_img,
        val_label,
        out_path_train,
        out_path_val
    }

    for elem in dirs:
        make_folder(elem)


    for i, zip_file in enumerate(files):
        make_folder(temp_path)

        # Unzip the files
        unzip_files(os.path.join(zip_folder, zip_file), temp_path)

        # Convert the xml files to YOLO format
        convert_to_yolo(xml_file, train_label, i)

        # Add the species to the list
        add_species(xml_file)

        # Move the files to the correct directory
        for fileName in os.listdir(temp_images_folder):
            new_filename = f"{i}_{fileName}"
            old_path = os.path.join(temp_images_folder, fileName)
            new_path = os.path.join(train_img, new_filename)

            if blur:
                image = cv2.imread(old_path)

                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

                # Add gaussian blur
                gaussian_blur = add_gaussian_blur(image)

                # If converting to YCbCr
                # Convert to YCbCr
#                cr = (cv2.cvtColor(image, cv2.COLOR_BGR2YCrCb))[:, :, 2]

                cv2.imwrite(new_path, gaussian_blur)

            else:
                shutil.copy(old_path, new_path)

            os.remove(old_path)
            image_count += 1

    os.remove(xml_file)

    # Moves 10% of the dataset to the validation directory
    val_amount = math.floor(label_count * 0.1)
    randomize_vals(val_amount)


    print(f"Done with {image_count} images and {label_count} labels")