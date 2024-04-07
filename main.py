import os
import shutil
import zipfile
import xml.etree.ElementTree as ET


def label_to_int(str):
    if str == "FISH":
        return 0
    elif str == "SEATROUT":
        return 1
    elif str == "SMOLT":
        return 2
    elif str == "SALMON":
        return 3
    elif str == "SEACHAR":
        return 4
    return 5


def unzip_files(zip_file, extract_folder):
    # Create extract folder if it doesn't exist
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    # Extract files
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)


def convert_to_yolo(xml_file, output_folder, index):
    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Iterate through each track in XML
    for track in root.findall('track'):
        track_id = track.attrib['id']
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
            txt_filename = f"{output_folder}/{index}frame_{frame:06d}.txt"
            with open(txt_filename, 'a') as txt_file:
                txt_file.write(f"{label} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":

    zip_files = ["test_images/task_fish_detection-2024_01_28_18_33_21-cvat for video 1.1.zip",
                 "test_images/task_fish_detection-2024_02_05_18_43_34-cvat for video 1.1.zip",
                 "test_images/task_fish_detection-2024_02_05_11_51_48-cvat for video 1.1.zip"]

    # zip_file = "test_images/task_fish_detection-2024_01_28_18_33_21-cvat for video 1.1.zip"
    # zip_file = "test_images/task_fish_detection-2024_02_05_18_43_34-cvat for video 1.1.zip"
    # zip_file = "test_images/task_fish_detection-2024_02_05_11_51_48-cvat for video 1.1.zip"

    output_folder_txt = "yolo_txt"

    output_folder_img = "yolo_img"

    if not os.path.exists(output_folder_txt):
        os.makedirs(output_folder_txt)

    if not os.path.exists(output_folder_img):
        os.makedirs(output_folder_img)

    for i, zip_file in zip_files:
        extract_folder = "extracted_data" + i
        if not os.path.exists(extract_folder):
            os.makedirs(extract_folder)

        unzip_files(zip_file, extract_folder)
        xml_file = f"{extract_folder}/annotations.xml"
        images_folder = f"{extract_folder}/images"
        convert_to_yolo(xml_file, output_folder_txt, i)
        files = os.listdir(images_folder)

        for fileName in enumerate(os.listdir(images_folder)):
            new_filename = f"{i}_{fileName}"
            shutil.copy2(os.path.join(images_folder, fileName), os.path.join(output_folder_img, new_filename))
