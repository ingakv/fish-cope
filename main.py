import os
import shutil
import zipfile
import xml.etree.ElementTree as ET


def unzip_files(zip_file, extract_folder):
    # Create extract folder if it doesn't exist
    if not os.path.exists(extract_folder):
        os.makedirs(extract_folder)

    # Extract files
    with zipfile.ZipFile(zip_file, 'r') as zip_ref:
        zip_ref.extractall(extract_folder)


def convert_to_yolo(xml_file, images_folder, output_folder):
    # Parse XML file
    tree = ET.parse(xml_file)
    root = tree.getroot()

    # Iterate through each track in XML
    for track in root.findall('track'):
        track_id = track.attrib['id']
        label = track.attrib['label']

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
            txt_filename = f"{output_folder}/frame_{frame:06d}.txt"
            with open(txt_filename, 'a') as txt_file:
                txt_file.write(f"{label} {x_center} {y_center} {width} {height}\n")


if __name__ == "__main__":

    zip_file = "test_images/task_fish_detection-2024_02_05_18_43_34-cvat for video 1.1.zip"
    extract_folder = "extracted_data"
    xml_file = f"{extract_folder}/annotations.xml"
    images_folder = f"{extract_folder}/images"
    output_folder = "yolo_nas_output"

    # Unzip files
    unzip_files(zip_file, extract_folder)

    # Convert to YOLO-NAS format
    convert_to_yolo(xml_file, images_folder, output_folder)

    files = os.listdir(images_folder)

    # iterating over all the files in
    # the source directory
    for fname in files:
        # copying the files to the
        # destination directory
        shutil.copy2(os.path.join(images_folder, fname), output_folder)
