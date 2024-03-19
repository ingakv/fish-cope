import comet_ml

from tensorflow.keras import initializers as tf

# MIT introduction to deep learning package
#import mitdeeplearning as mdl

import matplotlib.pyplot as plt
import numpy as np

#import itertools
import os
from xml.etree.ElementTree import parse
import zipfile
from PIL import Image, ImageDraw

os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'


annotation_folder = "../ANADROM/Annotation/"

# Test folder with a small subset of the original footage
# annotation_folder = "test_images/"

annotations_data = []
species_data = []

temp_path = 'temp/'
modified_image_path = 'modified_images/'
image_path = 'all_images/'

image_amount = 0

COMET_API_KEY = "56OgyQ1BuZfPOBm3zTxSmWzAx"



# Function to parse annotations XML file
def parse_annotations(annotations_file):
    tree = ET.parse(annotations_file)
    root = tree.getroot()
    annotations = []
    for obj in root.findall('.//object'):
        name = obj.find('name').text
        bbox = obj.find('bndbox')
        xmin = int(bbox.find('xmin').text)
        ymin = int(bbox.find('ymin').text)
        xmax = int(bbox.find('xmax').text)
        ymax = int(bbox.find('ymax').text)
        annotations.append((name, (xmin, ymin, xmax, ymax)))
    return annotations


# Function to analyze images using Comet
def analyze_images(images_folder, annotations):
    # Initialize Comet experiment
    experiment = Experiment(project_name="fish-classification", workspace="ingakv", auto_metric_logging=True, api_key="56OgyQ1BuZfPOBm3zTxSmWzAx")
    experiment.log_parameters({"num_images": len(os.listdir(images_folder))})

    # Process each image
    for image_file in os.listdir(images_folder):
        image_path = os.path.join(images_folder, image_file)
        with Image.open(image_path) as img:
            # Analyze image (example: log number of fish in each image)
            num_fish = len([anno for anno in annotations if anno[0] == 'fish'])
            # Log analysis results
            experiment.log_metric("num_fish", num_fish)
            experiment.log_image(image_path)




# Read Image Data and Overlay Annotations
def read_data():
    for annotation in annotations_data:

        all_species = []

        zip_path, frame_number, fish_group = annotation

        # Add all the species and their box color in the current xml
        for elem in species_data:
            species_zip_path, _, _ = elem
            if species_zip_path == zip_path:
                _, fish_species, color = elem
                all_species.append((fish_species, color))

        image_path = os.path.join(annotation_folder, zip_path)

        try:
            # Open the zip file in read mode
            with (zipfile.ZipFile(image_path, 'r') as zip_ref):

                # Get the file from the zip
                file_list = [file for file in zip_ref.namelist() if
                             os.path.basename(file) == f"frame_{frame_number:06d}.PNG"]

                # Only one file should be found
                if len(file_list) > 1:
                    print(f"Error: More than one file found for {image_path}")

                image_file = file_list[0]

                # Extract the file to a temporary location
                zip_ref.extract(image_file, temp_path)

                if os.path.exists(os.path.join(temp_path, image_file)):
                    image = Image.open(os.path.join(temp_path, image_file))

                    for _, species, (xtl, ytl, xbr, ybr) in fish_group:

                        # Draw the box around the fish
                        draw = ImageDraw.Draw(image)

                        # Default to None if the species is not found
                        species_color = None

                        for s, c in all_species:
                            if s.text == species:
                                species_color = c
                                break

                        if species_color is None:
                            print(f"Error: No color found for {species}")

                        draw.rectangle([(xtl, ytl), (xbr, ybr)], outline=species_color.text, width=3)

                    # Save the modified image with annotations
                    annotated_image_path = modified_image_path + f"frame_{frame_number:06d}-{zip_path.split('-')[1]}.png"

                    if os.path.exists(annotated_image_path):
                        print(f"Replaced {annotated_image_path}")

                    image.save(annotated_image_path)

                # Remove the extracted file after processing
                os.remove(os.path.join(temp_path, image_file))
                


        except Exception as e:
            print(f"An error occurred: {e}")



def build_fc_model():
    fc_model = tf.keras.Sequential([
        # First define a Flatten layer
        tf.keras.layers.Flatten(),

        tf.keras.layers.Dense(128, activation=tf.nn.relu),

        tf.keras.layers.Dense(10, activation=tf.nn.softmax)  # Assuming 10 classes for MNIST

    ])
    return fc_model



def build_cnn_model():
    cnn_model = tf.keras.Sequential([

        # Define the first convolutional layer
        tf.keras.layers.Conv2D(filters=24, kernel_size=(3, 3), activation=tf.nn.relu),

        # Define the first max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        # Define the second convolutional layer
        tf.keras.layers.Conv2D(filters=36, kernel_size=(3, 3), activation=tf.nn.relu),

        # Define the second max pooling layer
        tf.keras.layers.MaxPool2D(pool_size=(2, 2)),

        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation=tf.nn.relu),

        # Define the last Dense layer to output the classification
        # probabilities. Pay attention to the activation needed a probability output
        tf.keras.layers.Dense(10, activation=tf.nn.softmax)

    ])

    return cnn_model



# start a first comet experiment for the first part of the lab
comet_ml.init(project_name="6S191lab2_part1_NN")
comet_model_1 = comet_ml.Experiment(COMET_API_KEY)

mnist = tf.keras.datasets.mnist



# Extract Data from XML Files
for zip_file in os.listdir(annotation_folder):

    if zip_file.endswith(".zip"):

        try:
            # Open the zip file in read mode
            with zipfile.ZipFile(os.path.join(annotation_folder, zip_file), 'r') as zip_ref:

                zip_path = os.path.join(image_path, f'{zip_file}/')

                # Extract the files to a temporary location
                zip_ref.extractall(path=zip_path)

                image_amount += len(os.listdir(zip_path + 'images'))

                # Filter file list to include only files named "annotation.xml"
                file = [file for file in zip_ref.namelist() if os.path.basename(file) == "annotations.xml"][0]

                if os.path.basename(file) == "annotations.xml":

                    tree = parse(os.path.join(zip_path, file))
                    root = tree.getroot()

                    labels = root.find(".//labels")

                    for label in labels.findall("./label"):
                        species = label.find("name")
                        color = label.find("color")

                        # Save data about the fish species and box color
                        isin = False

                        for elem in species_data:
                            if elem[0].text == species.text:
                                isin = True
                                break

                        if not isin:
                            species_data.append((species, color))



                    for track in root.findall("./track"):
                        species = track.get("label")
                        for box in track.findall("./box"):
                            frame_number = int(box.get("frame"))
                            coordinates = [float(box.get(attr)) for attr in ["xtl", "ytl", "xbr", "ybr"]]

                            # Save data about frames which contains fish
                            annotations_data.append((zip_file, frame_number, species, coordinates))

                    # Remove the extracted file after processing
                    os.remove(os.path.join(zip_path, file))

                print(f"Extracting: {zip_file}")


        except Exception as e:
            print(f"Error: {e}")


train_images = os.path.join(image_path, "images")
train_labels = annotations_data
test_images = test_labels = train_images

mnist = tf.keras.datasets.mnist
_, (test_images, test_labels) = mnist.load_data()
train_images = (np.expand_dims(train_images, axis=-1) / 255.).astype(np.float32)
train_labels = train_labels.astype(np.int64)
test_images = (np.expand_dims(test_images, axis=-1) / 255.).astype(np.float32)
test_labels = test_labels.astype(np.int64)

plt.figure(figsize=(10, 10))

random_inds = np.random.choice(len(annotations_data), 36)

for i in range(36):
    plt.subplot(6, 6, i + 1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    image_ind = random_inds[i]
    plt.imshow(np.squeeze(train_images[image_ind]), cmap=plt.cm.binary)
    plt.xlabel(train_labels[image_ind])

comet_model_1.log_figure(figure=plt)

model = build_fc_model()

model.compile(optimizer=tf.keras.optimizers.SGD(learning_rate=1e-1),
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])

# Define the batch size and the number of epochs to use during training
BATCH_SIZE = 64
EPOCHS = 5

model.fit(train_images, train_labels, batch_size=BATCH_SIZE, epochs=EPOCHS)
comet_model_1.end()

test_loss, test_acc = model.evaluate(test_images, test_labels)

print('Test accuracy:', test_acc)




cnn_model = build_cnn_model()

# Initialize the model by passing some data through
cnn_model.predict(train_images[[0]])

# Print the summary of the layers in the model.
print(cnn_model.summary())




comet_model_2 = comet_ml.Experiment(COMET_API_KEY)

cnn_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])



