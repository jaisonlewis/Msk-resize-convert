import pandas as pd
import tensorflow as tf
import cv2
import os
import ast
from tqdm import tqdm
from tensorflow.keras.optimizers import Adam
import numpy as np

# Define directories and directory name for images
image_dir = 'nor_img'
mask_dir = 'norm_img_mask'
label_dir = 'nor_label'
dirname = 'images2'

# Read the CSV file into a DataFrame
df = pd.read_csv('nor_img.csv')

# Extract filenames and annotations from the DataFrame
filenames = df['id'].values
annotations = df['annotations'].apply(ast.literal_eval).values

# Create output directories if they don't exist
os.makedirs(image_dir, exist_ok=True)
os.makedirs(mask_dir, exist_ok=True)
os.makedirs(label_dir, exist_ok=True)

# Initialize a counter for naming the saved files
start_no = 0

# Loop through each image and its corresponding annotations
for filename, annotation in tqdm(zip(filenames, annotations)):

    # Load the large image using TensorFlow and convert it into a tensor
    large_image = tf.keras.preprocessing.image.load_img(os.path.join(dirname, filename + '.tif'))
    large_image = tf.keras.preprocessing.image.img_to_array(large_image)

    # Resize the large image to 256x256 pixels
    resized_image = tf.image.resize(large_image, size=(256, 256))

    # Save the resized image as a PNG file
    imname = os.path.join(image_dir, "{}.png".format(filename))  # save the images
    tf.keras.preprocessing.image.save_img(imname, resized_image, scale=False)

    # Get the coordinates from the annotation and scale them to 256x256
    coordinates = annotation[0]['coordinates']
    scaled_coordinates = []
    for coord_set in coordinates:
        scaled_coord_set = [(int(coord[0] * 256 / 512), int(coord[1] * 256 / 512)) for coord in coord_set]
        scaled_coordinates.append(scaled_coord_set)

    # Create a mask using the scaled coordinates
    mask = np.zeros((256, 256, 3), dtype=np.uint8)
    for coord_set in scaled_coordinates:
        points = np.array(coord_set, dtype=np.int32)
        cv2.fillPoly(mask, [points], (255, 255, 255))

    # Save the mask as a PNG file
    mask_name = os.path.join(mask_dir, "{}.png".format(filename))  # save the mask
    tf.keras.preprocessing.image.save_img(mask_name, mask, scale=False)

    # Save the corresponding label as a text file
    label_name = os.path.join(label_dir, "{}.txt".format(filename))  # save the label
    with open(label_name, 'w') as f:
        f.write(annotation[0]['type'])

    # Increment the counter for naming the saved files
    start_no += 1