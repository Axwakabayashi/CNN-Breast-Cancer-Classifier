from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

data_dir = "/Users/axwakabayashi/Desktop/CNN-Breast-Cancer-Classifier/Breast-Cancer-Images"
image_exts = ['jpeg', 'jpg', 'png', 'bmp']

for image_class in os.listdir(data_dir):
    class_dir = os.path.join(data_dir, image_class)

    # Skip non-directories
    if not os.path.isdir(class_dir):
        continue

    for image in os.listdir(class_dir):
        image_path = os.path.join(class_dir, image)
        try:
            # Skip if the current path is a directory
            if os.path.isdir(image_path):
                continue

            # Open image using PIL
            img = Image.open(image_path)

            # Check if the file extension is in the allowed extensions
            _, ext = os.path.splitext(image)
            if ext[1:].lower() not in image_exts:
                print('Image not in ext list {}'.format(image_path))
                os.remove(image_path)
        except IsADirectoryError:
            print('Skipping directory {}'.format(image_path))
        except Exception as e:
            print('Issue with image {}: {}'.format(image_path, e))

            # Check if the user has permission to remove the file
            if os.access(image_path, os.W_OK):
                os.remove(image_path)
            else:
                print(f"Permission denied: {image_path}")


data = tf.keras.utils.image_dataset_from_directory(data_dir)
data_iterator = data.as_numpy_iterator()
batch = data_iterator.next()




