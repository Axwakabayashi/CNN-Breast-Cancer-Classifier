from PIL import Image
import numpy as np
from matplotlib import pyplot as plt
import tensorflow as tf
import os

gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
data_dir = os.path.join('CNN-Breast-Cancer-Classifier', 'Breast-Cancer-Images')
image_exts = ['jpeg', 'jpg', 'png', 'bmp']
print("Absolute path to data_dir:", os.path.abspath(data_dir))

data = tf.keras.utils.image_dataset_from_directory('data')