import csv

import tensorflow as tf
import numpy as np

from PIL import Image

from tensorflow.keras.preprocessing import image_dataset_from_directory


MAKE_PLOTS = False
img_height, img_width = 300, 300
data_dir = '/Users/sb/Datasets/tensorflow_datasets/CUB_200_2011/images'


def get_image_data(data_dir):
    
    # Generate dataset from local image files
    ds = image_dataset_from_directory(
        data_dir,
        image_size = (img_height, img_width),
        label_mode='categorical',
        validation_split=.017,
        subset="validation",
        shuffle=False,
        batch_size=8,
        seed=420,)
    
    # Get numpy array from tf.dataset.Dataset object 
    images = []
    labels = []
    for X, y in ds.as_numpy_iterator():
        images.extend(X)
        labels.append(y)

    return np.array(images) / 255, np.array(labels), ds.file_paths



if __name__ == "__main__":

    _,_,paths = get_image_data(data_dir)
    sizes = [Image.open(path).size for path in paths]

    with open("original_sizes.txt", "w") as f:
        writer = csv.writer(f)
        for size in sizes:
            writer.writerow(size)

