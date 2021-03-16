import tensorflow as tf
import numpy as np

import datetime
from PIL import Image

import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
import pandas as pd

from tensorflow.keras import layers
from tensorflow.keras.utils import plot_model
from tensorflow.keras.preprocessing import image_dataset_from_directory
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.losses import MeanSquaredError


"""
    # Whether to generate plots 
    MAKE_PLOTS = False

    # Target image dimensions
    img_height, img_width = 300, 300
    
    # Image data directory
    data_dir = '/Users/sb/Datasets/tensorflow_datasets/CUB_200_2011/'

"""

MAKE_PLOTS = False
img_height, img_width = 300, 300
data_dir = '/Users/sb/Datasets/tensorflow_datasets/CUB_200_2011/'


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

def get_bboxes(bbox_dir):
    
    # Load in bounding boxes
    bboxes = pd.read_csv(bbox_dir, 
                         index_col=0,
                         delimiter=" ", 
                         names=['x','y','width','height'])

    # Slice last 200 rows to get values for validation split when shuffle=False
    size = 200
    bboxes = bboxes.iloc[-size:]

    # Arrays for image sizes
    image_widths = np.zeros(size) + img_width
    image_heights = np.zeros(size) + img_height

    # Original image sizes
    original_widths = np.array([w for w, h in sizes])
    original_heights = np.array([h for w, h in sizes])

    # Find Scale factor for height/width of each image
    scale_factors = [image_widths / original_widths,
                     image_heights / original_heights]

    # Scale bounding box values
    for i, col in enumerate(bboxes.columns):

        bboxes[col+"_scaled"] = bboxes[col] * scale_factors[i % 2]
    
    
    return bboxes

def produce_plot():
    
    index = 32
    original = False

    # Create figure and axes
    fig, ax = plt.subplots()


    if original:
        image = Image.open(paths[index])
        x, y, w, h = bboxes.iloc[index,:4].values
        
    else:
        image = images[index] / 255
        x, y, w, h = bboxes.iloc[index,4:].values

    # Bounding box    
    bbox = Rectangle((x, y), w, h, 
                    linewidth=1, 
                    edgecolor='r', 
                    facecolor='none')

    ax.imshow(image)
    ax.add_patch(bbox)
    plt.show()

def build_small_model(EfficientNetB3=EfficientNetB3):
    
    
    # Freeze pretrained layers
    for pretrained_layer in base.layers:
        pretrained_layer.trainable = False

    # Output from pretrained base
    base_output = base.get_layer('block1b_add').output
    
    # Size reduction layers
    pool_1 = layers.AveragePooling2D(5, 5)(base_output)
    conv_final = layers.Conv2D(16, 4)(pool_1)
    pool_2 = layers.MaxPooling2D(3, 2)(conv_final)
    flatten = layers.Flatten()(pool_2)
    
    # Branch hidden layers
    y1_hidden = layers.Dense(5, activation='relu')(flatten)
    y2_hidden = layers.Dense(5, activation='relu')(flatten)
    y3_hidden = layers.Dense(5, activation='relu')(flatten)
    y4_hidden = layers.Dense(5, activation='relu')(flatten)
    
    # Output layers
    output1 = layers.Dense(1, name='output1')(y1_hidden)
    output2 = layers.Dense(1, name='output2')(y2_hidden)
    output3 = layers.Dense(1, name='output3')(y3_hidden)
    output4 = layers.Dense(1, name='output4')(y4_hidden)
    
    # Build final model using functional api 
    model = tf.keras.Model(base.input, [output1, output2, output3, output4])
    
    return model

def plot_results():
    # Plot results
    plt.style.use('ggplot')
    hist_df = pd.DataFrame(history.history)
    hist_df.plot(title='Loss');


if __name__ == "__main__":


    # Get image, labels, and original image sizes
    images, labels, paths = get_image_data(data_dir + "images")
    sizes = [Image.open(file).size for file in paths]
 
    # Get bounding boxes
    bboxes = get_bboxes(data_dir + "bounding_boxes.txt")
 
    # Load pretrained model
    base = EfficientNetB3()
 
    # Build model
    model = build_small_model()
 
    
    if MAKE_PLOTS:

        # Inspect architecture
        plot_model(model, dpi=120, show_shapes=True, show_layer_names=False)
 
        # Examine data
        produce_plot()


    # Scaled X, Y, width and height for bboxes
    outputs = (
        bboxes.iloc[:,4] / img_width,
        bboxes.iloc[:,5] / img_height,
        bboxes.iloc[:,6] / img_width,
        bboxes.iloc[:,7] / img_height,
    )


    # Callbacks
    early_stop = tf.keras.callbacks.EarlyStopping(patience=10, monitor='loss'),
    checkpoint = tf.keras.callbacks.ModelCheckpoint(f"checkpoints_{datetime.datetime.today()}"),
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(patience=4, min_lr=1e-5, monitor='loss'),


    loss = MeanSquaredError()
    losses = {f'output{i}': loss for i in range(1, 5)}
    model.compile(optimizer='adam', loss=losses)
    history = model.fit(images, 
                        outputs, 
                        epochs=15, 
                        batch_size=8, 
                        callbacks=[early_stop, checkpoint, reduce_lr])

    if MAKE_PLOTS:
        plot_results()
