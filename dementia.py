import cv2
import numpy as np
import random
import os
import sys

import tensorflow as tf
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D,
    MaxPooling2D,
    Flatten,
    Dense,
    Dropout
)
from tensorflow.keras.optimizers import Adam

EPOCHS = 10
IMG_WIDTH = 128
IMG_HEIGHT = 128
CATEGORIES_NAME = [
    'Very mild Dementia',
    'Non Demented',
    'Moderate Dementia',
    'Mild Dementia'
]
TEST_SIZE = 0.4


def main():
    # Check command-line arguments
    if len(sys.argv) not in [2, 3]:
        sys.exit("Usage: python dementia.py data_directory [model.h5]")

    print("Loading data...")
    # Get image arrays and labels for all image files
    images, labels = load_data(sys.argv[1])

    # Split data into training and testing sets
    labels = tf.keras.utils.to_categorical(labels)
    x_train, x_test, y_train, y_test = train_test_split(
        np.array(images), np.array(labels), test_size=TEST_SIZE
    )

    # Normalize pixel value by
    # converting to range (0 - 1) from (1 - 256)
    x_train = x_train.astype('float32')/255.0
    x_test = x_test.astype('float32')/255.0

    print("Building model...")
    # Get a compiled neural network
    model = get_model()

    print("Training model...")
    # Fit model on training data
    model.fit(x_train, y_train, epochs=EPOCHS)

    print("Testing model...")
    # Evaluate neural network performance
    model.evaluate(x_test,  y_test, verbose=2)

    # Save model to file
    if len(sys.argv) == 3:
        filename = sys.argv[2]
        model.save(filename)
        print(f"Model saved to {filename}.")


def load_data(data_dir):
    """
    Load images from the category subfolders in `data_dir`.

    Each image is read from disk, converted to RGB,
    resized to IMG_WIDTH × IMG_HEIGHT,
    and assigned a label based on its folder.
    All (image, label) pairs are then shuffled
    and returned.

    Returns:
        tuple: (images, labels)
            images: list of processed image arrays
            labels: list of corresponding integer class labels
    """
    images = []
    labels = []

    # loop thrugh each category folder
    for category_index, category_name in enumerate(CATEGORIES_NAME):
        category_dir = os.path.join(data_dir, category_name)

        # loop through each image in category
        for image_file in os.listdir(category_dir):
            image_path = os.path.join(category_dir, image_file)

            # only open .jpg files
            if image_file.lower().endswith((".jpg")):
                img = cv2.imread(image_path)

                # converts from BGR TO RGB format
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

                # reduce image dimention
                img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
                images.append(img)
                labels.append(category_index)

    # combine images and labels as a list of
    combined = list(zip(images, labels))

    # shuffle for randomness
    random.shuffle(combined)

    # unpack the list of tuples
    images, labels = zip(*combined)

    # Convert back to lists
    images = list(images)
    labels = list(labels)

    return (images, labels)


def get_model():
    """
    Build and return a compiled convolutional neural network (CNN) model
    for 128x128 RGB images and 4 output classes.

    Architecture:
    - 2 convolutional layers with 32 filters each (3x3 kernels)
    - Max pooling and dropout
    - 2 convolutional layers with 64 filters each (3x3 kernels)
    - Max pooling and dropout
    - Flatten layer
    - Fully connected layer with 512 units + dropout
    - Output layer with 4 units (softmax)

    Returns:
        model: A compiled Keras Sequential CNN model.
    """
    model = Sequential()

    # convolution Block 1
    model.add(Conv2D
              (32, kernel_size=(3, 3),
               input_shape=(128, 128, 3),
               activation='relu'))
    model.add(Conv2D(32, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # convolution Block 2
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(Conv2D(64, kernel_size=(3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Dropout(0.25))

    # Flatten to 1D
    model.add(Flatten())

    # Dense layer
    model.add(Dense(512, activation='relu'))
    model.add(Dropout(0.5))

    # output layer
    model.add(Dense(4, activation='softmax'))

    # compile the model
    model.compile(
        loss='categorical_crossentropy',
        optimizer=Adam(),
        metrics=['accuracy'])

    return model


if __name__ == "__main__":
    main()
