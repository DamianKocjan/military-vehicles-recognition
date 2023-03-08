import os
import pandas as pd

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

import matplotlib.pyplot as plt

import numpy as np
from PIL import Image
from PIL import ImageColor
from PIL import ImageDraw
from PIL import ImageFont
from PIL import ImageOps

import time

print(tf.__version__)

print("The following GPU devices are available: %s" %
      tf.test.gpu_device_name())


# Load the dataset
TRAINING_DATASET_PATH = "./datasets/train"
VALIDATION_DATASET_PATH = "./datasets/valid"
TEST_DATASET_PATH = "./datasets/test"

vehicle_classes = []  # Initialize an empty list for the classes
IMG_SIZE = (256, 256)

BATCH_SIZE = 32

HISTORY_PATH = "./history/model_history.json"


def history_model_path_generator(i: int):
    return "./history/model_history_{}.csv".format(i)


# Get the subdirectories in the dataset directory as class names
for class_name in os.listdir(TRAINING_DATASET_PATH):
    if os.path.isdir(os.path.join(TRAINING_DATASET_PATH, class_name)):
        vehicle_classes.append(class_name)

MODEL_SEQUENTIALS = [
    [
        layers.Conv2D(16, 3, activation="relu",
                      input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(32, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(64, activation="relu"),
        layers.Dense(len(vehicle_classes))
    ],
    # [
    #     layers.Conv2D(32, 3, activation="relu",
    #                   input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(64, 3, activation="relu"),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(128, 3, activation="relu"),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(128, activation="relu"),
    #     layers.Dense(len(vehicle_classes))
    # ],
    # [
    #     layers.Conv2D(64, 3, activation="relu",
    #                   input_shape=(IMG_SIZE[0], IMG_SIZE[1], 1)),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(128, 3, activation="relu"),
    #     layers.MaxPooling2D(),
    #     layers.Conv2D(256, 3, activation="relu"),
    #     layers.MaxPooling2D(),
    #     layers.Flatten(),
    #     layers.Dense(256, activation="relu"),
    #     layers.Dense(len(vehicle_classes))
    # ],
]


def unpack_history(history, old_history=None):
    if old_history is None:
        new_history = {
            'accuracy': [],
            'val_accuracy': [],
            'loss': [],
            'val_loss': [],
        }
    else:
        new_history = old_history

    new_history['accuracy'] += history.history['accuracy']
    new_history['val_accuracy'] += history.history['val_accuracy']
    new_history['loss'] += history.history['loss']
    new_history['val_loss'] += history.history['val_loss']

    return new_history


def plot_history(training_history):
    acc = training_history['accuracy']
    val_acc = training_history['val_accuracy']
    loss = training_history['loss']
    val_loss = training_history['val_loss']
    epochs = np.arange(len(acc)) + 1

    fig = plt.figure(figsize=(12, 4))

    ax1 = fig.add_subplot(121)
    ax1.plot(epochs, loss, c='g', label='Train')
    ax1.plot(epochs, val_loss, c='r', label='Valid')
    ax1.set_title('Loss')
    ax1.legend(loc='lower left')
    ax1.grid(True)

    ax2 = fig.add_subplot(122)
    ax2.plot(epochs, acc, c='g', label='Train')
    ax2.plot(epochs, val_acc, c='r', label='Valid')
    ax2.set_title('Accuracy')
    # ax2.legend(loc='upper left');
    ax2.grid(True)

    plt.show()


def process_and_display(history, old_history=None):
    new_history = unpack_history(history, old_history)
    plot_history(new_history)
    return new_history


def save_history(history, i: int):
    try:
        os.mkdir("./history")
    except FileExistsError:
        pass

    history_df = pd.DataFrame.from_dict(history)
    history_df.to_csv(history_model_path_generator(i))

    # Append the history to the history file
    try:
        old = pd.read_json(HISTORY_PATH)
        new = history_df.append(old)
        new.to_json(HISTORY_PATH)
    except:
        history_df.to_json(HISTORY_PATH)


def create_model(sequential):
    model = keras.Sequential(sequential)
    return model


def create_data_generators():
    train_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        fill_mode='nearest')

    train_generator = train_datagen.flow_from_directory(
        TRAINING_DATASET_PATH,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42)

    valid_datagen = keras.preprocessing.image.ImageDataGenerator(
        rescale=1./255)

    valid_generator = valid_datagen.flow_from_directory(
        VALIDATION_DATASET_PATH,
        target_size=IMG_SIZE,
        color_mode='grayscale',
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        shuffle=True,
        seed=42)

    return train_generator, valid_generator


def main():
    strategy = tf.distribute.MirroredStrategy()

    old_history = None
    history = None

    with strategy.scope():
        train_generator, valid_generator = create_data_generators()

        for i, model_sequential in enumerate(MODEL_SEQUENTIALS):
            print(f"Model {i + 1} of {len(MODEL_SEQUENTIALS)}")
            model = create_model(model_sequential)
            model.compile(optimizer='adam',
                          loss=tf.keras.losses.CategoricalCrossentropy(
                              from_logits=True),
                          metrics=['accuracy'])

            model.summary()

            old_history = history
            history = model.fit(
                train_generator,
                epochs=1,
                steps_per_epoch=10,
                validation_data=valid_generator,
                verbose=1)

            process_and_display(history, old_history)
            process_and_display(history)

            save_history(unpack_history(history), i)

            model.save(f"./models/model_{i + 1}.h5")


if __name__ == "__main__":
    main()
