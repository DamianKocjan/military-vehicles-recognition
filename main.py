import os

import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow import keras
from tensorflow.keras import layers


# Load the dataset
dataset_path = "./datasets"
vehicle_classes = []  # Initialize an empty list for the classes
img_size = (256, 256)

# Get the subdirectories in the dataset directory as class names
for class_name in os.listdir(dataset_path):
    if os.path.isdir(os.path.join(dataset_path, class_name)):
        vehicle_classes.append(class_name)


def load_previous_model():
    model = create_model()
    model.load_weights("vehicle_recognition_model.h5")

    return model


def unpack_history(history, old_history=None):
    print(history.history)
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


def create_model():
    """Create the model"""

    model = keras.Sequential([
        layers.Conv2D(32, 3, activation="relu",
                      input_shape=(img_size[0], img_size[1], 1)),
        layers.MaxPooling2D(),
        layers.Conv2D(64, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Conv2D(128, 3, activation="relu"),
        layers.MaxPooling2D(),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(len(vehicle_classes))
    ])

    return model


strategy = tf.distribute.MirroredStrategy()

with strategy.scope():
    model = create_model()
# model = load_previous_model()

model.compile(
    optimizer="adam",
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

# Create a data generator for the dataset
data_generator = keras.preprocessing.image.ImageDataGenerator(
    validation_split=0.2
)

train_dataset = data_generator.flow_from_directory(
    dataset_path,
    target_size=img_size,
    color_mode="grayscale",
    classes=vehicle_classes,
    subset="training"
)

validation_dataset = data_generator.flow_from_directory(
    dataset_path,
    target_size=img_size,
    color_mode="grayscale",
    classes=vehicle_classes,
    subset="validation"
)

# Compile the model
model.compile(
    optimizer="adam",
    loss=tf.keras.losses.CategoricalCrossentropy(from_logits=True),
    metrics=["accuracy"]
)

# Train the model
history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

tl_history = process_and_display(history)

# Save the model
model.save("vehicle_recognition_model.h5")
