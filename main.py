import os

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


# Load the dataset
dataset_path = "./datasets"
vehicle_classes = []  # Initialize an empty list for the classes
img_size = (400, 400)

# Get the subdirectories in the dataset directory as class names
for class_name in os.listdir(dataset_path):
    if os.path.isdir(os.path.join(dataset_path, class_name)):
        vehicle_classes.append(class_name)


def load_previous_model():
    model = create_model()
    model.load_weights("vehicle_recognition_model.h5")

    return model


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
model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=10
)

# Save the model
model.save("vehicle_recognition_model.h5")
