import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
import cv2

dataset_path = "./datasets"

# Define the classes
vehicle_classes = []

for class_name in os.listdir(dataset_path):
    if os.path.isdir(os.path.join(dataset_path, class_name)):
        vehicle_classes.append(class_name)

# Load the model
model = tf.keras.models.load_model('vehicle_recognition_model.h5')

# Define the image size and file path
img_size = (128, 128)
img_path = './process/datasets/test_image.jpg'


# Load the image and preprocess it
img = load_img(img_path, target_size=(img_size[0], img_size[1], 1))

img_array = img_to_array(img)

img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
img_array = cv2.resize(img_array, (128, 128))
img_array = np.expand_dims(img_array, axis=0)
img_array = np.expand_dims(img_array, axis=3)
img_array = preprocess_input(img_array)


# Make a prediction
prediction = model.predict(img_array)

# Get the predicted class
predicted_class = np.argmax(prediction)

# Print the result
print('Predicted class:', vehicle_classes[predicted_class])

# Show probability of each class
labels = []
probabilities = []

for i in range(len(vehicle_classes)):
    if prediction[0][i] > 0:
        labels.append(vehicle_classes[i])
        probabilities.append(prediction[0][i])


results = zip(labels, probabilities)
results = sorted(results, key=lambda x: x[1], reverse=True)


print('Probability of each class:')
for label, probability in results:
    print('%s: %.2f%%' % (label, probability * 100))
