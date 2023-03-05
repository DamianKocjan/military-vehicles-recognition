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
img_size = (400, 400)
img_path = './process/datasets/test_image.jpg'


# Load the image and preprocess it
img = load_img(img_path, target_size=(img_size[0], img_size[1], 1))

img_array = img_to_array(img)

img_array = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
img_array = cv2.resize(img_array, img_size)
img_array = np.expand_dims(img_array, axis=0)
img_array = np.expand_dims(img_array, axis=3)
img_array = preprocess_input(img_array)


# Make a prediction
predictions = model.predict(img_array)

# Get the predicted class
predicted_class = np.argmax(predictions)

# Print the result
print('Predicted class:', vehicle_classes[predicted_class])

# Show probability of each class
labels = []
probabilities = []

for i in range(len(vehicle_classes)):
    if predictions[0][i] > 0:
        labels.append(vehicle_classes[i])
        probabilities.append(predictions[0][i])


results = zip(labels, probabilities)
results = sorted(results, key=lambda x: x[1], reverse=True)


print('Probability of each class:')
for label, probability in results:
    print('%s: %.2f%%' % (label, probability * 100))


# Show the image with the predicted class and bounding box
img = cv2.imread(img_path)
img = cv2.resize(img, img_size)
img = cv2.putText(img, vehicle_classes[predicted_class], (10, 30),
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)


# Get the bounding box
# for prediction in predictions:
#     # Extract the bounding box coordinates
#     x_min = prediction[0]
#     y_min = prediction[1]
#     width = prediction[2]
#     height = prediction[3]

#     x_min = int(x_min * img_size[0])
#     y_min = int(y_min * img_size[1])
#     width = int(width * img_size[0])
#     height = int(height * img_size[1])

#     cv2.rectangle(img, (x_min, y_min),
#                   (x_min+width, y_min+height), (0, 255, 0), 2)


cv2.imshow('Image', img)

(label, probability) = results[0]
title = "Prediction: %s - %.2f%%" % (label, probability * 100)
cv2.setWindowTitle('Image', title)

cv2.waitKey(0)
