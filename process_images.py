import os
import random

from PIL import Image, ImageFilter

img_size = (128, 128)

input_dir = "./data"
output_dir = "./process/datasets"


def random_blur(img: Image.Image):
    radius = random.random() * 2
    return img.filter(ImageFilter.GaussianBlur(radius=radius))


def process_image(path: str):
    img = Image.open(path)
    # Resize the image
    img = img.resize(img_size)
    # Convert the image to grayscale
    img = img.convert("L")

    path = os.path.join(output_dir, "noise", os.path.basename(path))

    filterred_img = random_blur(img)
    filterred_img.save(os.path.join(
        output_dir, "blur." + os.path.basename(path)))

    # Save the image
    img.save(os.path.join(output_dir, os.path.basename(path)))


def process_images():
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Get the paths of the images in the dataset directory
    image_paths = [
        os.path.join(input_dir, file_name)
        for file_name in os.listdir(input_dir)
        if file_name.endswith(".jpg")
    ]

    print("Processing images...")
    print("Number of images:", len(image_paths))

    # Process the images
    for image_path in image_paths:
        process_image(image_path)


if __name__ == "__main__":
    process_images()
