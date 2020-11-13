"""
This program is part of the image pre-processing pipeline.
It takes an image and resizes it to desired dimensions using the OpenCV library.
Version: 06/07/2020
"""
import os
import sys
import cv2


def resize_image(image_path: str) -> None:
    """
    This function takes an original input image and resizes it to fit the dimensions
    accepted by Inception V3 Model (299x299). The data can then be further downsized in PyTorch,
    to fit models which have input of 224x224.
    :param image_path: Path to image which is to be resized
    """

    image = cv2.imread(image_path)

    CNN_DIMENSIONS = (224, 224)  # Output image dimensions - 224x224 pixels for pre-trained CNN input images
    image = cv2.resize(image, CNN_DIMENSIONS, interpolation=cv2.INTER_AREA)  # Appropriate Interpolation for shrinking

    print("Resized Image: " + image_path + " at dimensions: " + str(image.shape))

    root_path = "../iss_image_data/resized_iss_images/train/"
    image_class = image_path.split("/")[4] + "/"
    image_name = image_path.split("/")[-1]
    save_path = str(root_path + image_class + image_name)

    if not os.path.exists(save_path.split(image_name)[0]):
        os.makedirs(save_path.split(image_name)[0])

    cv2.imwrite(save_path, image)  # Writes the resized image to disk

    return


def main():

    all_images = []
    for root, directories, files in os.walk(sys.argv[1]):
        for file in files:
            if file.endswith(".jpg"):
                all_images.append(os.path.join(root, file))

    for i in range(len(all_images)):
        resize_image(all_images[i])

    return


if __name__ == "__main__":
    main()
