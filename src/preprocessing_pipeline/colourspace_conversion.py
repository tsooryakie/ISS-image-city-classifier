"""
This script uses OpenCV library to convert ISS imagery from RGB colour space
into other colour spaces. The colour spaces supported by this script are: HSV, LAB and YUV.
The colour space to which the images are to be converted to is chosen by inputting the relevant
colour space as an "mode" keyword argument from STDIN.
Version: 11/07/2020
"""
import os
import sys

import cv2
import numpy as np


def load_image(image_path: str) -> np.ndarray:
    """
    This function loads a given ISS image as a Numpy Array in BGR colour space using OpenCV.
    It then converts the image to RGB colour space and returns the new RGB image as a Numpy array.
    :param image_path: Path to the ISS image.
    :return:
    """
    image = cv2.imread(image_path)  # Reads image in BGR colour space as a Numpy array
    image = cv2.cvtColor(
        image, cv2.COLOR_BGR2RGB
    )  # Converts BGR Numpy array to RGB colour space Numpy Array
    print("Loaded Image: " + str(image_path))

    return image


def colour_space_conversion(img: np.ndarray, mode=None) -> np.ndarray:
    """
    This function undertakes the actual colour space conversion from RGB into other specified colour spaces.
    The function takes a Numpy array of an image in RGB, and converts into the colour space specified
    in the "mode" keyword argument. If "mode" is unspecified or not supported, a Run Time Error is raised.
    :param img: ISS image as a Numpy array
    :param mode: The colour space for conversion, must be: "hsv", "lab" or "yuv".
    :return: ISS Image as Numpy array in the specified colour space.
    """

    if mode == "hsv":
        image = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        print("Converting Image to HSV Colour Space...")
    elif mode == "lab":
        image = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        print("Converting Image to LAB Colour Space...")
    elif mode == "yuv":
        image = cv2.cvtColor(img, cv2.COLOR_RGB2YUV)
        print("Converting Image to YUV Colour Space...")
    elif mode == "hls":
        image = cv2.cvtColor(img, cv2.COLOR_RGB2HLS)
        print("Converting Image to HLS Colour Space...")
    else:
        raise RuntimeError("Improper or No Colour Space Argument Supplied")

    return image


def save_as_tiff(img: np.ndarray, image_path: str, mode=None) -> None:
    """
    This function takes the converted image and saves it to disk as a .tiff image.
    The .tiff format is appropriate as it is not compressed and supports other colour spaces apart from RGB.
    The relevant string processing is undertaken in this function in order to save the output in the relevant
    directories.
    :param img: Colour space converted ISS Image as a Numpy array
    :param mode: Colour space in which the image is represented, must be: "hsv", "lab" or "yuv", otherwise
    a Run Time Error is raised.
    :param image_path: Path to the original image. This is used in string processing to save the converted image
    in the relevant directory.
    """

    image_class = (
        image_path.split("/")[4] + "/"
    )  # Extracts the class name (the city) of the image
    image_name = (
        image_path.split("/")[-1].split(".jpg")[0] + ".tiff"
    )  # Extracts the ISS image name

    if mode == "hsv":
        root_path = "../iss_image_data/hsv_iss_images/train/"
    elif mode == "lab":
        root_path = "../iss_image_data/lab_iss_images/train/"
    elif mode == "yuv":
        root_path = "../iss_image_data/yuv_iss_images/train/"
    elif mode == "hls":
        root_path = "../iss_image_data/hls_iss_images/train/"
    else:
        raise RuntimeError("Improper or No Colour Space Argument Supplied")

    save_path = str(
        root_path + image_class + image_name
    )  # Full path to which the image is to be saved to
    if not os.path.exists(
        save_path.split(image_name)[0]
    ):  # If the class directory does not exist, creates it
        os.makedirs(save_path.split(image_name)[0])

    cv2.imwrite(save_path, img)  # Writes the image to disk

    return


def main():

    """
    The code in the main function creates a list of all images and their paths to be converted.
    It uses os.walk() function which recursively passes through the directories to find all paths.
    The three functions defined in the script, load_image(), colour_space_conversion() and save_as_tiff()
    are then applied to each individual image in the "all_images" list.
    """
    all_images = []
    for root, directories, files in os.walk(
        "../iss_image_data/resized_iss_images/train/"
    ):
        for file in files:
            if file.endswith(".jpg"):
                all_images.append(os.path.join(root, file))

    for i in range(len(all_images)):
        image = load_image(all_images[i])
        image = colour_space_conversion(image, mode=sys.argv[1])
        save_as_tiff(image, all_images[i], mode=sys.argv[1])

    return


if __name__ == "__main__":
    main()
