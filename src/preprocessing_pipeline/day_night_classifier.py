"""
This program classifies every image in the dataset on whether it is an image of a city at
day time or night time. It performs the classification based on average histogram value between the RGB channels
of the image. Images classified as day time are then deleted.
Version 26/07/2020
"""
import os
import sys

import cv2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import stats


def classify_image(img: str, make_plots=False) -> None:
    """
    Classifies the image according to the average of the RGB histogram.
    Images of cities at night will have a lower mode value than images of cities at daytime.
    The value of mode < 15 for night time images and mode > 15 for day time images was chosen
    from empirical observations to be sufficient for classification.
    Image classified as day time images are then deleted from the dataset.
    :param img: Image to classify
    :param make_plots: If true, shows plots of the histogram, else shows no plots
    :return: histogram_mode, value upon which the classification is based
    """

    image = cv2.imread(img, cv2.IMREAD_COLOR)
    image_name = img.split("/")[-1]
    image_name = image_name.split(".jpg")[0]
    image_city = img.split("/")[-2]

    colours = ["r", "g", "b"]

    if make_plots:
        plt.figure()

    total_mode = 0
    for i, colour in enumerate(colours):
        histogram, edges = np.histogram(image[:, :, i], bins=256, range=(0, 256))
        histogram_mode = stats.mode(histogram)
        histogram_mode = histogram_mode[1][0]
        total_mode += histogram_mode
        if make_plots:
            plt.plot(edges[:-1], histogram, color=colour)

    histogram_mode = total_mode / 3

    if make_plots:
        if histogram_mode < 15:
            plt.title(
                "Histogram for the night time image: "
                + image_name
                + " "
                + "("
                + image_city
                + ")"
            )
        else:
            plt.title(
                "Histogram for the day time image: "
                + image_name
                + " "
                + "("
                + image_city
                + ")"
            )
        plt.xlabel("Digital Number (DN)")
        plt.ylabel("Frequency")
        plt.legend(["Red", "Green", "Blue"])
        plt.xlim([0, 256])
        plt.show()

    if histogram_mode > 50:
        print("Removing image: " + img)
        os.remove(img)

    return


def main():

    sns.set()

    all_images = []
    for root, directories, files in os.walk(sys.argv[1]):
        for file in files:
            if file.endswith(".jpg"):
                all_images.append(os.path.join(root, file))

    for i in range(len(all_images)):
        classify_image(all_images[i], make_plots=False)

    return


if __name__ == "__main__":
    main()
