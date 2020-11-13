"""
This program contains helper functions which plot examples of augmented
images from the dataset which will then be fed into the CNN via dataloader
Version: 08/08/2020
"""

import numpy as np
import torchvision
import matplotlib.pyplot as plt


def undo_normalisation(image):
    """
    This function reverses the normalisation transformation to allow the
    images to show properly on the plot.
    :param image: Image from which normalisation transformation is removed
    :return:
    """
    image = image.numpy().transpose((1, 2, 0))
    means = np.array([0.485, 0.456, 0.406])
    std_dev = np.array([0.229, 0.224, 0.225])
    image = (image * std_dev + means).clip(0, 1)

    return image


def visualise_augmented_images(image_dataset, class_names):
    """
    This function plots the examples of augmented images.
    :param image_dataset: Dataset of images from which a data loader is built.
    :param class_names: Names of the classes in the dataset.
    :return:
    """
    images, indices = next(iter(image_dataset["train"]))
    
    input_images = torchvision.utils.make_grid(images)
    
    plt.figure(figsize=(10,8))
    plt.imshow(undo_normalisation(input_images))
    plt.title("Batch Visualisation with Augmentation")
    plt.axis("off")
    plt.savefig("../visualisations/batch_visualisation/batch_transformation.png")

    return
