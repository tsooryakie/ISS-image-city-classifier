"""
This script splits the full dataset into individual training, validation and testing dataset,
in a 80/10/10 split, respectively. It takes a random sample of images from each class and moves them
into the assigned directories.
Version: 14/07/2020
"""
import os
import sys
import random
import shutil


def split_dataset(root_dir: str, mode: str) -> None:
    """
    This function is responsible for building the validation and testing sets from the full dataset.
    The "mode" parameter sets which set to build.
    The function random.sample() is used instead of random.choice() because random.sample() removes the
    elements from the population once they have been chosen.
    :param root_dir: The path to the full dataset
    :param mode: "validation" to build validation set, "test" to build testing set.
    """

    random.seed(42)  # RNG is seeded to ensure that same images are used for the split (ensures fairness in experiment)

    classes_available = os.listdir(root_dir)

    for i in range(len(classes_available)):
        images_in_class = os.listdir(root_dir + classes_available[i])

        images_to_export = random.sample(images_in_class, int(len(images_in_class)*0.2))

        if not os.path.exists(root_dir.replace("train", mode) + classes_available[i]):
            os.makedirs(root_dir.replace("train", mode) + classes_available[i])

        for j in range(len(images_to_export)):
            shutil.move(root_dir + classes_available[i] + "/" + images_to_export[j],
                        root_dir.replace("train", mode) + classes_available[i] + "/" + images_to_export[j])

    return


def main():
    split_dataset(sys.argv[1], "validation")
    split_dataset(sys.argv[1], "test")

    return


if __name__ == "__main__":
    main()
