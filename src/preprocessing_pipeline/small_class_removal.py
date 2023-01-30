"""
This program deletes city classes which do not have enough training data in order
to be properly trained by the classifier.
Version: 26/07/2020
"""
import os
import shutil
import sys


def delete_small_classes(root_dir: str) -> None:
    """
    This function gets a list of all images in a directory and checks
    for the number of training examples in the directory. Any directories with less training samples
    than the threshold will get deleted.
    :param root_dir: Dataset directory
    :return:
    """
    all_cities = os.listdir(root_dir)
    number_removed = 0
    for city in all_cities:
        num_training_samples = len(os.listdir(root_dir + city))
        if num_training_samples < 15:
            shutil.rmtree(root_dir + city, ignore_errors=True)
            number_removed += 1
            print("Removed: " + root_dir + city)

    print("Number of city classes removed: " + str(number_removed))

    return


def main():

    delete_small_classes(sys.argv[1])

    return


if __name__ == "__main__":
    main()
