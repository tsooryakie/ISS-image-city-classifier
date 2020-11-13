"""
This program automates the download of training images
from the NASA ISS imagery repository using wget.
It additionally sorts the imagery into the relevant directories
in order to create the necessary training directory structure
used by PyTorch.
Version: 25/06/2020
"""

import os
import sys
import pandas as pd


def download_images(csv: str) -> None:

    """
    This function takes a path to .csv file of labelled images and
    returns a list of images to download
    :param csv: .csv file of labelled images
    """

    raw_csv = pd.read_csv(csv, delimiter=",")

    repo = "https://eol.jsc.nasa.gov/DatabaseImages/ESC/small/"

    for i in range(len(raw_csv)):
        mission_id = raw_csv["IMAGE"][i].split("-")[0]
        os.system("wget " + repo + mission_id + "/" + raw_csv["IMAGE"][i] + ".jpg")

        if not os.path.exists("../iss_image_data/train/" + raw_csv["CITY"][i]):
            os.mkdir("../iss_image_data/train/" + raw_csv["CITY"][i])
            os.system("mv " + raw_csv["IMAGE"][i] + ".jpg" + " ../iss_image_data/train/" + raw_csv["CITY"][i] + "/")
        else:
            os.system("mv " + raw_csv["IMAGE"][i] + ".jpg" + " ../iss_image_data/train/" + raw_csv["CITY"][i] + "/")

    return


def main():

    download_images(sys.argv[1])  # Path to .csv file

    return


if __name__ == "__main__":
    main()
