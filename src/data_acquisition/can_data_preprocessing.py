"""
This script is responsible for cleaning and preparing raw Cities at Night (CAN)
classification data to be usable for building a training dataset for the project.
The CAN data is supplied as a .csv file, is then cleaned and statistics as well as
relevant visualisations are produced in order to explore the CAN classification dataset more.
The final processed dataset is then saved as a new .csv file and can be plugged
into the DataDownload.py script
Version: 30/06/2020
"""
import sys

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


def load_raw_csv(csv_path: str) -> pd.DataFrame:
    """
    This function loads the .csv file as a Pandas Data Frame.
    :param csv_path: Path to .csv file
    :return: Pandas Data Frame of data
    """
    return pd.read_csv(csv_path, encoding="utf-8", delimiter=",")


def clean_data(data: pd.DataFrame) -> pd.DataFrame:
    """
    This function takes the loaded Pandas Data Frame and performs data cleaning and
    wrangling in order to only get the relevant data from the .csv file.
    :param data: CAN classification data as Pandas Data Frame
    :return: Cleaned and wrangled CAN data
    """

    #  Initial NaN drop - drops NaNs found in the .csv originally
    data.dropna(axis=0, inplace=True)

    #  Transform all city entries into upper case and replace spaces in city names with underscore
    #  This is done to match the data_downloader.py and training directory format
    data["CITY"] = data["CITY"].apply(lambda x: x.strip())
    data["CITY"] = data["CITY"].apply(lambda x: x.upper())
    data["CITY"] = data["CITY"].apply(lambda x: x.replace(" ", "_"))

    #  Drop duplicate entries
    data.drop_duplicates(inplace=True)

    #  Drop rows with NaN entries as a result of dropping duplicates
    data.dropna(axis=0, inplace=True)
    data.reset_index(drop=True, inplace=True)

    return data


def data_stats_and_plots(data: pd.DataFrame, plot_name: str) -> None:
    """
    This function calculates basic statistics for the CAN data and plots
    relevant visualisations in order to help understand the data better.
    :return:
    """
    sns.set()  # Uses Seaborn style instead of default PyPlot style

    #  Print basic statistics about the data
    print("The total number of training samples is: " + str(len(data)))
    print(
        "The average number of samples per city is: "
        + str(len(data) / int(data["CITY"].nunique()))
    )

    #  Bar Chart of 15 most classified cities
    labels = data["CITY"].value_counts()[:15].index.tolist()
    labels = [i.replace("_", " ") for i in labels]
    num_classifications = data["CITY"].value_counts()[:15].values.tolist()

    fig, ax = plt.subplots()
    plt.figure(figsize=(12, 14))
    plt.bar(labels, num_classifications, width=0.8, color="orange")
    plt.xticks(rotation="vertical", size=10)
    plt.xlabel("Cities")
    plt.ylabel("Number of Classifications")
    plt.title("CAN Dataset: 15 Most Classified Cities")
    plt.savefig(
        "../visualisations/" + plot_name.split(".csv")[0] + "_most_classified.png"
    )

    return


def export_to_csv(data: pd.DataFrame, name: str) -> None:
    """
    This function takes the cleaned Pandas DataFrame, aggregates it
    and then exports it as a .csv file usable in the data_downloader.py script.
    :return:
    """
    data = pd.DataFrame({"IMAGE": data["IMAGE"], "CITY": data["CITY"]})
    data.to_csv("../raw_training_data/" + name, index=False, sep=",")

    return


def main():

    data = load_raw_csv(
        sys.argv[1]
    )  # Takes path to .csv file from STDIN and loads the data
    data = clean_data(data)  # Executes the clean data function and returns cleaned data
    data_stats_and_plots(data, sys.argv[2])  # Returns data statistics and plots
    export_to_csv(
        data, sys.argv[2]
    )  # Exports the cleaned dataset ready for data_downloader.py script

    return


if __name__ == "__main__":
    main()
