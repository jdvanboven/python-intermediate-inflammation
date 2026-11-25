"""Module containing models representing patients and their data.

The Model layer is responsible for the 'business logic' part of the software.

Patients' data is held in an inflammation table (2D array) where each row contains
inflammation data for a single patient taken over a number of days
and each column represents a single day across all patients.
"""

import numpy as np
import json
import glob
import os

class CSVDataSource:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Loads data from a given directory

        Collects all inflammation data from csv files within a given directory
        and returns the data."""

        data_file_paths = glob.glob(os.path.join(self.data_dir, "inflammation*.csv"))
        if len(data_file_paths) == 0:
            raise ValueError(
                f"No inflammation data CSV files found in path {self.data_dir}"
            )
        data = map(load_csv, data_file_paths)

        return list(data)


class JSONDataSource:
    def __init__(self, data_dir):
        self.data_dir = data_dir

    def load_inflammation_data(self):
        """Loads data from a given directory

        Collects all inflammation data from JSON files within a given directory
        and returns the data."""

        data_file_paths = glob.glob(os.path.join(self.data_dir, "inflammation*.json"))
        if len(data_file_paths) == 0:
            raise ValueError(
                f"No inflammation data JSON files found in path {self.data_dir}"
            )
        data = map(load_json, data_file_paths)

        return list(data)

def load_csv(filename):
    """Load a Numpy array from csv

        Parameters
        ----------
        filename : str
            path to the csv file

        Returns
        -------
        np.ndarray
            2D array of inflammation data
    """
    return np.loadtxt(fname=filename, delimiter=",")


def load_json(filename):
    """Load a numpy array from a JSON document.

    Expected format:
    [
        {
        "observations": [0, 1]
        },
        {
        "observations": [0, 2]
        }
    ]
    :param filename: Filename of JSON to load
    """
    with open(filename, "r", encoding="utf-8") as file:
        data_as_json = json.load(file)
        return [np.array(entry["observations"]) for entry in data_as_json]


def daily_mean(data):
    """Calculate the daily mean of a 2D inflammation data array.

    :param data: A 2D data array with inflammation data (each row contains measurements
    for a single patient across all days).
    :returns: An array of mean values of measurements for each day.
    """
    return np.mean(data, axis=0)


def daily_max(data):
    """Calculate the daily max of a 2D inflammation data array.

    :param data: A 2D data array with inflammation data (each row contains measurements
    for a single patient across all days).
    :returns: An array of max values of measurements for each day.
    """
    return np.max(data, axis=0)


def daily_min(data):
    """Calculate the daily min of a 2D inflammation data array.

    :param data: A 2D data array with inflammation data (each row contains measurements
    for a single patient across all days).
    :returns: An array of minimum values of measurements for each day.
    """
    return np.min(data, axis=0)


def calculate_standard_deviation(data):
    """Computes and returns standard deviation for data."""
    if np.any(np.isnan(data)):
        raise TypeError
    mean_inflammation = np.mean(data, axis=0)
    devs = []
    for entry in data:
        devs.append((entry - mean_inflammation) * (entry - mean_inflammation))

    standard_deviation = sum(devs) / len(data)
    return {'standard deviation': standard_deviation}
def patient_normalise(data):
    """
    Normalise patient data between 0 and 1 of a 2D inflammation data array.

    Any NaN values are ignored, and normalised to 0

    :param data: 2D array of inflammation data
    :type data: ndarray

    """
    if not isinstance(data, np.ndarray):
        raise TypeError('data input should be ndarray')
    if len(data.shape) != 2:
        raise ValueError('inflammation array should be 2-dimensional')
    if np.any(data < 0):
        raise ValueError('inflammation values should be non-negative')
    maximum_value = np.nanmax(data, axis=1)
    with np.errstate(invalid='ignore', divide='ignore'):
        normalised = data / maximum_value[:, np.newaxis]
    normalised[np.isnan(normalised)] = 0
    return normalised

def analyse_data(data_source):
    """Calculates the standard deviation by day between datasets.

    Gets all the inflammation data from CSV files within a directory,
    works out the mean inflammation value for each day across all datasets,
    and returns the standard deviation of these means."""

    data = data_source.load_inflammation_data()
    daily_standard_deviation = compute_standard_deviation_by_day(data)

    return daily_standard_deviation

def compute_standard_deviation_by_day(data):
    means_by_day = map(daily_mean, data)
    means_by_day_matrix = np.stack(list(means_by_day))
    daily_standard_deviation = np.std(means_by_day_matrix, axis=0)

    return daily_standard_deviation
