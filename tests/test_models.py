"""Tests for statistics functions within the Model layer."""

import os
import numpy as np
import numpy.testing as npt
import pytest
from unittest.mock import Mock
from inflammation.models import (
    daily_max,
    daily_min,
    daily_mean,
    analyse_data,
    patient_normalise,
    compute_standard_deviation_by_day,
    CSVDataSource,
    JSONDataSource,
)


def test_daily_mean_zeros():
    """Test that mean function works for an array of zeros."""

    test_input = np.array([[0, 0], [0, 0], [0, 0]])
    test_result = np.array([0, 0])

    # Need to use Numpy testing functions to compare arrays
    npt.assert_array_equal(daily_mean(test_input), test_result)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
        ([[1, 2], [3, 4], [5, 6]], [3, 4]),
        ([[1.5, 2.1], [3.4, 4.2], [5.3, 6.3]], [3.4, 4.2]),
    ],
)
def test_daily_mean(test, expected):
    """Test that mean function works for an array of zeros, positive integers and positive float numbers."""
    npt.assert_array_equal(daily_mean(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[4, 2, 5], [1, 6, 2], [4, 1, 9]], [4, 6, 9]),
        ([[4, -2, 5], [1, -6, 2], [-4, -1, 9]], [4, -1, 9]),
    ],
)
def test_daily_max(test, expected):
    """Test max function works for zeroes, positive integers, mix of positive/negative integers."""
    npt.assert_array_equal(daily_max(np.array(test)), np.array(expected))


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[0, 0, 0], [0, 0, 0], [0, 0, 0]], [0, 0, 0]),
        ([[4, 2, 5], [1, 6, 2], [4, 1, 9]], [1, 1, 2]),
        ([[4, -2, 5], [1, -6, 2], [-4, -1, 9]], [-4, -6, 2]),
    ],
)
def test_daily_min(test, expected):
    """Test min function works for zeroes, positive integers, mix of positive/negative integers."""
    npt.assert_array_equal(daily_min(np.array(test)), np.array(expected))


def test_daily_min_string():
    """Test for TypeError when passing strings"""

    with pytest.raises(TypeError):
        error_expected = daily_min([["Hello", "there"], ["General", "Kenobi"]])


@pytest.mark.parametrize(
    "test, expected, expect_raises",
    [
        (
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            [[0, 0, 0], [0, 0, 0], [0, 0, 0]],
            None,
        ),
        (
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            [[1, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[float("nan"), 1, 1], [1, 1, 1], [1, 1, 1]],
            [[0, 1, 1], [1, 1, 1], [1, 1, 1]],
            None,
        ),
        (
            [[1, 2, 3], [4, 5, float("nan")], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.8, 1, 0], [0.78, 0.89, 1]],
            None,
        ),
        (
            [[-1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            ValueError,
        ),
        (
            "unexpected_string",
            None,
            TypeError,
        ),
        (
            3,
            None,
            TypeError,
        ),
        (
            [4, 5, 6],
            None,
            ValueError,
        ),
        (
            [[1, 2, 3], [4, 5, 6], [7, 8, 9]],
            [[0.33, 0.67, 1], [0.67, 0.83, 1], [0.78, 0.89, 1]],
            None,
        ),
        (
            [[1, 2, 10], [2, 8, 100], [9, 21, 1123]],
            [[0.1, 0.2, 1], [0.02, 0.08, 1], [0.008, 0.019, 1]],
            None,
        ),
    ],
)
def test_patient_normalise(test, expected, expect_raises):
    """Test normalisation works for arrays of one and positive integers."""
    if isinstance(test, list):
        test = np.array(test)
    if expect_raises is not None:
        with pytest.raises(expect_raises):
            patient_normalise(test)

    else:
        result = patient_normalise(test)
        npt.assert_allclose(result, np.array(expected), rtol=1e-2, atol=1e-2)


def test_analyse_data():
    path = os.path.join(os.getcwd(), "./data")
    data_source = CSVDataSource(path)
    result = analyse_data(data_source)
    expected = [
        0.0,
        0.22510286,
        0.18157299,
        0.1264423,
        0.9495481,
        0.27118211,
        0.25104719,
        0.22330897,
        0.89680503,
        0.21573875,
        1.24235548,
        0.63042094,
        1.57511696,
        2.18850242,
        0.3729574,
        0.69395538,
        2.52365162,
        0.3179312,
        1.22850657,
        1.63149639,
        2.45861227,
        1.55556052,
        2.8214853,
        0.92117578,
        0.76176979,
        2.18346188,
        0.55368435,
        1.78441632,
        0.26549221,
        1.43938417,
        0.78959769,
        0.64913879,
        1.16078544,
        0.42417995,
        0.36019114,
        0.80801707,
        0.50323031,
        0.47574665,
        0.45197398,
        0.22070227,
    ]
    npt.assert_almost_equal(result, expected)


def test_analyse_data_mock_source():
    data_source = Mock()
    data_source.load_inflammation_data.return_value = [[[1, 2, 3]], [[8, 12, 24]]]

    analyse_data(data_source)


@pytest.mark.parametrize(
    "test, expected",
    [
        ([[1, 2, 3], [4, 5, 6]], [1.5, 1.5, 1.5]),
        ([[[1, 2, 3]], [[4, 5, 6]], [[7, 8, 9]]], [2.44948974, 2.44948974, 2.44948974]),
        ([[0, 0], [0, 0], [0, 0]], [0, 0]),
    ],
)
def test_compute_standard_deviation_by_day(test, expected):
    result = compute_standard_deviation_by_day(np.array(test))
    npt.assert_array_almost_equal(result, np.array(expected))
