# Copyright (c) 2022, NVIDIA CORPORATION & AFFILIATES, ETH Zurich, and University of Toronto
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

import os
import pickle
from typing import Any


def load_pickle(filename: str) -> Any:
    """Loads an input PKL file safely.

    Args:
        filename (str): The path to pickled file.

    Raises:
        FileNotFoundError: When the specified file does not exist.

    Returns:
        Any: The data read from the input file.
    """
    if not os.path.exists(filename):
        raise FileNotFoundError(f"File not found: {filename}")
    with open(filename, "rb") as f:
        data = pickle.load(f)
    return data


def dump_pickle(filename: str, data: Any):
    """Saves data into a pickle file safely.

    Note:
        The function creates any missing directory along the file's path.

    Args:
        filename (str): The path to save the file at.
        data (Any): The data to save.
    """
    # check ending
    if not filename.endswith("pkl"):
        filename += ".pkl"
    # create directory
    if not os.path.exists(os.path.dirname(filename)):
        os.makedirs(os.path.dirname(filename), exist_ok=True)
    # save data
    with open(filename, "wb") as f:
        pickle.dump(data, f)
