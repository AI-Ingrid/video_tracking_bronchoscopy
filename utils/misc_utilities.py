"""
Various utilities for data handling
"""

import os
import numpy as np


def find_first_free_idx(arr):
    """
    Find the next unused integer in a sequence (e.g. 003 if indices 001 and 002
    are already used)

    Parameters
    ----------
    arr : array-like
        An array containing 'used' indices in a sequence
    """

    # Prepend the array with -1 and 0 so np.diff works
    arr = np.concatenate((np.array([-1, 0], dtype=np.int), np.array(arr, dtype=np.int)))
    # Compute differences between consecutive values
    where_sequence_breaks = np.where(np.diff(arr) > 1)[0]

    # If there ary any "holes" return the id of the first one
    if len(where_sequence_breaks) > 0:
        return where_sequence_breaks[0]
    # Otherwise return the integer succeeding the last element.
    else:
        return arr[-1] + 1


def find_next_folder_nbr(dataset_dir):
    """
    In a folder of numbered folders (e.g. Patient_001,... or Sequence_001,...),
    find the next integer that has not yet been used for a folder.

    Example: If the folders in the path-like object 'dataset_dir' are labelled
        'Patient_001', 'Patient_002', and 'Patient_003', the function will
        return 4 (the next integer in the sequence).

    Parameters
    ----------
    dataset_dir : str, or path-like
        Path to a folder containing numbered subfolders
    """
    # List subfolders in the folder
    subfolders = [p for p in os.listdir(dataset_dir)]
    if len(subfolders) == 0:
        return 1

    # Find highest subfolder number in folder
    subfolders = [int(folder.split('_')[-1]) for folder in subfolders]
    subfolders.sort()
    next_free_integer = find_first_free_idx(subfolders)
    return next_free_integer
