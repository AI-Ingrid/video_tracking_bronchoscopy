"""
Utilities for reading and processing .csv (and .txt) files
"""

import csv


class NoMatchingFileError(Exception):
    def __init__(self, msg=None):
        if msg is not None:
            self.msg = msg
        else:
            self.msg = "No matching file was found"


# =====================================
#   HELPER FUNCTIONS
# =====================================

def read_csv_file(filename, has_header=False, include_header=False):
    """
    Read a .csv file with samples/data specified by row and return the content
    as a list of lists.

    filename : str, or path-like
        The path to the .csv file
    has_header : bool
        Whether the .csv file has a header specifying column contents
    include_header : bool
        Whether to include the header as the first row, or just return the
        actual file contents

    """
    print(f"Reading {filename} as .csv")

    with open(filename, newline='') as csvfile:
        reader = csv.reader(
            csvfile, delimiter=';', #quotechar='|',
            skipinitialspace=True,
        )

        if has_header:
            if include_header:
                i = 0   # start with first (header) row
            else:
                i = 1   # start with first content row
        else:
            i = 0   # start with first row and first content row

        data = []
        for row in reader:
            data.append([])
            for c in row:
                data[i].append(c)
            i += 1

        return data


def sort_data_by_row(data, has_header=False, include_header=False):
    if has_header:
        if include_header:
            samples = data      # do not skip first row
        else:
            samples = data[1:]  # skip first (header) row
    else:
        samples = data  # use all data if has_header=False

    d = {}
    for idx, sample in enumerate(samples):  # For each row
        entry = {
            'item': sample[0],  # name of item (filename or lymph node station)
            'start_time': float(sample[1]),  # time in seconds
            'end_time': float(sample[2]),    # time in seconds
        }
        d[idx] = entry

    return d


def parse_time_point_file(filename):
    """
    Example function for how to parse and use a file with time points
    associated with a video clip to extract sequences. Say each line in the
    file is formatted as
        indicator;start trim;end trim
    with time given in seconds. The function will then return a list of
        [indicator, VideoTrimmingLimits]
    items to use when extracting sequences. The indicator can for example be a
    short string describing the relevant part of the video
    """

    from video_utilities import VideoTrimmingLimits

    data = read_csv_file(filename, has_header=False)    # files have no header
    data_dict = sort_data_by_row(data, has_header=False)

    time_points_per_station = []
    for key, entry in data_dict.items():
        indicator = entry['item']
        trim_at = VideoTrimmingLimits(
            t1=entry['start_time'],
            t2=entry['end_time'],
        )
        time_points_per_station.append([indicator, trim_at])

    return time_points_per_station

