"""
Utilities for video handling
"""


class VideoTrimmingLimits(object):
    """
    Simple class to keep two time points that belong together.
    Example: Use for specifying start/end frames of parts of a video
    """

    def __init__(self, t1, t2):
        self.t1 = t1
        self.t2 = t2

    def __str__(self):
        return self.__repr__()

    def __repr__(self):
        return f"{self.__class__.__name__}( t1={self.t1}, t2={self.t2} )"


def get_trim_start_end_frames(trim_time, fps, nb_frames):
    """

    Parameters
    ----------
    trim_time : VideoTrimmingLimits
        The start (t1) and end (t2) times to trim the video
    fps : int/float
        The frame rate (frames per second) of the video
    nb_frames : int
        The total number of frames in the video sequence
    """

    start_frame = round(trim_time.t1 * fps)
    end_frame = round(trim_time.t2 * fps)
    if not (0 <= start_frame < nb_frames and 0 < end_frame <= nb_frames):
        raise ValueError(f"Error: start_frame or end_frame is outside of range [0, nb_frames]")
    if start_frame >= end_frame:
        raise ValueError(f"Error: start_frame {start_frame} >= end_frame {end_frame}")
    return start_frame, end_frame
