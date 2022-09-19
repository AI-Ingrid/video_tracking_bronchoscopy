"""
Author: Ingrid Tveten (SINTEF)
Convert video files to frames. With this script you are given some examples of
how to determine how sequences are generated, etc.
The output structure is as follows:
--Patient_001
    |---Sequence_001
        |---frame_1.png
        |---frame_2.png
        |---...
    |---Sequence_002
        |---frame_1.png
        |---frame_2.png
        |---...
    |...
--Patient_002
    |---Sequence_001
        |---frame_1.png
        |---frame_2.png
        |---...
    |---Sequence_002
        |...
--Patient_003
    |...
"""

import os
import math
import shutil
import imageio

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm   # create progress bars, see sample usage below

from utils.video_utilities import get_trim_start_end_frames, VideoTrimmingLimits
from utils.misc_utilities import find_next_folder_nbr
#os.environ["IMAGEIO_FFMPEG_EXE"] = "/Users/ikolderu/Library/Caches/pip/wheels/1d/57/24/4eff6a03a9ea0e647568e8a5a0546cdf957e3cf005372c0245"


matplotlib.use('Agg')   # Use 'Agg' backend to avoid memory overload


# =======================================================
#   Set input/output folders and frame interval
# =======================================================
#INPUT_DATA_PATH = "/data/virtual_videos"     # folder containing raw videos
#OUTPUT_PATH = "/path/to/processed/dataset"  # folder in which to put the processed dataset

def convert_video_to_frames(INPUT_DATA_PATH, OUTPUT_PATH):
    EXTRACT_FRAME_INTERVAL = 1  # Extract every x frames
    # =======================================================

    VIDEO_LIST = [fn for fn in os.listdir(INPUT_DATA_PATH) if
                (fn.lower().endswith('.avi') or fn.lower().endswith('.mpg')or fn.lower().endswith('.mp4'))]
    NB_PATIENTS = len(VIDEO_LIST)

    for p in tqdm(range(NB_PATIENTS), 'Patient'):

        # Create ./Patient_XX directory
        next_patient_nbr = find_next_folder_nbr(dataset_dir=OUTPUT_PATH)
        patient_dir = os.path.join(OUTPUT_PATH, f'Patient_{next_patient_nbr:03d}')
        try:
            os.makedirs(patient_dir, exist_ok=False)
        except OSError as exc:
            print(f"OSError: Patient folder {patient_dir} probably already exists")
            exit(-1)

        # TODO: Adjust! Should return list of videos belonging to current patient.
        #   Could be determined by videos having same date/time or other indicator.
        videos_for_patient = [fn for fn in VIDEO_LIST if fn.__contains__(f'{p:03d}')]

        # Generate sequences
        for video_fn in tqdm(videos_for_patient, 'Sequences'):

            # Create ./Patient_XX/Sequence_XX directory
            seq_nbr = find_next_folder_nbr(patient_dir)
            seq_dir = os.path.join(patient_dir, f'Sequence_{seq_nbr:03d}')
            try:
                os.makedirs(seq_dir, exist_ok=False)
            except OSError as exc:
                print(f"OSError: Sequence folder {seq_dir} probably already exists")
                exit(-1)

            # Get full path to video file and read video data
            video_path = os.path.join(INPUT_DATA_PATH, video_fn)
            vid_reader = imageio.get_reader(video_path)
            metadata = vid_reader.get_meta_data()
            fps = metadata['fps']
            duration = metadata['duration']
            nb_frames = math.floor(metadata['fps'] * metadata['duration'])

            # TODO: If using full video, set start time = 0 and end time = duration,
            #   but you can also get these numbers elsewhere and specify
            trim_time = VideoTrimmingLimits(t1=0., t2=duration)
            start_frame, end_frame = get_trim_start_end_frames(trim_time, fps, nb_frames)

            # Loop through the frames of the video
            for frnb, fr in enumerate(tqdm(range(start_frame, end_frame, int(EXTRACT_FRAME_INTERVAL)), 'Frames')):
                arr = np.asarray(vid_reader.get_data(fr))   # Array: [H, W, 3]

                # Display figure and image
                figure_size = (metadata['size'][0] / 100, metadata['size'][1] / 100)
                fig = plt.figure(figsize=figure_size)
                plt.imshow(arr, aspect='auto')

                # Adjust layout to avoid margins, axis ticks, etc. Save and close.
                plt.axis('off')
                plt.tight_layout(pad=0)
                plt.savefig(os.path.join(seq_dir, f'frame_{frnb:d}.png'))
                #plt.show()     # show() used only when debugging
                plt.close(fig)

            # Close reader before moving (video) files
            vid_reader.close()

            # ===============
            # After processing a video, move it to 'converted_files' folder to
            #   avoid double processing
            # ===============
            # Create Patient_XX directory to store converted files
            converted_files_path = os.path.join(
                INPUT_DATA_PATH, f'converted_files/Patient_{next_patient_nbr:03d}')
            try:
                os.makedirs(converted_files_path, exist_ok=False)
            except OSError as exc:
                print(f"OSError: Converted files folder {converted_files_path} probably already exists")
                exit(-1)

            # Move files after converting frames
            shutil.move(video_path, converted_files_path)