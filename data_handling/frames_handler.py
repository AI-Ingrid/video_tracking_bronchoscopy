import os
import math
import imageio
import cv2

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm   # create progress bars, see sample usage below

from utils.video_utilities import get_trim_start_end_frames, VideoTrimmingLimits
from utils.misc_utilities import find_next_folder_nbr


matplotlib.use('Agg')   # Use 'Agg' backend to avoid memory overload


def convert_video_to_frames(INPUT_DATA_PATH, OUTPUT_PATH):
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
    EXTRACT_FRAME_INTERVAL = 1  # Extract every x frames
    # =======================================================

    VIDEO_LIST = [fn for fn in os.listdir(INPUT_DATA_PATH) if
                (fn.lower().endswith('.avi') or fn.lower().endswith('.mpg')or fn.lower().endswith('.mp4'))]
    #NB_PATIENTS = len(VIDEO_LIST)
    print("Number of videos to convert: ", len(VIDEO_LIST))
    NB_PATIENTS = 1

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
        videos_for_patient = [fn for fn in VIDEO_LIST]

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
                plt.close(fig)

            # Close reader before moving (video) files
            vid_reader.close()


def crop_and_scale_the_frames(dataset_type, path_to_patients):
    """ Crops the frames such that only the frame from the virtual video of the airway
    is stored and passed into the network. Overwrites the frames by storing the cropped
    frame as the frame """

    # Crop the frame by specifications from the virtual dataset
    if dataset_type == 'virtual':
        x_start = 538
        y_start = 107
        x_end = 1364
        y_end = 1015

        # Go through all persons and theirs sequences to get every frame
        for patient in os.listdir(path_to_patients):
            print("Going through patient: ", patient)
            path_to_sequences = path_to_patients + "/" + patient

            for sequence in os.listdir(path_to_sequences):
                print("Going through sequence: ", sequence)
                path_to_frames = path_to_sequences + "/" + sequence

                for frame in os.listdir(path_to_frames):
                    path_to_frame = path_to_frames + "/" + frame
                    frame_cropped = cv2.imread(path_to_frame)

                    # Crop the frame
                    #frame_cropped = frame[y_start:y_end, x_start:x_end]

                    cv2.imshow("Original", frame_cropped)
                    # Scale the frame down to (227 x 206)
                    frame_scaled = cv2.resize(frame_cropped, (0, 0), fx=0.25, fy=0.25)
                    cv2.imshow("Cropped", frame_scaled)
                    print("Shape: ", frame_scaled.shape)
                    cv2.waitKey(0)

                    # Save the new frame
                    #cv2.imwrite(path_to_frame, frame_scaled)


