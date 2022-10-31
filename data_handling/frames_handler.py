import os
import math
import imageio
import cv2
import re

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm   # create progress bars, see sample usage below
from scipy.spatial.distance import cdist

import parameters
from utils.video_utilities import get_trim_start_end_frames, VideoTrimmingLimits
from utils.misc_utilities import find_next_folder_nbr


matplotlib.use('Agg')   # Use 'Agg' backend to avoid memory overload


def convert_video_to_frames(input_data_path, output_data_path):
    """
    Code build upon work from author: Ingrid Tveten (SINTEF)
    """
    # Create frames from video if no frames exist
    if len(os.listdir(output_data_path)) <= 1:  # != 0
        print("Converting frames..")

        extract_frame_interval = 1  # Extract every x frames

        # Hard coded for now
        nb_patients = 1
        files_list = os.listdir(input_data_path)

        # Create list of videos
        video_list = [fn for fn in files_list if
                    (fn.lower().endswith('.avi') or fn.lower().endswith('.mpg') or fn.lower().endswith('.mp4'))]

        # Create list of tuples with video, belonging timestamps and positions
        label_list = []
        video_list.sort()
        for video_file in video_list:
            video_name = video_file.split(".")[0]
            timestamps_index = files_list.index(video_name + "_timestamps.txt")
            position_index = files_list.index(video_name + "_positions.txt")
            label_list.append((video_file, files_list[timestamps_index], files_list[position_index]))

        for p in tqdm(range(nb_patients), 'Patient'):
            # Create ./Patient_XX directory
            next_patient_nbr = find_next_folder_nbr(dataset_dir=output_data_path)
            patient_dir = os.path.join(output_data_path, f'Patient_{next_patient_nbr:03d}')
            try:
                os.makedirs(patient_dir, exist_ok=False)
            except OSError as exc:
                print(f"OSError: Patient folder {patient_dir} probably already exists")
                exit(-1)

            videos_for_patient = [fn for fn in label_list]

            # Generate sequences
            for (video_fn, timestamp, position) in tqdm(videos_for_patient, 'Sequences'):

                # Create ./Patient_XX/Sequence_XX directory
                seq_nbr = find_next_folder_nbr(patient_dir)
                seq_dir = os.path.join(patient_dir, f'Sequence_{seq_nbr:03d}')
                try:
                    os.makedirs(seq_dir, exist_ok=False)
                except OSError as exc:
                    print(f"OSError: Sequence folder {seq_dir} probably already exists")
                    exit(-1)

                # Save the timestamps belonging to the video in the correct Sequence folder
                read_timestamp_file = pd.read_csv(input_data_path + "/" + timestamp)
                read_timestamp_file.to_csv(seq_dir + "/" + timestamp, index=None)

                # Save the positions belonging to the video in the correct Sequence folder
                read_positions_file = pd.read_csv(input_data_path + "/" + position)
                read_positions_file.to_csv(seq_dir + "/" + position, index=None)

                # Get full path to video file and read video data
                video_path = os.path.join(input_data_path, video_fn)
                vid_reader = imageio.get_reader(video_path)
                metadata = vid_reader.get_meta_data()
                FPS = metadata['fps']
                # Update fps in parameters.py
                parameters.fps = FPS
                duration = metadata['duration']
                nb_frames = math.floor(metadata['fps'] * metadata['duration'])

                trim_time = VideoTrimmingLimits(t1=0., t2=duration)
                start_frame, end_frame = get_trim_start_end_frames(trim_time, FPS, nb_frames)

                # Loop through the frames of the video
                for frnb, fr in enumerate(tqdm(range(start_frame, end_frame, int(extract_frame_interval)), 'Frames')):
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
    else:
        print("Frames exist! No converting necessary")


def get_positions_from_video(path_to_position_file):
    """ Help function that reads positions from a position file and returns an array with tuples
    representing a position (x_pos, y_pos, z_pos). The position file is complex and uses 3 lines to
    express one position. So this function extracts only the necessary variables
    x, y and z from three different lines and stores it as a tuple in an array
    """
    # Find position file for video
    position_file = open(path_to_position_file)

    x_pos = None
    y_pos = None
    z_pos = None
    positions = []

    # Go through all lines in position file
    for line in position_file:
        # Find position value from line in position file
        new_pos = line.split(" ")[3]
        # Set x_pos
        if x_pos is None:
            x_pos = new_pos

        # Set y_pos
        elif y_pos is None:
            y_pos = new_pos

        # Set z_pos
        elif z_pos is None:
            z_pos = new_pos
            # Add total position (x_pos, y_pos, z_pos) to list of positions
            positions.append((float(x_pos), float(y_pos), float(z_pos)))

            # Reset position
            x_pos = None
            y_pos = None
            z_pos = None

    position_file.close()
    return positions


def get_possible_positions_and_its_labels():
    # Create a function for reading branches_position_numbers.txt file
    positions_labels_file = open(parameters.label_file_path)
    lines = positions_labels_file.readlines()

    is_first_line = True
    positions_and_labels = {}

    for line in lines:
        # Handle columns names in first line
        if is_first_line:
            is_first_line = False
            continue

        # Handle last empty line
        elif line == "\n":
            break

        # Fetch only the necessary info from the line
        label = int(line.split(";")[0])
        position = line.split(";")[4]

        x_pos = float(position.split(",")[0])
        y_pos = float(position.split(",")[1])
        z_pos = float(position.split(",")[2])

        # Add position (x_pos, y_pos, z_pos) to dictionary with the belonging label as value
        positions_and_labels[(x_pos, y_pos, z_pos)] = label

    positions_labels_file.close()
    return positions_and_labels


def match_frames_with_positions_and_timestamps(positions, path_to_timestamp_file, frames, positions_and_labels, dataframe):
    # Get frame sampling ratio in ms
    frame_sampling_ratio = 1/parameters.fps * 1000

    # Get timestamps
    timestamp_file = open(path_to_timestamp_file)
    timestamp_list = np.loadtxt(timestamp_file, delimiter=" ", dtype='int')
    timestamp = 0

    # Check for errors
    if len(timestamp_list) != len(positions):
        print("Num timestamps and num positions does not match")

    #
    possible_positions = positions_and_labels.keys()
    possible_positions_2D_array = map(np.array, possible_positions)
    possible_positions_2D_array = np.array(list(possible_positions_2D_array))

    # Find label for every frame
    for frame_index, frame in enumerate(frames):
        # Find nearest timestamp
        timestamp_array = np.asarray(timestamp_list)
        nearest_timestamp_index = (np.abs(timestamp_array - timestamp)).argmin()
        nearest_timestamp = timestamp_array[nearest_timestamp_index]
        #print("Nearest timestamp: ", nearest_timestamp)

        # Get position from the nearest timestamp
        nearest_position = positions[nearest_timestamp_index]
        nearest_position_list = [float(nearest_position[0]), float(nearest_position[1]), float(nearest_position[2])]
        #print("Nearest position: ", nearest_position)

        # Match the nearest position with a possible position to get a label
        best_match_index = cdist([nearest_position_list], possible_positions_2D_array).argmin()
        best_match_position = possible_positions_2D_array[best_match_index]

        # Find label
        best_match_position_tuple = (best_match_position[0], best_match_position[1], best_match_position[2])
        label = positions_and_labels[best_match_position_tuple]

        # Add the frame (or frame path?) and label to dataframe
        labeled_frame = pd.DataFrame({
            "Frame": frame,
            "Label": label,
        }, index=[0])
        # Store labeled frame in dataframe
        new_dataframe = pd.concat([labeled_frame, dataframe.loc[:]]).reset_index(drop=True)
        dataframe = new_dataframe

        # Set timestamp to its next value for the next iteration in the loop
        timestamp += frame_sampling_ratio
    return new_dataframe


# The function: atoi and natural_keys are taken from Stackoverflow in order to sort a
# string that contains integers the correct way
def atoi(text):
    """
    Code from Stackoverflow: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    """
    return int(text) if text.isdigit() else text


def natural_keys(text):
    '''
    Code from Stackoverflow: https://stackoverflow.com/questions/5967500/how-to-correctly-sort-a-string-with-a-number-inside
    '''
    return [atoi(c) for c in re.split(r'(\d+)', text)]


def add_labels(network_type, path_to_timestamp_file, path_to_position_file, frames, dataframe):
    """A help function for crop_scale_and_label_frames that labels the frames in a specific
    sequence given from the path_to_frames parameter
    """
    # Sort the list of frames from the video
    frames.sort(key=natural_keys)
    if network_type == "direction_det_net":
        # TODO: Maybe 5 frames is to little?
        # TODO put this in a help function?
        for index, frame in enumerate(frames[5:]):
            # Add a new forward row
            new_forward_row = pd.DataFrame({
                "frame 1": frames[index-4],
                "frame 2": frames[index-3],
                "frame 3": frames[index-2],
                "frame 4": frames[index-1],
                "frame 5": frames[index],
                "label": 1  # Forward,
            })
            new_backward_row = pd.DataFrame({
                "frame 1": frames[index],
                "frame 2": frames[index-1],
                "frame 3": frames[index-2],
                "frame 4": frames[index-3],
                "frame 5": frames[index-4],
                "label": 0  # Backward,
            })
            dataframe = dataframe.append(new_forward_row, ignore_index=True)
            dataframe = dataframe.append(new_backward_row, ignore_index=True)
            print(dataframe.head())

    elif network_type == "segment_det_net":
        # Get all positions in current video
        positions = get_positions_from_video(path_to_position_file)

        # Get all possible positions and its labels
        positions_and_labels = get_possible_positions_and_its_labels()

        # Match positions from video with possible positions to label the positions in video
        dataframe = match_frames_with_positions_and_timestamps(positions, path_to_timestamp_file, frames, positions_and_labels, dataframe)

    else:
        print("No network type registered")

    return dataframe


def crop_scale_and_label_the_frames(dataset_type, network_type, path_to_patients):
    """ Crops the frames such that only the frame from the virtual video of the airway
    is stored and passed into the network. Overwrites the frames by storing the cropped
    frame as the frame """

    # Crop the frame by specifications from the virtual dataset
    if dataset_type == 'virtual' or dataset_type == 'phantom':
        x_start = 538
        y_start = 107
        x_end = 1364
        y_end = 1015

        if network_type == "direction_det_net":
            dataframe = pd.DataFrame(columns=["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Frame 5", "Label"])

        elif network_type == "segment_det_net":
            dataframe = pd.DataFrame(columns=["Frame", "Label"])
        else:
            print("No network type registered")
            dataframe = None

        # Go through all persons and theirs sequences to get every frame
        for patient in os.listdir(path_to_patients):
            # Avoid going through hidden directories like .DS_Store
            if not patient.startswith("."):
                print("Going through patient: ", patient)
                path_to_sequences = path_to_patients + "/" + patient

                # Go through all video sequences
                for sequence in os.listdir(path_to_sequences):
                    # Avoid going through hidden directories like .DS_Store
                    if not sequence.startswith("."):
                        print("Going through sequence: ", sequence)
                        path_to_frames = path_to_sequences + "/" + sequence
                        path_to_timestamp_file = ""
                        path_to_position_file = ""
                        frame_list = []

                        # Go through every frame in a video sequence
                        for file in os.listdir(path_to_frames):
                            # Avoid going through hidden directories like .DS_Store
                            if not file.startswith("."):
                                # Check for file being a frame
                                if file.endswith(".png"):
                                    path_to_frame = path_to_frames + "/" + file
                                    frame = cv2.imread(path_to_frame)

                                    # Check if frame is not cropped and scaled before
                                    if frame.shape[0] > parameters.frame_dimension[0]:

                                        # Crop the frame
                                        frame_cropped = frame[y_start:y_end, x_start:x_end]

                                        # Scale the frame down to frame_dimension set in parameters.py f.ex: (256, 256)
                                        frame_scaled = cv2.resize(frame_cropped, parameters.frame_dimension, interpolation=cv2.INTER_AREA)

                                        # Save the new frame
                                        cv2.imwrite(path_to_frame, frame_scaled)
                                        frame_list.append(path_to_frame)

                                    else:
                                        #print("Frame is already cropped")
                                        frame_list.append(path_to_frame)

                                # File is the timestamp.txt file
                                elif file.endswith("timestamps.txt"):
                                    path_to_timestamp_file = path_to_frames + "/" + file

                                # File is position.txt file
                                elif file.endswith("positions.txt"):
                                    path_to_position_file = path_to_frames + "/" + file

                        # Add labels to the frames in current video
                        dataframe = add_labels(network_type, path_to_timestamp_file, path_to_position_file, frame_list, dataframe)
                        print("Converting dataframe to csv file..")
                        # Convert dataframe into CSV file in order to store the dataset as a file
                        dataset_path = parameters.root_directory_path + f"/{parameters.dataset_type}_{parameters.network_type}_dataset.csv"
                        dataframe.to_csv(dataset_path, index=False)
