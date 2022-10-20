import os
import math
import imageio
import cv2

import pandas as pd
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm   # create progress bars, see sample usage below

from parameters import frame_dimension, dataset_type
from utils.video_utilities import get_trim_start_end_frames, VideoTrimmingLimits
from utils.misc_utilities import find_next_folder_nbr


matplotlib.use('Agg')   # Use 'Agg' backend to avoid memory overload


def convert_video_to_frames(INPUT_DATA_PATH, OUTPUT_PATH):
    """
    Code build upon work from author: Ingrid Tveten (SINTEF)
    """
    # Create frames from video if no frames exist
    if len(os.listdir(OUTPUT_PATH)) <= 1:
        print("Converting frames..")

        EXTRACT_FRAME_INTERVAL = 1  # Extract every x frames

        # Hard coded for now
        NB_PATIENTS = 1
        FILES_LIST = os.listdir(INPUT_DATA_PATH)

        # Create list of videos
        VIDEO_LIST = [fn for fn in FILES_LIST if
                    (fn.lower().endswith('.avi') or fn.lower().endswith('.mpg') or fn.lower().endswith('.mp4'))]

        # TODO: Dette kan være litt for hard koda for virtual dataset.. vi får se
        # Create list of tuples with labels and belonging video
        LABEL_LIST = []
        for video_file in VIDEO_LIST:
            video_name = video_file.split(".")[0]
            index = FILES_LIST.index(video_name + "_branching.txt" )
            LABEL_LIST.append((video_file, FILES_LIST[index]))

        for p in tqdm(range(NB_PATIENTS), 'Patient'):

            # Create ./Patient_XX directory
            next_patient_nbr = find_next_folder_nbr(dataset_dir=OUTPUT_PATH)
            patient_dir = os.path.join(OUTPUT_PATH, f'Patient_{next_patient_nbr:03d}')
            try:
                os.makedirs(patient_dir, exist_ok=False)
            except OSError as exc:
                print(f"OSError: Patient folder {patient_dir} probably already exists")
                exit(-1)

            videos_for_patient = [fn for fn in LABEL_LIST]

            # Generate sequences
            for (video_fn, label) in tqdm(videos_for_patient, 'Sequences'):

                # Create ./Patient_XX/Sequence_XX directory
                seq_nbr = find_next_folder_nbr(patient_dir)
                seq_dir = os.path.join(patient_dir, f'Sequence_{seq_nbr:03d}')
                try:
                    os.makedirs(seq_dir, exist_ok=False)
                except OSError as exc:
                    print(f"OSError: Sequence folder {seq_dir} probably already exists")
                    exit(-1)

                # Save the labels belonging to the video in the Sequence folder
                read_file = pd.read_csv(INPUT_DATA_PATH + "/" + label)
                read_file.to_csv(seq_dir + "/" + label, index=None)

                # Get full path to video file and read video data
                video_path = os.path.join(INPUT_DATA_PATH, video_fn)
                vid_reader = imageio.get_reader(video_path)
                metadata = vid_reader.get_meta_data()
                fps = metadata['fps']
                duration = metadata['duration']
                nb_frames = math.floor(metadata['fps'] * metadata['duration'])

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
        else:
            print("Frames exist! No converting necessary")


def add_labels(network_type, path_to_frames, path_to_branching_file, frame_counter, frames, dataframe):
    """A help function for crop_scale_and_label_frames that labels the frames in a specific
    sequence given from the path_to_frames parameter
    """
    # Create an array of the branching txt file
    branches_file = open(path_to_branching_file)
    branches_array = np.loadtxt(branches_file, delimiter=" ", dtype='int')

    if network_type == "direction_det_net":
        # Order the list of frames from the video
        print("before: ", frames)
        frames.sort()
        print("after: ", frames)
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
        # Find the ratio between num frames and timestamps from the video
        ratio = len(branches_array) / frame_counter

        # TODO: Have (frame_num - 3) * ratio to get 3 frames before bifurcation such that you can see the bifurcation
        # Create a label list with the correct label for a frame_num given by the list index
        labels = [branches_array[int(frame_num * ratio)] for frame_num in range(0, frame_counter)]

        # TODO: Go through labels again and set 5-10 frames before bifurcation true?
        # Match label and frames

        # TODO: Sort the frames  by path names, currently not the right order
        labeled_frames = list(zip(frames, labels))

        # TODO: Remove overwriting of csv file
        # Write array to csv file
        dataframe = pd.DataFrame(labeled_frames)
        path_to_dataset = f"/Users/ikolderu/PycharmProjects/video_tracking_bronchoscopy/data_handling/data/{dataset_type}/{dataset_type}_dataset.csv"
        dataframe.to_csv(path_to_dataset, header=False, index=False)

    else:
        print("No network type registered")
        # Save frames and label in a dataset file
        #file = open(f'f"/Users/ikolderu/PycharmProjects/video_tracking_bronchoscopy/data_handling/data/{dataset_type}/{dataset_type}_dataset_csv','w')
        #writer = csv.writer(file)

        #for index, frame in enumerate(frames):
            #writer.writerow(frame, labels[index])
        #file.close()

        # Save only the labels in a file
        #np.savetxt(path_to_frames + "/labels_for_each_frame.csv", labels, delimiter=",\n")



def crop_scale_and_label_the_frames(dataset_type, network_type, path_to_patients, root_directory_path):
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
            data_frame = pd.Dataframe(columns=["Frame 1", "Frame 2", "Frame 3", "Frame 4", "Frame 5", "Label"])

        elif network_type == "segment_det_net":
            data_frame = None
        else:
            print("No network type registered")
            data_frame = None

        # Go through all persons and theirs sequences to get every frame
        for patient in os.listdir(path_to_patients):
            # Avoid going through hidden directories
            if not patient.startswith("."):
                print("Going through patient: ", patient)
                path_to_sequences = path_to_patients + "/" + patient

                # Go through all video sequences
                for sequence in os.listdir(path_to_sequences):
                    # Avoid going through hidden directories
                    if not patient.startswith("."):
                        print("Going through sequence: ", sequence)
                        path_to_frames = path_to_sequences + "/" + sequence
                        path_to_branching_file = ""
                        frame_counter = 0
                        frame_list = []

                        # Go through every frame in a video sequence
                        for file in os.listdir(path_to_frames):

                            # Check for file being a frame
                            if file.endswith(".png"):
                                path_to_frame = path_to_frames + "/" + file
                                frame = cv2.imread(path_to_frame)

                                # Check if frame is not cropped and scaled before
                                if frame.shape[0] > frame_dimension[0]:

                                    # Crop the frame
                                    frame_cropped = frame[y_start:y_end, x_start:x_end]

                                    # Scale the frame down to frame_dimension set in parameters.py f.ex: (256, 256)
                                    frame_scaled = cv2.resize(frame_cropped, frame_dimension, interpolation=cv2.INTER_AREA)

                                    # Save the new frame
                                    cv2.imwrite(path_to_frame, frame_scaled)
                                    frame_list.append(path_to_frame)

                                else:
                                    print("Frame is already cropped")
                                    frame_list.append(path_to_frame)

                                # Count frames in sequence
                                frame_counter += 1

                            # File is the branching.txt file
                            elif file.endswith("branching.txt"):
                                path_to_branching_file = path_to_frames + "/" + file

                        # Add labels to the frames in current video
                        add_labels(network_type, path_to_frames, path_to_branching_file, frame_counter, frame_list, data_frame)

        # Convert data_frame into CSV file in order to store the dataset as a file
        dataset_path = root_directory_path + "/" + dataset_type + "_" + network_type + "_dataset.csv"
        data_frame.to_csv(dataset_path, ignore_index=True)
