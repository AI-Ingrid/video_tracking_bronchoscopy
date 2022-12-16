import pandas as pd
from parameters import root_directory_path, dataset_type, fps
import os
import random
import torch
from tqdm import tqdm


def get_num_videos():
    path = root_directory_path + f"/{dataset_type}_videos"
    videos = [file for file in os.listdir(path) if file.endswith(".mp4")]
    return len(videos)


def get_3_digit_string(video_indices):
    video_indices_string = []
    for element in video_indices:
        if len(element) == 1:
            element = '00' + element
        elif len(element) == 2:
            element = '0' + element
        video_indices_string.append(element)
    return video_indices_string


def create_csv_files_for_datasets(dataset_path, train_dataset_path, test_dataset_path, test_split, network_type):
    # Fetch the original dataset file
    dataset = pd.read_csv(dataset_path)

    # Find out how many videos the dataset contains
    num_videos = get_num_videos()
    num_test_videos = int(test_split * num_videos)

    # Get random videos
    test_video_indices = [str(random.randint(0, num_videos)) for i in range(num_test_videos)]

    # Get the train video indices
    dataset_indices = [str(index) for index in range(num_videos)]
    train_video_indices = [index for index in dataset_indices if index not in test_video_indices]

    test_video_indices_string = get_3_digit_string(test_video_indices)
    train_video_indices_string = get_3_digit_string(train_video_indices)

    test_videos = []
    train_videos = []

    # Create dataframe for every test video
    for test_video in test_video_indices_string:
        # Add test videos to temp dataframe
        if network_type == 'segment_det_net':
            temp_test_dataframe = dataset[dataset['Frame'].str.contains(f'Sequence_{test_video}')]

        elif network_type == 'direction_det_net':
            temp_test_dataframe = dataset[dataset['Frame_1'].str.contains(f'Sequence_{test_video}')]

        # Store the temporary dataframes
        test_videos.append(temp_test_dataframe)

    # Create dataframe for every train video
    for train_video in train_video_indices_string:
        # Add test videos to temp dataframe
        if network_type == 'segment_det_net':
            temp_train_dataframe = dataset[dataset['Frame'].str.contains(f'Sequence_{train_video}')]

        elif network_type == 'direction_det_net':
            temp_train_dataframe = dataset[dataset['Frame_1'].str.contains(f'Sequence_{train_video}')]

        # Store the temporary dataframes
        train_videos.append(temp_train_dataframe)

    # Create the final dataset by concatenating the dataframes in the array
    test_dataset = pd.concat(test_videos).reset_index(drop=True)
    train_dataset = pd.concat(train_videos).reset_index(drop=True)

    # Save dataframes as csv files
    test_dataset.to_csv(test_dataset_path, index=False)
    train_dataset.to_csv(train_dataset_path, index=False)


def get_class_distribution_for_batch(y_batch, count):
    for label in y_batch:
        if not count:
            count[int(label)] = 1
        else:
            labels = list(count.keys())
            if label not in labels:
                count[int(label)] = 1

            else:
                count[int(label)] += 1
    return count


def get_class_distribution(dataloader):
    count = {}
    for x_batch, y_batch in dataloader:
        count = get_class_distribution_for_batch(y_batch, count)
    return count


def compute_mean_std(dataset, dataloader, frame_dim):
    # placeholders
    psum = torch.tensor([0.0, 0.0, 0.0])
    psum_sq = torch.tensor([0.0, 0.0, 0.0])

    # loop through images
    for datatype in dataloader:
        for X_batch, Y_batch in tqdm(datatype):
            #psum += X_batch.sum(axis=[0, 2, 3])
            psum += X_batch.sum(axis=[0, 1, 2, 3])
            #psum_sq += (X_batch ** 2).sum(axis=[0, 2, 3])
            psum_sq += (X_batch ** 2).sum(axis=[0, 1, 2, 3])

    # pixel count
    count = len(dataset) * frame_dim[0] * frame_dim[1]

    # mean and std
    total_mean = psum / count
    total_var = (psum_sq / count) - (total_mean ** 2)
    total_std = torch.sqrt(total_var)

    # output
    print('mean: ' + str(total_mean))
    print('std:  ' + str(total_std))
    return total_mean, total_std

"""
def main():
    print("--- TRAIN ---")
    train_count = get_class_distribution(train)
    print(train_count)

    print("--- VALIDATION ---")
    validation_count = get_class_distribution(validation)
    print(validation_count)

    print("--- TEST ---")
    test_count = get_class_distribution(test)
    print(test_count)

    distribution_dict = {
        "train": train_count,
        "validation": validation_count,
        "test": test_count
    }
    df = pd.DataFrame.from_dict(distribution_dict)
    print(df.head())
    df.to_csv("distribution_of_classes.csv")
"""
