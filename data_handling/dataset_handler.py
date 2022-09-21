import os

from data_handling.frames_handler import *
from parameters import dataset_type


class Dataset:
    def __init__(self):
        self.input_data_path = f"/Users/ikolderu/PycharmProjects/video_tracking_bronchoscopy/data_handling/data/{dataset_type}/{dataset_type}_videos/"
        self.output_data_path = f"/Users/ikolderu/PycharmProjects/video_tracking_bronchoscopy/data_handling/data/{dataset_type}/{dataset_type}_frames"
        self.create_dataset()

    def create_dataset(self):
        # Create frames from video if no frames exist
        if len(os.listdir(self.output_data_path)) == 0:
            print("Converting frames..")
            convert_video_to_frames(self.input_data_path, self.output_data_path)
        else:
            print("Frames exist! No converting necessary")

        # Crop the frames if necessary
        crop_and_scale_the_frames(dataset_type, self.output_data_path)
