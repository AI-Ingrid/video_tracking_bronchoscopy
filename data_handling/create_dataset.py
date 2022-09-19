from data_handling.video_to_frames import *


class VirtualDataset:
    def __init__(self):
        self.input_data_path = "/Users/ikolderu/Documents/Skole/Prosjektoppgave/video_tracking_bronchoscopy/data_handling/data/virtual_videos/"
        self.output_data_path = "/Users/ikolderu/Documents/Skole/Prosjektoppgave/video_tracking_bronchoscopy/data_handling/data/virtual_frames"
    
    def create_dataset(self):
        convert_video_to_frames(self.input_data_path, self.output_data_path)
