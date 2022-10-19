"""
File for setting parameters for the project
"""

# Dataset & Preprocessing
dataset_type = 'phantom'  # {'virtual', 'phantom', 'human'}

root_directory = f"/Users/ikolderu/PycharmProjects/video_tracking_bronchoscopy/data_handling/data/{dataset_type}"
videos_path = f"/Users/ikolderu/PycharmProjects/video_tracking_bronchoscopy/data_handling/data/{dataset_type}/{dataset_type}_videos/"
frames_path = f"/Users/ikolderu/PycharmProjects/video_tracking_bronchoscopy/data_handling/data/{dataset_type}/{dataset_type}_frames"

frame_dimension = (256, 256)  # Dimension of the cropped frames that will be sent into CNN

# CNN
network_type = "direction_det_net"  # {"direction_det_net", "segment_det_net"}
output_shapes = [256, 128, 64, 32, 16, 8, 4, 1]
epochs = 5
batch_size = 32
learning_rate = 1e-2
early_stop_count = 5

# Visualization

