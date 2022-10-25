"""
File for setting parameters for the project
"""
# Dataset & Preprocessing
dataset_type = 'phantom'  # {'virtual', 'phantom', 'human'}
test_split = 0.1
validation_split = 0.2    # {0.1, 0.2, 0.3}
frame_dimension = (256, 256)  # Dimension of the cropped frames that will be sent into CNN
fps = 10

# Data paths
root_directory_path = f"/Users/ikolderu/PycharmProjects/video_tracking_bronchoscopy/data_handling/data/{dataset_type}"
videos_path = root_directory_path + f"/{dataset_type}_videos/"
frames_path = root_directory_path + f"/{dataset_type}_frames"
label_file_path = root_directory_path + f"/{dataset_type}_branches_positions_numbers.txt"
names_file_path = root_directory_path + f"/{dataset_type}_branch_number_name.txt"

# CNN
network_type = "segment_det_net"  # {"direction_det_net", "segment_det_net"}
# output_shapes = [256, 128, 64, 32, 16, 8, 4, 1]
epochs = 5
batch_size = 32
learning_rate = 1e-2
early_stop_count = 5

# Visualization
