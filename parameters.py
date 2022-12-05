"""
File for setting parameters for the project
"""
# Dataset & Preprocessing
dataset_type = 'phantom'  # {'virtual', 'phantom', 'human'}
test_split = 0.2
validation_split = 0.1
frame_dimension = (256, 256)  # Dimension of the cropped frames that will be sent into CNN
fps = 10

# CNN
network_type = "direction_det_net"  # {"direction_det_net", "segment_det_net"}
num_bronchus_generations = None  # {None, 1, 2, 3, 4}
path_to_trained_models = f"neural_net_handling/trained_models/{network_type}_{num_bronchus_generations}_gens.pt"
epochs = 5
batch_size = 32
learning_rate = 7e-5
early_stop_count = 5
alpha = 0.25
gamma = 2.0


# Data paths
root_directory_path = f"data_handling/data/{dataset_type}"
videos_path = root_directory_path + f"/{dataset_type}_videos/"
frames_path = root_directory_path + f"/{dataset_type}_frames"
label_file_path = root_directory_path + f"/labeling_info/{dataset_type}_branches_positions_numbers.txt"
names_file_path = root_directory_path + f"/labeling_info/{dataset_type}_branch_number_name.txt"
dataset_path = root_directory_path + f"/raw_data/{dataset_type}_{network_type}_data.csv"
test_dataset_path = root_directory_path + f"/datasets/test/{dataset_type}_{network_type}_test_dataset.csv"
train_dataset_path = root_directory_path + f"/datasets/train/{dataset_type}_{network_type}_train_dataset.csv"


# Visualization
train_plot_path = f"data_handling/plots/training/{network_type}/"
train_plot_name = f"{network_type}_batchsize_{batch_size}_epochs_{epochs}_generations_{num_bronchus_generations}"
test_plot_path = f"data_handling/plots/testing/{network_type}/"
