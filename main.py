from parameters import *
from data_handling.dataset_handler import *
from torchvision import transforms

from data_handling.utils.dataset_utils import create_csv_files_for_datasets

from neural_net_handling.network_architectures.segment_net import SegmentDetNet
from neural_net_handling.network_architectures.direction_net import DirectionDetNet
from neural_net_handling.train_network import Trainer, create_plots, compute_loss_and_accuracy
from neural_net_handling.utils.neural_net_utilities import plot_predictions_test_set


def main():
    """ The function running the entire pipeline of the project """
    random.seed(0)
    # ---------------- PREPROCESS ----------------------------------------------
    # Create frames, label them and preprocess them
    #convert_video_to_frames(videos_path, frames_path)
    #crop_scale_and_label_the_frames(dataset_type, network_type, frames_path)

    # ---------------- DATASET ----------------------------------------------
    # Create a csv file for all frames for train dataset and test dataset
    create_csv_files_for_datasets(dataset_path, train_dataset_path, test_dataset_path, test_split)

    # Create train and test dataset
    train_dataset = BronchusDataset(
        csv_file=train_dataset_path,
        root_directory=root_directory_path,
        network_type=network_type,
        num_bronchus_generations=num_bronchus_generations,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))

    # Create train dataset
    test_dataset = BronchusDataset(
        csv_file=test_dataset_path,
        root_directory=root_directory_path,
        network_type=network_type,
        num_bronchus_generations=num_bronchus_generations,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))

    # Load train and test dataset
    train_dataloaders = train_dataset.get_train_dataloaders(batch_size, validation_split)
    test_dataloader = test_dataset.get_test_dataloaders(batch_size)

    # ---------------- ARTIFICIAL NEURAL NETWORK ----------------------------------------------
    # Create a CNN model
    if network_type == "segment_det_net":
        num_classes = train_dataset.get_num_classes()
        neural_net = SegmentDetNet(num_classes)

    elif network_type == "direction_det_net":
        neural_net = DirectionDetNet()

    else:
        print("Neural network type not set")
        neural_net = None

    # ---------------- TRAINING ----------------------------------------------
    # Train the CNN model
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        neural_net,
        train_dataloaders
    )
    trainer.train()

    # Visualize training
    create_plots(trainer, plot_name)

    # ---------------- TESTING ----------------------------------------------
    # Load neural net model
    #best_model = trainer.load_best_model()

    # Split the datasets in train, test and validation
    train, validation = train_dataloaders
    test = test_dataloader

    # Test CNN model
    print("---- TRAINING ----")
    train_loss, train_acc = compute_loss_and_accuracy(train, neural_net, torch.nn.CrossEntropyLoss())
    print("---- VALIDATION ----")
    val_loss, val_acc = compute_loss_and_accuracy(validation, neural_net, torch.nn.CrossEntropyLoss())
    print("---- TEST ----")
    test_loss, test_acc = compute_loss_and_accuracy(test, neural_net, torch.nn.CrossEntropyLoss())

    # Plot test images with predicted and original label on it
    plot_predictions_test_set(test, trainer)


if __name__ == "__main__":
    main()
