from torch import nn
from parameters import *
from random import seed
from data_handling.dataset_handler import *
from torchvision import transforms

from neural_net_handling.network_architectures.segment_net import SegmentDetNet
from neural_net_handling.network_architectures.direction_net import DirectionDetNet
from neural_net_handling.train_network import Trainer, create_plots, compute_loss_and_accuracy


def main():
    """ The function running the entire pipeline of the project """
    seed()

    # Create frames, label them and preprocess them
    convert_video_to_frames(videos_path, frames_path)
    crop_scale_and_label_the_frames(dataset_type, network_type, frames_path, root_directory_path)
    """
    # Create dataset
    data_transform = transforms.Compose([
        Rescale(256),
        RandomCrop(224),
        ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
        ])

    bronchoscopy_dataset = BronchoscopyDataset(
        csv_file=root_directory + "/" + dataset_type + "_dataset.csv",
        root_directory=root_directory,
        transform=data_transform)

    # Load dataset
    dataloaders = bronchoscopy_dataset.get_dataloaders(batch_size, test_split, validation_split)

    # Create a CNN model
    if network_type == "segment_det_net":
        neural_net = SegmentDetNet()

    elif network_type == "direction_det_net":
        neural_net = DirectionDetNet()

    else:
        print("Neural network type not set")
        neural_net = None

    # Train the CNN model
    trainer = Trainer(
        batch_size,
        learning_rate,
        early_stop_count,
        epochs,
        neural_net,
        dataloaders
    )
    trainer.train()

    # Visualize training
    create_plots(trainer, "Training")

    # Test CNN model
    train, validation, test = dataloaders
    print("---- TRAINING ----")
    train_loss, train_acc = compute_loss_and_accuracy(train, neural_net, nn.CrossEntropyLoss())
    print("---- VALIDATION ----")
    val_loss, val_acc = compute_loss_and_accuracy(validation, neural_net, nn.CrossEntropyLoss())
    print("---- TEST ----")
    test_loss, test_acc = compute_loss_and_accuracy(test, neural_net, nn.CrossEntropyLoss())
    """

if __name__ == "__main__":
    main()
