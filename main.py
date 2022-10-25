from parameters import *
from data_handling.dataset_handler import *
from torchvision import transforms

from neural_net_handling.network_architectures.segment_net import SegmentDetNet
from neural_net_handling.network_architectures.direction_net import DirectionDetNet
from neural_net_handling.train_network import Trainer, create_plots, compute_loss_and_accuracy


def main():
    """ The function running the entire pipeline of the project """
    # Create frames, label them and preprocess them
    convert_video_to_frames(videos_path, frames_path)
    #crop_scale_and_label_the_frames(dataset_type, network_type, frames_path)

    # Create dataset
    # TODO: Add RandomCrop? Normalize?
    data_transform = transforms.Compose([
        ToTensor(),
        ])

    bronchoscopy_dataset = BronchoscopyDataset(
        csv_file=root_directory_path + f"/{dataset_type}_{network_type}_dataset.csv",
        root_directory=root_directory_path,
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
    train, validation, test = dataloaders
    """
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
