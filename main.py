from parameters import *
from data_handling.dataset_handler import *
from torchvision import transforms, datasets

from neural_net_handling.network_architectures.segment_net import SegmentDetNet
from neural_net_handling.network_architectures.direction_net import DirectionDetNet
from neural_net_handling.train_network import Trainer, create_plots, compute_loss_and_accuracy


def main():
    """ The function running the entire pipeline of the project """

    # ---------------- TRAINNG ----------------------------------------------
    random.seed(1)
    # Create frames, label them and preprocess them
    #convert_video_to_frames(videos_path, frames_path)
    #crop_scale_and_label_the_frames(dataset_type, network_type, frames_path)

    # Create dataset
    bronchus_dataset = BronchusDataset(
        csv_file=root_directory_path + f"/{dataset_type}_{network_type}_dataset.csv",
        root_directory=root_directory_path,
        network_type=network_type,
        num_bronchus_generations=num_bronchus_generations,
        transform=transforms.Compose([
            transforms.ToTensor()
        ]))

    # Load dataset
    dataloaders = bronchus_dataset.get_dataloaders(batch_size, test_split, validation_split)

    # Create a CNN model
    if network_type == "segment_det_net":
        num_classes = bronchus_dataset.get_num_classes()
        neural_net = SegmentDetNet(num_classes)

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
    #trainer.train()
    # Load neural net model
    trainer.load_best_model()
    train, validation, test = dataloaders

    # Test CNN model
    print("---- TRAINING ----")
    train_loss, train_acc = compute_loss_and_accuracy(train, trainer.model, torch.nn.CrossEntropyLoss())
    print("---- VALIDATION ----")
    val_loss, val_acc = compute_loss_and_accuracy(validation, trainer.model, torch.nn.CrossEntropyLoss())
    print("---- TEST ----")
    test_loss, test_acc = compute_loss_and_accuracy(test, trainer.model, torch.nn.CrossEntropyLoss())
    # ---------------- TESTING ----------------------------------------------
    # Visualize training
    create_plots(trainer, "bs_16_gen_4_second_try")


if __name__ == "__main__":
    main()
