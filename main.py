from parameters import *
from data_handling.dataset_handler import *
from torchvision import transforms

from data_handling.utils.dataset_utils import create_csv_files_for_datasets, compute_mean_std, create_datasets_for_a_given_fps
from neural_net_handling.network_architectures.segment_net import SegmentDetNet
from neural_net_handling.network_architectures.direction_net import DirectionDetNet
from neural_net_handling.train_network import Trainer, create_plots, compute_loss_and_accuracy
from neural_net_handling.utils.neural_net_utilities import plot_predictions_test_set, compute_f1_score, plot_confusion_matrix


def main():
    """ The function running the entire pipeline of the project """
    random.seed(0)
    # ---------------- PREPROCESS ----------------------------------------------
    # Create frames, label them and preprocess them
    #convert_video_to_frames(videos_path, frames_path)
    crop_scale_and_label_the_frames(dataset_type, network_type, frames_path)

    # ---------------- DATASET ----------------------------------------------
    # Create a csv file for all frames for train dataset and test dataset
    create_csv_files_for_datasets(dataset_path, train_dataset_path, test_dataset_path, test_split, network_type)
    """"
    # Create train and test dataset
    train_dataset = BronchusDataset(
        csv_file=train_dataset_path,
        root_directory=root_directory_path,
        network_type=network_type,
        num_bronchus_generations=num_bronchus_generations,
        transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=train_mean, std=train_std)
        ]),
    )

    # Create train dataset
    test_dataset = BronchusDataset(
        csv_file=test_dataset_path,
        root_directory=root_directory_path,
        network_type=network_type,
        num_bronchus_generations=num_bronchus_generations,
        transform=transforms.Compose([
            transforms.ToTensor(),
            #transforms.Normalize(mean=test_mean, std=test_std)
        ]),
    )

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
        train_dataloaders,
        network_type,
        fps,
    )
    trainer.train()

    # ---------------- TESTING ----------------------------------------------
    # Load neural net model
    #print("loading best model.. ")
    #best_model = trainer.load_best_model()

    # Visualize training
    create_plots(trainer, train_plot_path, train_plot_name)

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

    # Plot confusion matrix
    print("plotting confusion matrix")
    plot_confusion_matrix(test, trainer, confusion_matrix_path)

    print("plotting test images")
    # Plot test images with predicted and original label on it
    plot_predictions_test_set(test, trainer, test_plot_path, network_type)

    # F1 score
    print("computing f1 score..")
    compute_f1_score(test, trainer)
    """

if __name__ == "__main__":
    main()
