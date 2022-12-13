import typing
import time
import collections
import torchvision

import neural_net_handling.utils.neural_net_utilities as utils
import pathlib
import numpy as np
import matplotlib.pyplot as plt
from parameters import alpha, gamma, network_type
import torchmetrics.classification as tm
import torch
from tqdm import tqdm


def compute_loss_and_accuracy(
        dataloader: torch.utils.data.DataLoader,
        model: torch.nn.Module,
        loss_criterion: torch.nn.modules.loss._Loss):
    """
    Computes the average loss and the accuracy over the whole dataset
    in dataloader.
    Args:
        dataloder: Validation/Test dataloader
        model: torch.nn.Module
        loss_criterion: The loss criterion, e.g: torch.nn.CrossEntropyLoss()
    Returns:
        [average_loss, accuracy]: both scalar.
    """
    average_loss = 0
    f1 = 0
    num_samples = 0
    batch_size = 0
    if network_type == "segment_det_net":
        f1_metric = tm.F1Score(average='macro', task='multilabel', num_classes=27)
    else:
        f1_metric = tm.F1Score(average='macro', task='multilabel', num_classes=2)
    with torch.no_grad():
        for (X_batch, Y_batch) in tqdm(dataloader):
            # Transfer images/labels to GPU VRAM, if possible
            X_batch = utils.to_cuda(X_batch)
            Y_batch = utils.to_cuda(Y_batch)

            # Forward pass the images through our model
            output_probs = model(X_batch)

            predictions = torch.softmax(output_probs, dim=1)
            decoded_Y_batch = utils.decode_one_hot_encoded_labels(Y_batch)
            targets = torch.tensor(np.array(decoded_Y_batch))

            predictions = utils.to_cuda(predictions)
            targets = utils.to_cuda(targets)

            num_samples += Y_batch.shape[0]

            # Compute F1 Score
            f1 += f1_metric(predictions, targets)

            # Compute Loss
            average_loss += loss_criterion(output_probs, Y_batch)
            batch_size += 1

    average_loss = average_loss / batch_size
    f1 = f1 / batch_size
    print(f'F1 score: {f1}')
    print(f'Loss: {average_loss}')
    return average_loss, f1


class Trainer:

    def __init__(self,
                 batch_size: int,
                 learning_rate: float,
                 early_stop_count: int,
                 epochs: int,
                 model: torch.nn.Module,
                 dataloaders: typing.List[torch.utils.data.DataLoader],
                 network_type: str):
        """
            Initialize our trainer class.
        """
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.early_stop_count = early_stop_count
        self.epochs = epochs
        self.network_type = network_type

        # Since we are doing multi-class classification, we use CrossEntropyLoss
        self.loss_criterion = torch.nn.CrossEntropyLoss()

        # Initialize the model
        self.model = model

        # Transfer model to GPU VRAM, if possible.
        self.model = utils.to_cuda(self.model)

        # Define our optimizer.
        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          self.learning_rate)

        # Load our dataset
        self.dataloader_train, self.dataloader_val = dataloaders

        # Validate our model everytime we pass through 25% of the dataset
        self.num_steps_per_val = len(self.dataloader_train) // 4
        self.global_step = 0
        self.start_time = time.time()

        # Tracking variables
        self.train_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict()

        )
        self.validation_history = dict(
            loss=collections.OrderedDict(),
            accuracy=collections.OrderedDict()
        )
        self.checkpoint_dir = pathlib.Path(f"checkpoints_{self.network_type}")

    def validation_step(self):
        """
            Computes the loss/accuracy for all three datasets.
            Train, validation and test.
        """
        self.model.eval()

        training_loss, training_acc = compute_loss_and_accuracy(self.dataloader_train, self.model, self.loss_criterion)
        self.train_history["accuracy"][self.global_step] = training_acc

        validation_loss, validation_acc = compute_loss_and_accuracy(self.dataloader_val, self.model, self.loss_criterion)
        self.validation_history["loss"][self.global_step] = validation_loss
        self.validation_history["accuracy"][self.global_step] = validation_acc

        used_time = time.time() - self.start_time
        print(
            f"Epoch: {self.epoch:>1}",
            f"Batches per seconds: {self.global_step / used_time:.2f}",
            f"Global step: {self.global_step:>6}",
            f"Validation Loss: {validation_loss:.2f}",
            f"Validation Accuracy: {validation_acc:.3f}",
            f"Train Accuracy: {training_acc:.3f}",
            sep=", ")
        self.model.train()

    def should_early_stop(self):
        """
            Checks if validation loss doesn't improve over early_stop_count epochs.
        """
        # Check if we have more than early_stop_count elements in our validation_loss list.
        val_loss = self.validation_history["loss"]
        if len(val_loss) < self.early_stop_count:
            return False
        # We only care about the last [early_stop_count] losses.
        relevant_loss = list(val_loss.values())[-self.early_stop_count:]
        first_loss = relevant_loss[0]
        if first_loss == min(relevant_loss):
            print("Early stop criteria met")
            return True
        return False

    def train_step(self, X_batch, Y_batch):
        """
        Perform forward, backward and gradient descent step here.
        The function is called once for every batch (see trainer.py) to perform the train step.
        The function returns the mean loss value which is then automatically logged in our variable self.train_history.
        Args:
            X: one batch of images
            Y: one batch of labels
        Returns:
            loss value (float) on batch
        """
        # X_batch is the CIFAR10 images. Shape: [batch_size, 3, 32, 32]
        # Y_batch is the CIFAR10 image label. Shape: [batch_size]
        # Transfer images / labels to GPU VRAM, if possible

        # Shape is [64, 3, 32, 32]
        # Er av typen <class 'torch.Tensor'>
        X_batch = utils.to_cuda(X_batch)
        Y_batch = utils.to_cuda(Y_batch)

        # Perform the forward pass
        predictions = self.model(X_batch)

        # Compute the cross entropy loss for the batch
        loss = self.loss_criterion(predictions, Y_batch)

        # Backpropagation
        loss.backward()

        # Gradient descent step
        self.optimizer.step()

        # Reset all computed gradients to 0
        self.optimizer.zero_grad()

        return loss.detach().cpu().item()

    def train(self):
        """
        Trains the model for [self.epochs] epochs.
        """

        def should_validate_model():
            return self.global_step % self.num_steps_per_val == 0

        for epoch in range(self.epochs):
            self.epoch = epoch
            print("Epoch: ", epoch)
            # Perform a full pass through all the training samples
            for X_batch, Y_batch in tqdm(self.dataloader_train):
                loss = self.train_step(X_batch, Y_batch)
                self.train_history["loss"][self.global_step] = loss
                self.global_step += 1
                # Compute loss/accuracy for validation set
                if should_validate_model():
                    self.validation_step()
                    self.save_model()
                    if self.should_early_stop():
                        print("Early stopping.")
                        return

    def save_model(self):
        def is_best_model():
            """
                Returns True if current model has the lowest validation loss
            """
            val_loss = self.validation_history["loss"]
            validation_losses = list(val_loss.values())
            return validation_losses[-1] == min(validation_losses)

        state_dict = self.model.state_dict()
        filepath = self.checkpoint_dir.joinpath(f"{self.global_step}.ckpt")

        utils.save_checkpoint(state_dict, filepath, is_best_model())

    def load_best_model(self):
        state_dict = utils.load_best_checkpoint(self.checkpoint_dir)
        if state_dict is None:
            print(
                f"Could not load best checkpoint. Did not find under: {self.checkpoint_dir}")
            return
        self.model.load_state_dict(state_dict)


def create_plots(trainer: Trainer, path: str, name: str):
    plot_path = pathlib.Path(path)
    plot_path.mkdir(exist_ok=True)
    # Save plots and show them
    plt.figure(figsize=(20, 8))
    plt.subplot(1, 2, 1)
    plt.title("Cross Entropy Loss")
    utils.plot_loss(trainer.train_history["loss"], label="Training loss", npoints_to_average=10)
    utils.plot_loss(trainer.validation_history["loss"], label="Validation loss")
    plt.legend()
    plt.subplot(1, 2, 2)
    plt.title("Accuracy")
    utils.plot_loss(trainer.validation_history["accuracy"], label="Validation Accuracy")
    utils.plot_loss(trainer.train_history["accuracy"], label="Training Accuracy")
    plt.legend()
    plt.savefig(plot_path.joinpath(f"{name}.png"))
    plt.show()