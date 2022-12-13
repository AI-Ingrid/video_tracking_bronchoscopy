import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random
from sklearn.metrics import f1_score, ConfusionMatrixDisplay, confusion_matrix
from parameters import network_type


# Allow torch/cudnn to optimize/analyze the input/output shape of convolutions
# To optimize forward/backward pass.
# This will increase model throughput for fixed input shape to the network
torch.backends.cudnn.benchmark = True

# Cudnn is not deterministic by default. Set this to True if you want
# to be sure to reproduce your results
torch.backends.cudnn.deterministic = True


def set_seed(seed: int):
    np.random.seed(seed)
    torch.manual_seed(seed)
    random.seed(seed)


def to_cuda(elements):
    """
    Transfers every object in elements to GPU VRAM if available.
    elements can be a object or list/tuple of objects
    """
    if torch.cuda.is_available():
        if type(elements) == tuple or type(elements) == list:
            return [x.cuda() for x in elements]
        return elements.cuda()
    return elements


def save_checkpoint(state_dict: dict,
                    filepath: pathlib.Path,
                    is_best: bool,
                    max_keep: int = 1):
    """
    Saves state_dict to filepath. Deletes old checkpoints as time passes.
    If is_best is toggled, saves a checkpoint to best.ckpt
    """
    filepath.parent.mkdir(exist_ok=True, parents=True)
    list_path = filepath.parent.joinpath("latest_checkpoint")
    torch.save(state_dict, filepath)
    if is_best:
        torch.save(state_dict, filepath.parent.joinpath("best.ckpt"))
    previous_checkpoints = get_previous_checkpoints(filepath.parent)
    if filepath.name not in previous_checkpoints:
        previous_checkpoints = [filepath.name] + previous_checkpoints
    if len(previous_checkpoints) > max_keep:
        for ckpt in previous_checkpoints[max_keep:]:
            path = filepath.parent.joinpath(ckpt)
            if path.exists():
                path.unlink()
    previous_checkpoints = previous_checkpoints[:max_keep]
    with open(list_path, 'w') as fp:
        fp.write("\n".join(previous_checkpoints))


def get_previous_checkpoints(directory: pathlib.Path) -> list:
    assert directory.is_dir()
    list_path = directory.joinpath("latest_checkpoint")
    list_path.touch(exist_ok=True)
    with open(list_path) as fp:
        ckpt_list = fp.readlines()
    return [_.strip() for _ in ckpt_list]


def load_best_checkpoint(directory: pathlib.Path):
    filepath = directory.joinpath("best.ckpt")
    if not filepath.is_file():
        return None
    return torch.load(directory.joinpath("best.ckpt"))


def plot_loss(loss_dict: dict, label: str = None, npoints_to_average=1, plot_variance=True):
    """
    Args:
        loss_dict: a dictionary where keys are the global step and values are the given loss / accuracy
        label: a string to use as label in plot legend
        npoints_to_average: Number of points to average plot
    """
    global_steps = list(loss_dict.keys())
    loss = list(loss_dict.values())
    if npoints_to_average == 1 or not plot_variance:
        plt.plot(global_steps, loss, label=label)
        return

    npoints_to_average = 10
    num_points = len(loss) // npoints_to_average
    mean_loss = []
    loss_std = []
    steps = []
    for i in range(num_points):
        points = loss[i*npoints_to_average:(i+1)*npoints_to_average]
        step = global_steps[i*npoints_to_average + npoints_to_average//2]
        mean_loss.append(np.mean(points))
        loss_std.append(np.std(points))
        steps.append(step)
    plt.plot(steps, mean_loss,
             label=f"{label} (mean over {npoints_to_average} steps)")
    plt.fill_between(
        steps, np.array(mean_loss) -
        np.array(loss_std), np.array(mean_loss) + loss_std,
        alpha=.2, label=f"{label} variance over {npoints_to_average} steps")


def get_predicted_labels(predictions):
    predictions = predictions.cpu()
    predicted_labels = []

    for batch_index, batch in enumerate(predictions.detach().numpy()):
        predicted_label = np.argmax(batch)
        predicted_labels.append(predicted_label)

    return predicted_labels


def get_label_name(label):
    label_names = {
        1: "Trachea",
        2: "Right Main Bronchus",
        3: "Left Main Bronchus",
        4: "Right/Left Upper Lobe Bronchus",
        5: "Right Truncus Intermedicus",
        6: "Left Lower Lobe Bronchus",
        7: "Left Upper Lobe Bronchus",
        8: "Right B1",
        9: "Right B2",
        10: "Right B3",
        11: "Right Middle Lobe Bronchus 2",
        12: "Right Lower Lobe Bronchus 1",
        13: "Right Lower Lobe Bronchus 2",
        14: "Left Main Bronchus",
        15: "Left B6",
        26: "Left Upper Division Bronchus",
        27: "Left Singular Bronchus",
    }
    if label not in list(label_names.keys()):
        name = label
    else:
        name = label_names[label]
    return name


def plot_predictions_test_set(test_set, trainer, path, network_type):
    # Store images with predicted and true label on it
    batch_num = 0
    for X_batch, Y_batch in test_set:
        print("Batch num: ", batch_num)
        X_batch_cuda = to_cuda(X_batch)

        # Perform the forward pass
        predictions = trainer.model(X_batch_cuda)

        predictions = predictions.cpu()
        # DirectionDetNet
        if network_type == 'direction_det_net':
            label_names = {1: "Forward",
                           0: "Backward"}
            # Find predicted label
            for batch_index, batch in enumerate(predictions.detach().numpy()):
                predicted_label = int(np.argmax(batch))
                predicted_name_label = label_names[predicted_label]
                original_label = int(np.argmax(Y_batch[batch_index]))
                original_name_label = label_names[original_label]

                name = f"batch_{batch_num}_index_{batch_index}"
                print("Predicted label: ", predicted_name_label, " Original label: ", original_name_label)

                # Create plots
                plot_path = pathlib.Path(path)
                plot_path.mkdir(exist_ok=True)
                fig = plt.figure(figsize=(25, 6), constrained_layout=True)
                images = X_batch[batch_index]
                fig.suptitle(f"Predicted Label: {predicted_name_label} \n Original Label: {original_name_label}")

                # Image 1
                plt.subplot(1, 5, 1)
                image_1 = images[0].numpy()
                plt.title("Frame 1")
                plt.imshow(image_1)

                # Image 2
                plt.subplot(1, 5, 2)
                image_2 = images[1].numpy()
                plt.title("Frame 2")
                plt.imshow(image_2)

                # Image 3
                plt.subplot(1, 5, 3)
                image_3 = images[2].numpy()
                plt.title("Frame 3")
                plt.imshow(image_3)

                # Image 4
                plt.subplot(1, 5, 4)
                image_4 = images[3].numpy()
                plt.title("Frame 4")
                plt.imshow(image_4)

                # Image 5
                plt.subplot(1, 5, 5)
                image_5 = images[4].numpy()
                plt.title("Frame 5")
                plt.imshow(image_5)

                plt.savefig(plot_path.joinpath(f"{name}.png"))
                print("Saving figure..")

            batch_num += 1
            if batch_num == 10:
                break

        # SegmentDetNet
        else:
            for batch_index, batch in enumerate(predictions.detach().numpy()):
                # Find predicted label
                predicted_label = int(np.argmax(batch))
                original_label = int(np.argmax(Y_batch[batch_index]))

                # Get label names
                original_name_label = get_label_name(original_label)
                predicted_name_label = get_label_name(predicted_label)

                name = f"batch_{batch_num}_index_{batch_index}"
                print("Predicted label: ", str(predicted_label), " Original label: ", str(original_label))

                # Create plots
                plot_path = pathlib.Path(path)
                plot_path.mkdir(exist_ok=True)
                plt.figure(figsize=(8, 8), constrained_layout=True)
                image = X_batch[batch_index]
                image = image.permute(1, 2, 0).numpy()

                # Predicted label and Original label image
                plt.subplot(1, 2, 1)
                plt.title(f"Predicted Label: {predicted_label} : {predicted_name_label} \n Original Label: {original_label}: {original_name_label}")
                plt.imshow(image)
                plt.savefig(plot_path.joinpath(f"{name}.png"))
                print("Figure saved..")

            batch_num += 1
            if batch_num == 10:
                break


def compute_f1_score(test_set, trainer):
    batch_num = 0
    f1_macro_score = 0
    f1_micro_score = 0
    f1_weighted_score = 0

    for X_batch, Y_batch in test_set:
        X_batch_cuda = to_cuda(X_batch)
        # Perform the forward pass
        predictions = trainer.model(X_batch_cuda)

        predicted_labels = get_predicted_labels(predictions)
        Y_batch_1d = get_predicted_labels(Y_batch)

        f1_macro_score += f1_score(Y_batch_1d, predicted_labels, average='macro')
        f1_micro_score += f1_score(Y_batch_1d, predicted_labels, average='micro')
        f1_weighted_score += f1_score(Y_batch_1d, predicted_labels, average='weighted')

        batch_num += 1

    print("F1 macro score: ", f1_macro_score/batch_num)
    print("F1 micro score: ", f1_micro_score/batch_num)
    print("F1 weighted score: ", f1_weighted_score/batch_num)


def plot_confusion_matrix(test_set, trainer, path):
    all_predicted_labels = []
    all_original_labels = []
    plot_path = pathlib.Path(path)
    plot_path.mkdir(exist_ok=True)
    for X_batch, Y_batch in test_set:
        X_batch_cuda = to_cuda(X_batch)

        # Perform the forward pass
        predictions = trainer.model(X_batch_cuda)

        predicted_labels = get_predicted_labels(predictions)
        original_labels = get_predicted_labels(Y_batch)

        all_predicted_labels += predicted_labels
        all_original_labels += original_labels

    if network_type == 'direction_det_net':
        classes = list(range(0, 2))
    else:
        classes = list(range(1, 28))

    cm = confusion_matrix(all_original_labels, all_predicted_labels, labels=classes)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=classes)
    fig, ax = plt.subplots(figsize=(20, 20))
    plt.title(f"Confusion Metrics for {network_type}")
    disp.plot(ax=ax)
    plt.savefig(plot_path.joinpath(f"confusion_matrix.png"))
    plt.show()

