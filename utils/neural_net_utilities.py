import torch
import matplotlib.pyplot as plt
import numpy as np
import pathlib
import random

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


def plot_predictions_test_set(test_set, trainer):
    # Store images with predicted and true label on it
    batch_num = 0
    for X_batch, Y_batch in test_set:
        print("Batch num: ", batch_num)
        X_batch = to_cuda(X_batch)
        Y_batch = to_cuda(Y_batch)

        # Perform the forward pass
        predictions = trainer.model(X_batch)

        # Find predicted label
        for batch_index, batch in enumerate(predictions):
            batch = to_cuda(batch)
            predicted_label = str(np.argmax(batch) + 1)  # Because zero-indexing
            original_label = str(Y_batch[batch_index])
            name = f"batch_{batch_num}_index_{batch_index}"
            print("Predicted label: ", predicted_label, " Original label: ", )

            # Create plots
            plot_path = pathlib.Path("data_handling/test_set_images")
            plot_path.mkdir(exist_ok=True)
            plt.figure(figsize=(20, 8))

            # Predicted label image
            plt.subplot(1, 2, 1)
            plt.title(f"Predicted Label: {predicted_label}")
            plt.imshow(X_batch[batch_index])
            plt.legend()

            # Original label image
            plt.subplot(1, 2, 2)
            plt.title(f"Original Label: {original_label}")
            plt.imshow(X_batch[batch_index])
            plt.legend()
            plt.savefig(plot_path.joinpath(f"{name}.png"))
            print("Figure saved..")

        batch_num += 1
