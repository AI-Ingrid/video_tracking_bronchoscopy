import torch
from data_handling.frames_handler import *
from torch.utils.data import Dataset, sampler
from skimage import io, transform


class BronchoscopyDataset(Dataset):
    """ The dataset class """
    def __init__(self, csv_file, root_directory, transform=None):
        # CSV file containing 2 columns: frame_path and label
        self.labeled_frames = pd.read_csv(csv_file, index_col=False)
        self.root_directory = root_directory
        self.transform = transform
        # TODO: Add them
        self.class_to_index = {}
        self.classes = []

    def __len__(self):
        return len(self.labeled_frames)

    def __getitem__(self, index):
        """ Enables the fetching of values with dataset[index] in the dataset
         for both columns """
        if torch.is_tensor(index):
            index = index.tolist()

        # Fetch columns from the csv for the given index
        frame_name = os.path.join(self.root_directory, self.labeled_frames.iloc[index, 0])

        # Normalize the frame
        #frame = io.imread(frame_name)/255.0
        frame = io.imread(frame_name)
        label = self.labeled_frames.iloc[index, 1]

        # Create a sample dictionary containing the column values
        #sample = {'Frame': frame, 'Label': label}

        if self.transform:
            frame = self.transform(frame)

        return frame, label

    def get_dataloaders(self, batch_size, test_split, validation_split):
        """ Splits the data into train, test and validation data """
        # TODO: Se på sklearn.model_selection sin train_test_split
        indices = list(range(len(self)))

        # Test
        test_split_index = int(np.floor(test_split * len(self)))
        test_indices = np.random.choice(indices, size=test_split_index, replace=False)
        test_sampler = sampler.SubsetRandomSampler(test_indices)
        test_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=test_sampler)

        # Train (temporary)
        temp_train_indices = list(set(indices) - set(test_indices))

        # Validation
        validation_split_index = int(np.floor(validation_split * len(temp_train_indices)))
        validation_indices = np.random.choice(temp_train_indices, size=validation_split_index, replace=False)
        validation_sampler = sampler.SubsetRandomSampler(validation_indices)
        validation_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=validation_sampler)

        # Train
        train_indices = list(set(temp_train_indices) - set(validation_indices))
        train_sampler = sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_sampler, drop_last=True)

        return train_loader, validation_loader, test_loader

""" Below: Classes for transforming the objects from the dataset class
such as rescaling or random cropping of the frames or converting the frames into
torch images """


class Rescale(object):
    """Rescale the image in a sample to a given size.

    Args:
        output_size (tuple or int): Desired output size. If tuple, output is
            matched to output_size. If int, smaller of image edges is matched
            to output_size keeping aspect ratio the same.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, sample):
        frame, label = sample['Frame'], sample['Label']

        h, w = frame.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)

        frame = transform.resize(frame, (new_h, new_w))

        return {'Frame': frame, 'Label': label}


class RandomCrop(object):
    """Crop randomly the image in a sample.

    Args:
        output_size (tuple or int): Desired output size. If int, square crop
            is made.
    """

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, sample):
        frame, label = sample['Frame'], sample['Label']

        h, w = frame.shape[:2]
        new_h, new_w = self.output_size

        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)

        frame = frame[top: top + new_h,
                      left: left + new_w]

        return {'Frame': frame, 'Label': label}


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, frame):
        #frame, label = sample['Frame'], sample['Label']

        # swap color axis because
        # numpy image: H x W x C
        # torch image: C x H x W
        frame = frame.transpose((2, 0, 1))
        return torch.from_numpy(frame)









