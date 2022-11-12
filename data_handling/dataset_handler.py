import torch
from data_handling.frames_handler import *
from torch.utils.data import Dataset, sampler
from skimage import io


class BronchusDataset(Dataset):
    """ The dataset class """
    def __init__(self, csv_file, root_directory, num_bronchus_generations, transform=None):
        # CSV file containing 2 columns: frame_path and label
        self.labeled_frames = pd.read_csv(csv_file, index_col=False)
        self.root_directory = root_directory
        self.num_generations = num_bronchus_generations
        self.transform = transform
        self.label_mapping = self.get_label_mapping()  # Mapping from original labels to new labels
        self.keys = list(self.label_mapping.keys())

    def __len__(self):
        return len(self.labeled_frames)

    def __getitem__(self, index):
        """ Enables the fetching of values with dataset[index] in the dataset
         for both columns """
        if torch.is_tensor(index):
            index = index.tolist()

        # Fetch columns from the csv for the given index
        frame_name = os.path.join(self.root_directory, self.labeled_frames.iloc[index, 0])

        # Frame
        frame = io.imread(frame_name)
        if self.transform:
            frame = self.transform(frame)

        # Get original label
        original_label = self.labeled_frames.iloc[index, 1]

        # Change label by using the given mapping system
        if original_label in self.keys:
            new_label = self.label_mapping[original_label]
        else:
            new_label = 0

        return frame, new_label

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

    def get_label_mapping(self):
        """
        Return a dictionary {"old originally label": "new label reducing class numbers"}
        """

        if self.num_generations == 1:
            label_mapping = {
                1: 1,  # Trachea
                5: 2,  # Right Main Bronchus
                4: 3,  # Left Main Bronchus
            }

        elif self.num_generations == 2:
            label_mapping = {
                1:  1,   # Trachea
                5:  2,   # Right Main Bronchus
                4:  3,   # Left Main Bronchus
                14: 4,  # Right/Left Upper Lobe Bronchus
                15: 5,  # Right Truncus Intermedicus
                12: 6,  # Left Lower Lobe Bronchus
                13: 7   # Left Upper Lobe Bronchus
            }

        elif self.num_generations == 3:
            label_mapping = {
                1:   1,   # Trachea
                5:   2,   # Right Main Bronchus
                4:   3,   # Left Main Bronchus
                14:  4,   # Right/Left Upper Lobe Bronchus
                15:  5,   # Right Truncus Intermedicus
                12:  6,   # Left Lower Lobe Bronchus
                13:  7,   # Left Upper Lobe Bronchus
                49:  8,   # Right B1
                50:  9,   # Right B2
                48:  10,  # Right B3
                2:   11,  # Right Middle Lobe Bronchus (parent for B4 og B5)
                3:   12,  # Right lower Lobe Bronchus (possible called right lower lobe bronchus (1))
                11:  13,  # Right Lower Lobe Bronchus (2)
                39:  14,  # Left Main Bronchus
                38:  15,  # Left B6
                42:  16,  # Left Upper Division Bronchus
                43:  17,  # Left Lingular Bronchus (or singular?)
            }

        elif self.num_generations == 4:
            label_mapping = {
                1: 1,  # Trachea
                5: 2,  # Right Main Bronchus
                4: 3,  # Left Main Bronchus
                14: 4,  # Right/Left Upper Lobe Bronchus
                15: 5,  # Right Truncus Intermedicus
                12: 6,  # Left Lower Lobe Bronchus
                13: 7,  # Left Upper Lobe Bronchus
                49: 8,  # Right B1
                50: 9,  # Right B2
                48: 10,  # Right B3
                2: 11,  # Right Middle Lobe Bronchus (parent for B4 og B5)
                3: 12,  # Right lower Lobe Bronchus (possible called right lower lobe bronchus (1))
                11: 13,  # Right Lower Lobe Bronchus (2)
                39: 14,  # Left Main Bronchus
                38: 15,  # Left B6
                42: 16,  # Left Upper Division Bronchus
                43: 17,  # Left Lingular Bronchus (or singular?)
                7:  18,  # Right B4
                6:  19,  # Right B5
                91: 20,  # Left B1+B2
                90: 21,  # Left B3
                40: 22,  # Left B4
                41: 23,  # Left B5
                82: 24,  # Left B8
                37: 25,  # Left B9
                36: 26,  # Left B10
            }

        else:
            label_mapping = None
            print("Did not find the number of bronchus generations")

        return label_mapping

    def get_num_classes(self):
        return len(list(self.label_mapping.keys()))+1  # Num defined classes + one class for all undefined classes
