import torch
from data_handling.frames_handler import *
from torch.utils.data import Dataset, sampler
from skimage import io
import random


class BronchusDataset(Dataset):
    """ The dataset class """
    def __init__(self, csv_file, root_directory, network_type, num_bronchus_generations=None, transform=None):
        # CSV file containing 2 columns: frame_path and label
        self.labeled_frames = pd.read_csv(csv_file, index_col=False)
        self.root_directory = root_directory
        self.network_type = network_type
        self.num_generations = num_bronchus_generations
        self.transform = transform

        if self.network_type == 'segment_det_net':
            self.label_mapping = self.get_label_mapping()  # Mapping from original labels to new labels
            self.keys = list(self.label_mapping.keys())

    def __len__(self):
        return len(self.labeled_frames)

    def __getitem__(self, index):
        """
        Enables the fetching of values with dataset[index] in the dataset
        """
        if torch.is_tensor(index):
            index = index.tolist()

        if self.network_type == 'segment_det_net':
            # Fetch columns from the csv for the given index
            frame_name = self.labeled_frames.iloc[index, 0]

            # Frame
            frame = io.imread(frame_name)
            if self.transform:
                frame = self.transform(frame)

            # Get original label
            original_label = self.labeled_frames.iloc[index, 1]
            return frame, original_label

        else:
            # Direction detection network
            # Fetch columns from the csv for the given index
            frame_name_1 = self.labeled_frames.iloc[index, 0]
            frame_name_2 = self.labeled_frames.iloc[index, 1]
            frame_name_3 = self.labeled_frames.iloc[index, 2]
            frame_name_4 = self.labeled_frames.iloc[index, 3]
            frame_name_5 = self.labeled_frames.iloc[index, 4]

            # Frame
            frame_1 = io.imread(frame_name_1)
            frame_2 = io.imread(frame_name_2)
            frame_3 = io.imread(frame_name_3)
            frame_4 = io.imread(frame_name_4)
            frame_5 = io.imread(frame_name_5)

            frames = np.array([frame_1, frame_2, frame_3, frame_4, frame_5])

            if self.transform:
                #frame_1 = self.transform(frame_1)
                #frame_2 = self.transform(frame_2)
                #frame_3 = self.transform(frame_3)
                #frame_4 = self.transform(frame_4)
                #frame_5 = self.transform(frame_5)
                # TODO: Inefficient to transform this way
                frames = torch.tensor(frames)

            # Frames tensor: [5, 3, 256, 256] or [15, 256, 256]
            #frames = torch.concat([frame_1, frame_2, frame_3, frame_4, frame_5])


            # Get label
            label = self.labeled_frames.iloc[index, 5]
            label = torch.nn.functional.one_hot(torch.tensor(label), num_classes=2)
            # Frames tensor: [5, 3, 256, 256] or [15, 256, 256]
            # [timestamp, channels, height, width]
            return frames, label

    def perform_mapping(self, label):
        # Label is >= num_generations
        if label in self.keys:
            new_label = self.label_mapping[label]
        # Label is > num_generations
        else:
            new_label = 0
        return new_label

    def relabel_dataset(self):
        """
        Method that relabels the dataset based on a mapping created after the numbers of generations entered.
        Method also deleted frames that has a larger generation than the one entered
        """
        # Change label by using the given mapping system
        self.labeled_frames["Label"] = self.labeled_frames["Label"].apply(lambda label: self.perform_mapping(label))

        # Remove frames not belonging to the generation chosen (labeled with 0)
        self.labeled_frames = self.labeled_frames[(self.labeled_frames.Label != 0)]

    def get_train_dataloaders(self, batch_size, validation_split):
        """ Splits the data into train, test and validation data """
        # Re-label the dataset based on num generations entered
        if self.network_type == 'segment_det_net':
            self.relabel_dataset()

        indices = list(range(len(self)))
        # Shuffle the dataset
        random.shuffle(indices)

        # Validation
        validation_split_index = int(np.floor(validation_split * len(self)))
        validation_indices = indices[:validation_split_index]
        validation_sampler = sampler.SubsetRandomSampler(validation_indices)
        validation_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=validation_sampler, drop_last=True)

        # Train
        train_indices = indices[validation_split_index:]
        train_sampler = sampler.SubsetRandomSampler(train_indices)
        train_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=train_sampler, drop_last=True)

        return train_loader, validation_loader

    def get_test_dataloaders(self, batch_size):
        """ Returns a dataloader for the test data """
        # Re-label the dataset based on num generations entered
        if self.network_type == 'segment_det_net':
            self.relabel_dataset()

        indices = list(range(len(self)))

        # Shuffle the dataset
        random.shuffle(indices)

        # Create Test Data Loader
        test_sampler = sampler.SubsetRandomSampler(indices)
        test_loader = torch.utils.data.DataLoader(self, batch_size=batch_size, sampler=test_sampler, drop_last=True)

        return test_loader

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
