"""
dataset and dataloader for the biovid dataset
the samples.csv provides an index for all samples
the samples.csv can be used to construct the paths to each sample (see getitem)
Usage and usecases:
    Use Case 1:
        get either the filtered data or the raw data
    Use Case 2:
        for cross-validation the model has to be trained in a leave-one-subject-out
        manner. the dataloader has to be able to exlude signals/recordings that belong
        to one specific subject during training
    Use Case 3:
        loading based on labels. The dataset offers 5 classes: BL, PA1, PA2, PA3, and PA4
        with class ids 0, 1, 2, 3, and 4 respectively. For some experiments only binary
        cases have to be evaluated, i.e., for example the binary classification of a signal
        based on 0 vs 4. The functionality required is the ability to load data that corresponds
        to the given labels.
    Use Case 4:
        load specified modalities. since the dataset is multimodal, the user should be able to 
        choose which modalities has to be loaded.
"""
import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
import random


class BioVid_PartA_bio(Dataset):
    def __init__(self, csv_file: str, root_dir: str, exclude_subject=None, include_subject=None ,biosignals_filtered: bool=True, 
                 classes=None, modalities=None, transform=None, dtype='float32') -> None:
        """
            Args:
                csv_file (string): Path to the csv file with the indexing (sample.csv)
                root_dir (string): Directory with all the samples/subdirectories (biosignals_raw/filtered, video)
                biosignals_filtered (bool): Whether to use filtered or raw biosignals
                transform (callable, optional): Optional transform to be applied on a sample.
                dtype (string): Data type for the loaded signals
                labels (list, optional): List of class labels to include
                modalities (list, optional): List of modalities to load
        """
        self.dtype = dtype
        self.samples_index = pd.read_csv(csv_file, sep='\t')
        self.root_dir = root_dir
        self.transform = transform

        if biosignals_filtered:
            biosignals = "biosignals_filtered"
        else:
            biosignals = "biosignals_raw"

        self.biosignals_dir = os.path.join(root_dir, biosignals)

        if classes is not None:
            assert len(classes) > 1, f"Required at least 2 classes. Only {len(classes)} were given."
            self.samples_index = self.samples_index[self.samples_index['class_id'].isin(classes)]

        if modalities is None:
            self.modalities = ['gsr', 'ecg', 'emg_trapezius']
        elif isinstance(modalities, str):
            self.modalities = [modalities]
        else:
            self.modalities = modalities 

        if isinstance(exclude_subject, int):
            exclude_subject = [exclude_subject]

        if exclude_subject is not None:
            self.samples_index = self.samples_index[~self.samples_index['subject_id'].isin(exclude_subject)]

        if isinstance(include_subject, int):
            include_subject = [include_subject]

        if include_subject is not None:
            self.samples_index = self.samples_index[self.samples_index['subject_id'].isin(include_subject)]

        
    def __len__(self) -> int:
        return len(self.samples_index)
    
    def __getitem__(self, index):
        if isinstance(index, int):
            return self._load_sample(index)
        if isinstance(index, slice):
            start, stop, step = index.indices(len(self))
            index = range(start, stop, step)
        return self._load_samples(index)
        
    def _load_samples(self, indices):
        if torch.is_tensor(indices):
            indices = indices.tolist()
        return [self._load_sample(index) for index in indices]
    
    def _load_sample(self, index):
        sample_path = os.path.join(self.samples_index.iloc[index, 1],
                                   self.samples_index.iloc[index,5])
        biosignal_path = os.path.join(self.biosignals_dir, sample_path + '_bio.csv')
        df_biosignals = pd.read_csv(biosignal_path, sep='\t')
        label = self.samples_index.iloc[index, 2]
        sample = {'label': label}
        sample['id'] = self.samples_index.iloc[index, 0] 
        for mod in self.modalities:
            sample[mod] = df_biosignals[mod].to_numpy(dtype=self.dtype)
        if self.transform:
            sample = self.transform(sample)
        return sample

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""
    def __call__(self, sample):
        sample_tensor = {}
        for key, value in sample.items():
            try:
                sample_tensor[key] = torch.from_numpy(value).unsqueeze(-1)
            except Exception as e:
                sample_tensor[key] = value
        return sample_tensor
    
def train_test_lkso(csv_file: str, k):
    samples_index = pd.read_csv(csv_file, sep='\t')
    ids = samples_index['subject_id'].unique()

    if isinstance(k, float):
        if not 0.0 <= k <= 1.0:
            raise ValueError('if k is float should be between 0.0 and 1.0')
        k = int(np.ceil(len(ids) * k))
    if not (1 <= k < len(ids)):
        raise ValueError("k must be at least 1 and less than the number of unique subjects")
    test_indices = np.random.choice(len(ids), size=k, replace=False)
    
    test_ids = ids[test_indices]
    
    mask = np.ones(len(ids), dtype=bool)
    mask[test_indices] = False
    train_ids = ids[mask]
    return train_ids, test_ids

def train_test_dataloader(csv_file: str, root_dir: str, test_size, 
                    biosignals_filtered: bool=True, classes=None,
                    modalities=None, transform=None, batch_size=128,
                    dtype='float32'):
    train_ids, test_ids = train_test_lkso(csv_file, test_size)
    train_data = BioVid_PartA_bio(csv_file=csv_file, root_dir=root_dir,
                                  classes=classes, include_subject=train_ids,
                                  biosignals_filtered=biosignals_filtered,
                                  modalities=modalities, transform=transform,
                                  dtype=dtype)
    test_data = BioVid_PartA_bio(csv_file=csv_file, root_dir=root_dir,
                                 classes=classes, include_subject=test_ids,
                                 biosignals_filtered=biosignals_filtered,
                                 modalities=modalities, transform=transform,
                                 dtype=dtype)
    train_dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_dataloader, test_dataloader