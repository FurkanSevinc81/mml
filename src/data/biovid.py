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


class BioVid_PartA(Dataset):
    def __init__(self, csv_file: str, root_dir: str, biosignals_filtered=True, multimodal=True, transform=None, dtype='float32'):
        """
        Args:
            csv_file (string): Path to the csv file with the indexing (sample.csv)
            root_dir (string): Directory with all the samples/subdirectories (biosignals_raw/filtered, video)
            transform (callable, optional): Optional transform to be applied on a sample.
        """
        self.dtype = dtype
        self.samples_index = pd.read_csv(csv_file, sep='\t')
        self.root_dir = root_dir
        self.transform = transform
        if biosignals_filtered:
            biosignals = "biosignals_filtered"
        else:
            biosignals = "biosignals_raw"
        self.video_dir = os.path.join(root_dir, "video")
        self.biosignals_dir = os.path.join(root_dir, biosignals)
        self.multimodal = multimodal
        
        
    def __len__(self)->int:
        return len(self.samples_index)

    def __getitem__(self, index:int) -> dict:
        """
        Reading of the samples is performerd here, while the index file is read in `__init__`
        """
        if torch.is_tensor(index):
            index = index.tolist()
        sample_path = os.path.join(self.samples_index.iloc[index, 1],
                                   self.samples_index.iloc[index,5])
        biosignal_path = os.path.join(self.biosignals_dir, sample_path + '_bio.csv')
        video_path = os.path.join(self.video_dir, sample_path)
        video = np.array([])
        df_biosignals = pd.read_csv(biosignal_path, sep='\t')
        label = self.samples_index.iloc[index, 2]
        # loads the biosignals as a whole
        if self.multimodal:
            biosignals = df_biosignals.to_numpy()
            sample = {'target': label,
                      'biosignals': biosignals,
                      'video': video}
        else: # loads the biosignals as a collection of unimodal datasets 
            timestamps = df_biosignals.time.to_numpy(dtype='int64')
            gsr_signal = df_biosignals.gsr.to_numpy(dtype=self.dtype)
            ecg_signal = df_biosignals.ecg.to_numpy(dtype=self.dtype)
            emg_trapezius = df_biosignals.emg_trapezius.to_numpy(dtype=self.dtype)
            emg_corrugator = df_biosignals.emg_corrugator.to_numpy(dtype=self.dtype)
            emg_zygomaticus = df_biosignals.emg_zygomaticus.to_numpy(dtype=self.dtype)
        
            sample = {'target': label,
                    'time': timestamps,
                    'gsr': gsr_signal,
                    'ecg': ecg_signal,
                    'emg_t': emg_trapezius,
                    'emg_c': emg_corrugator,
                    'emg_z': emg_zygomaticus,
                    'video': video}

        if self.transform:
            sample = self.transform(sample)
        
        return sample

class BioVid_PartA_bio(Dataset):
    def __init__(self, csv_file: str, root_dir: str, biosignals_filtered: bool=True,
                 subject_exclude=None, classes=None, modalities=None, 
                 transform=None, dtype='float32') -> None:
        """
            Args:
                csv_file (string): Path to the csv file with the indexing (sample.csv)
                root_dir (string): Directory with all the samples/subdirectories (biosignals_raw/filtered, video)
                biosignals_filtered (bool): Whether to use filtered or raw biosignals
                transform (callable, optional): Optional transform to be applied on a sample.
                dtype (string): Data type for the loaded signals
                subject_exclude (int or list, optional): Subject(s) to exclude for leave-one-subject-out
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

        if subject_exclude is not None:
            if isinstance(subject_exclude, int):
                subject_exclude = [subject_exclude]
            self.samples_index = self.samples_index[~self.samples_index['subject_id'].isin(subject_exclude)]

        if classes is not None:
            assert len(classes) > 1, f"Required at least 2 classes. Only {len(classes)} were given."
            self.samples_index = self.samples_index[self.samples_index['class_id'].isin(classes)]

        if modalities is None:
            self.modalities = ['gsr', 'ecg', 'emg_trapezius']
        elif isinstance(modalities, str):
            self.modalities = [modalities]
        else:
            self.modalities = modalities 
        
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