"""
dataset and dataloader for the biovid dataset
the samples.csv provides an index for all samples
the samples.csv can be used to construct the paths to each sample (see getitem)
"""

import torch
import pandas as pd
import numpy as np
import os
from torch.utils.data import Dataset, DataLoader
from typing import Optional, List, Callable, Union, Tuple, NamedTuple
from sklearn.model_selection import KFold

class BioVid_PartA_bio(Dataset):
    def __init__(self, csv_file: str, root_dir: str, exclude_subject: Optional[List[int]]=None, 
                 include_subject: Optional[List[int]]=None ,biosignals_filtered: bool=True, 
                 classes: Optional[List[str]]=None, modalities: Optional[List[str]]=None, 
                 transform: Optional[Callable]=None, dtype='float32') -> None:
        """
            Args:
                csv_file (string): Path to the csv file with the indexing (sample.csv)
                root_dir (string): Directory with all the samples/subdirectories (biosignals_raw/filtered, video)
                exclude_subject (list, optional): List of subject IDs to exclude from the dataset. Defaults to None.
                include_subject (list, optional): List of subject IDs to include in the dataset. 
                                                If specified, only these subjects will be included. Defaults to None.
                biosignals_filtered (bool): Whether to use filtered biosignals (True) or raw biosignals (False). 
                                            Defaults to True.
                classes (list, optional): List of class labels to include. Defaults to None, which includes all classes.  
                modalities (list, optional): ist of modalities (allowed are 'ecg', 'gsr', 'emg_trapezius') to load. 
                                             Defaults to None, which includes all available modalities.  
                transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
                dtype (string): Data type for the loaded signals. Defaults to 'float32'.            
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

class LKSOFold(NamedTuple):
    train_dataloader: DataLoader
    test_dataloader: DataLoader
    train_ids: List[int]
    test_ids: List[int]

def train_test_split_by_id(csv_file: str, k):
    samples_index = pd.read_csv(csv_file, sep='\t')
    ids = samples_index['subject_id'].unique()

    if isinstance(k, float):
        if not 0.0 <= k <= 1.0:
            raise ValueError('When k is float, it should be between 0.0 and 1.0')
        k = int(np.ceil(len(ids) * k))
    if not (1 <= k < len(ids)):
        raise ValueError("k must be at least 1 and less than the number of unique subjects")
    test_indices = np.random.choice(len(ids), size=k, replace=False)
    
    test_ids = ids[test_indices]
    
    mask = np.ones(len(ids), dtype=bool)
    mask[test_indices] = False
    train_ids = ids[mask]
    return train_ids.tolist(), test_ids.tolist()

def lkso_generator(csv_file: str, k: int):
    samples_index = pd.read_csv(csv_file, sep='\t')
    ids = samples_index['subject_id'].unique()
        
    kfold = KFold(n_splits=k)
    
    for train_index, test_index in kfold.split(ids):
        train_ids = ids[train_index].tolist()
        test_ids = ids[test_index].tolist()
        yield train_ids, test_ids

def lkso_dataloader(csv_file: str, root_dir: str, k:int,
                    biosignals_filtered: bool=True, classes: Optional[List[str]]=None,
                    modalities: Optional[List[str]]=None, transform: Optional[Callable]=None,
                    batch_size:int =128, dtype='float32', **kwargs):
    folds = []
    for train_ids, test_ids in lkso_generator(csv_file, k):
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
        folds.append(LKSOFold(train_dataloader, test_dataloader, train_ids, test_ids))
    return folds 


def train_test_dataloader(csv_file: str, root_dir: str, test_size: Union[float, int], 
                    biosignals_filtered: bool=True, classes: Optional[List[str]]=None,
                    modalities: Optional[List[str]]=None, transform: Optional[Callable]=None,
                    train_ids: Optional[List[int]]=None, test_ids: Optional[List[int]]=None,
                    batch_size:int =128, dtype='float32', **kwargs) -> Tuple[DataLoader, DataLoader]:
    """
        Splits the dataset into training and testing sets, and returns corresponding dataloaders.

        Args:
            csv_file (string): Path to the csv file with the indexing (sample.csv)
            root_dir (string): Directory with all the samples/subdirectories (biosignals_raw/filtered, video)
            test_size (Union[float, int]): Proportion of the dataset to include in the test split (float between 0 and 1) 
                                           or an absolute number of test samples (int).
            biosignals_filtered (bool): Whether to use filtered biosignals (True) or raw biosignals (False). 
                                        Defaults to True.
            classes (list, optional): List of class labels to include. Defaults to None, which includes all classes.  
            modalities (list, optional): ist of modalities (allowed are 'ecg', 'gsr', 'emg_trapezius') to load. 
                                         Defaults to None, which includes all available modalities.  
            transform (callable, optional): Optional transform to be applied on a sample. Defaults to None.
            batch_size (int, optional): Number of samples per batch to load. Defaults to 128.
            dtype (string): Data type for the loaded signals. Defaults to 'float32'. 

        Returns:
            Tuple[DataLoader, DataLoader]: A tuple containing the training DataLoader and the testing DataLoader.
    """
    if train_ids is None and test_ids is None:
        train_ids, test_ids = train_test_split_by_id(csv_file, test_size)
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

    return train_dataloader, test_dataloader, train_ids, test_ids