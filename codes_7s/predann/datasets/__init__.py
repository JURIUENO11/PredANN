import os
from .dataset import Dataset
from .preprocessing_eegmusic_dataset_7s import Preprocessing_EEGMusic_dataset


def get_dataset(dataset, dataset_dir, subset ,download=True):

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "preprocessing_eegmusic":
        d = Preprocessing_EEGMusic_dataset(root=dataset_dir, download=download, subset=subset)
    else:    
        raise NotImplementedError("Dataset not implemented")
    return d
