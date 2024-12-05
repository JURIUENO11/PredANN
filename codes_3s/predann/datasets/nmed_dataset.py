from ast import Str
from locale import normalize
import os
import glob
import pickle
import pstats
import numpy as np
import torch
import torchaudio
from collections import defaultdict
from pathlib import Path
from torch import Tensor, FloatTensor
from tqdm import tqdm
from typing import Any, Tuple, Optional
import random
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
import pandas as pd
import torch.nn.functional as F

from audiomentations import(
    Compose,
    AddGaussianNoise,
    Gain
)
from predann.datasets import Dataset

def get_file_list(root):
    # EEG file list
    EEG_list = []
    BASE = os.path.join(root, "EEG_pkl")
    if not os.path.exists(BASE):
        raise RuntimeError('BASE folder not found')

    
    EEG_path_list = [p for p in glob.glob(os.path.join(BASE, '*.pkl'), recursive=True) if os.path.isfile(p)]
    for path in EEG_path_list:
        name = os.path.splitext(os.path.basename(path))[0]

        # we can prepare only 22, 23 and 24 song id.
        if int(name.split('_')[1]) in [21,23,24]:
            # EEG_list is [subject_id, song_id, trial_id, path]
            EEG_list.append([name.split('_')[0], name.split('_')[1], name.split('_')[2], path])


    # Audio file list
    audio_dict = {}
    BASE1 = os.path.join(root, "audio")
    if not os.path.exists(BASE1):
        raise RuntimeError('BASE1 folder is not found')

    audio_path_list = [p for p in glob.glob(os.path.join(BASE1, '*.wav'), recursive=True) if os.path.isfile(p)]
    for path in audio_path_list:
        name = os.path.splitext(os.path.basename(path))[0]
        audio_dict[name] = path

    return EEG_list, audio_dict
        

class NMED_dataset(Dataset):

    _base_dir = "/workdir/share/NMED/NMED-H/NMED-H_dataset"

    def __init__(
        self,
        root: str,
        base_dir: str = _base_dir,
        download: bool = False,
        subset: Optional[str] = None,
    ):
        # if download:
        #     raise Exception("The NMED Dataset is not publicly available")

        self.root = root
        self.base_dir = base_dir
        self.subset = subset
        self.eeg_normalization = None
        self.transform = None
        self.eeg_sample_rate = 125
        self.eeg_length = 375
        self.audio_length = 132300
        self.EEG_subset_list = []

        

        assert subset is None or subset in ["train", "valid", "test"], (
            "When `subset` not None, it must take a value from "
            + "{'train', 'valid', 'test'}."
        )

        # self._path = os.path.join(self.root, self.base_dir)

        if not os.path.exists(self._base_dir):
            raise RuntimeError(
                "Dataset not found. Please place the MSD files in the {} folder.".format(
                    self.base_dir
                )
            )

        EEG_list, self.audio_dict= get_file_list(self._base_dir)
        # Save valid dataset this time
        train_list, test_list = train_test_split(EEG_list, test_size=0.2, shuffle=True, random_state=42)
        if self.subset == "train":
            self.EEG_subset_list = train_list
        elif self.subset == "test" or subset == "valid":
            self.EEG_subset_list = test_list 
 

    def file_path(self, n: int) -> str:
        pass

    def set_transform(self, transform):
        self.transform = Compose(transform)
        print(self.transform)

    def set_eeg_normalization(self, eeg_normalization, clamp_value=None):
        self.eeg_normalization = eeg_normalization
        self.clamp_value = clamp_value


    def getitem(self, n, isClip=True, isTest=False):
        # Load EEG
        with open(self.EEG_subset_list[n][3], 'rb') as f:
            EEG = pickle.load(f)
        # Load audio 
        song_id = self.EEG_subset_list[n][1]
        audio, audio_sample_rate = torchaudio.load(self.audio_dict[song_id])

        # random clip
        if isClip:
            all_eeg_length = EEG.size(1)
            eeg_start = random.randint(0,all_eeg_length-self.eeg_length-1)
            EEG = EEG[:, eeg_start : eeg_start + self.eeg_length]
            audio_start = int(eeg_start/self.eeg_sample_rate * audio_sample_rate)
            audio = audio[:, audio_start : audio_start + self.audio_length]


        if self.eeg_normalization == "channel_mean":
            EEG = self.normalize_EEG(EEG)
        elif self.eeg_normalization == "all_mean":
            EEG = self.normalize_EEG_2(EEG)
        elif self.eeg_normalization == "constant_multiple":
            EEG = self.normalize_EEG_3(EEG)
        elif self.eeg_normalization == "MetaAI":
            EEG = self.normalize_EEG_4(EEG, self.clamp_value)

    
        if isTest==True:
            #song_info= [song_id,subject_id]
            song_info=[int(self.EEG_subset_list[n][1]),int(self.EEG_subset_list[n][0])]
            return EEG, song_info

        else:
            if self.transform != None:
                audio = audio.to('cpu').detach().numpy().copy()
                audio = self.transform(audio, sample_rate=self.audio_sample_rate)
                audio = torch.from_numpy(audio.astype(np.float32)).clone()

                EEG = EEG.to('cpu').detach().numpy().copy()
                EEG = self.transform(EEG, sample_rate=self.eeg_sample_rate)
                EEG = torch.from_numpy(EEG.astype(np.float32)).clone()
            
            # Padding EEG and audio
            padding = (0, 177147-self.audio_length)
            audio = F.pad(audio, padding)
            padding = (0, 729-self.eeg_length)
            EEG = F.pad(EEG, padding)

            return audio, EEG

    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        audio, EEG = self.getitem(n)
        return audio, EEG

    def __len__(self) -> int:
        return len(self.EEG_subset_list)

    def normalize_EEG(self,eeg):
        eeg_mean=torch.mean(eeg,1)
        eeg=eeg-eeg_mean.unsqueeze(1)
        max_eeg=torch.max(abs(eeg),1)
        eeg=eeg/max_eeg.values.unsqueeze(1)
        return eeg
    
    def normalize_EEG_2(self,eeg):
        eeg_mean=torch.mean(eeg)*torch.ones(eeg.shape[0])
        eeg=eeg-eeg_mean.unsqueeze(1)
        max_eeg=torch.max(abs(eeg),1)
        eeg=eeg/max_eeg.values.unsqueeze(1)
        return eeg

    def normalize_EEG_3(self,eeg):
        eeg=100*eeg
        return eeg

    def normalize_EEG_4(self,eeg,clamp_value):
        # baseline
        # eeg_mean=torch.mean(eeg,1)
        # eeg=eeg-eeg_mean.unsqueeze(1)

        # Robust scaler
        for idx, ch_eeg in enumerate(eeg):
            transformer = RobustScaler().fit(ch_eeg.view(-1,1))
            ch_eeg = transformer.transform(ch_eeg.view(-1,1))
            ch_eeg = torch.from_numpy(ch_eeg.astype(np.float32)).clone()
            eeg[idx] = ch_eeg.view(1,-1)

        # clamp
        eeg = torch.clamp(eeg, min=int(-1*clamp_value), max=int(clamp_value))
        return eeg