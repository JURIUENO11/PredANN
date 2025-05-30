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
import collections
from audiomentations import(
    Compose,
    AddGaussianNoise,
    Gain
)
from predann.datasets import Dataset
import torch.nn.functional as F
from sklearn.model_selection import KFold
from typing import List, Tuple
import logging
import os

def setup_logger():
    if os.path.exists('dataloader_debug.log'):
        os.remove('dataloader_debug.log')
    logger = logging.getLogger('dataloader_debug')
    logger.setLevel(logging.DEBUG)
    fh = logging.FileHandler('dataloader_debug.log')
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    return logger

debug_logger = setup_logger()

def get_file_list(root, class_song_id):
    class_song_id = list(map(int, class_song_id.strip('[]').split(',')))
    dict_song_id= {song_id: idx for idx, song_id in enumerate(class_song_id)}

    BASE = os.path.join(root, "DS_EEG_pkl")

    if not os.path.exists(BASE):
        raise RuntimeError('BASE folder is not found') 
    EEG_path_list =[p for p in glob.glob(os.path.join(BASE, '*.pkl'), recursive=True) if os.path.isfile(p)]
    EEG_path_list = sorted(EEG_path_list)


    BASE = os.path.join(root, "audio")
    if not os.path.exists(BASE):
        raise RuntimeError('BASE folder is not found')
    Audio_path_list = [p for p in glob.glob(os.path.join(BASE, '*.wav'), recursive=True) if os.path.isfile(p)]
    Audio_path_list = sorted(Audio_path_list)

    df = pd.DataFrame(columns=['subject', 'song', 'trial', "eeg_path", "audio_path"])
    
    for idx, r_path in enumerate(EEG_path_list):
        r_name = os.path.splitext(os.path.basename(r_path))[0]
        r_song_id = int(r_name.split('_')[1])
        index_of_song = class_song_id.index(r_song_id)
        c_path = Audio_path_list[index_of_song]
        c_name = os.path.splitext(os.path.basename(c_path))[0]
        c_song_id = int(c_name.split('_')[0])
        assert r_song_id==c_song_id, "sort_error_1"

        if r_song_id in class_song_id:
            r_subject = r_name.split('_')[0]
            r_trial = r_name.split('_')[2]
            song = dict_song_id[r_song_id]
            c_subject = r_name.split('_')[0]
            c_trial = r_name.split('_')[2]
            assert r_subject==c_subject and r_trial==c_trial, "sort_error_2"
            df.loc[idx] = [r_subject, song, r_trial, r_path, c_path]

    return df

def get_5s_file(df):
    df.insert(5, "chunk", np.zeros(len(df.index)), True)
    newdf = pd.DataFrame(np.repeat(df.values, 48, axis=0))
    newdf.columns = df.columns
    print("check")

    for i in range(len(newdf.index)):
        newdf.at[i, 'chunk']=i%48

    return newdf

def get_30s_file(df):
    df.insert(5, "chunk", np.zeros(len(df.index)), True)
    newdf = pd.DataFrame(np.repeat(df.values, 8, axis=0))
    newdf.columns = df.columns

    for i in range(len(newdf.index)):
        newdf.at[i, 'chunk']=i%8

    return newdf

def get_window(df,chunk_length,window_size,stride):
    df.insert(6, "window", np.zeros(len(df.index)), True)
    newdf = pd.DataFrame(np.repeat(df.values,int((chunk_length - window_size)/stride + 1), axis=0))
    newdf.columns = df.columns

    for i in range(len(newdf.index)):
        newdf.at[i, 'window']=i%int((chunk_length - window_size)/stride + 1)

    return newdf

def K_split_valid(df,fold_num):
    print("fold_num:",fold_num)
    kf = KFold(n_splits=4, shuffle=True, random_state=42)
    splits = kf.split(df)
    modified_splits = []
    for fold, (train_index, valid_index) in enumerate(splits):
        if fold == fold_num:
            train_df = df.iloc[train_index]
            test_df = df.iloc[valid_index]

    return train_df, test_df

def check_accessed_data(chunk_length,window_size,stride):
    accessed_data = np.zeros(((int((chunk_length-window_size)/stride)+1)*400, window_size))
    print("check")

    return accessed_data


from scipy.signal import butter, filtfilt

def butter_bandpass_filter(data, lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='band')
    y = filtfilt(b, a, data)

    return y

class Preprocessing_EEGMusic_dataset(Dataset):

    _base_dir = "/dataset/NMED-T_dataset"

    def __init__(
        self,
        root: str,
        base_dir: str = _base_dir,
        download: bool = False,
        subset: Optional[str] = None,
    ):

        self.root = root
        self.base_dir = base_dir
        self.subset = subset
        self.eeg_normalization = None
        self.transform = None
        self.eeg_sample_rate = 125
        self.eeg_clip_length = 375
        self.audio_length = 0
        self.audio_sample_rate = 44100
        self.class_song_id="[21,22,23,24,25,26,27,28,29,30]"
        self.train_test_splitting="random_split_30s"
        self.random_numbers = [] 
        self.start_position = 0
        self.evaluation_length=375

        assert subset is None or subset in ["train", "valid", "test", "CV","SW_valid","SW_train"], (
            "When `subset` not None, it must take a value from "
            + "{'train', 'valid', 'test', 'CV'}."
        )

        self.window_size = None
        self.stride = None
        self.fold = None
        self.mode = None
        self.start = []
        self.start_value = 0
        self.fs = 125.0       
        self.lowcut = 1.0     
        self.highcut = 50.0   
        self.df = get_file_list(self._base_dir, self.class_song_id).reset_index(drop=True)

        if self.train_test_splitting=='random_split_5s':
            self.df = get_5s_file(self.df)
            self.chunk_length = 5*125
        if self.train_test_splitting=='random_split_30s':
            self.df = get_30s_file(self.df)
            self.chunk_length = 30*125
        if self.subset == "CV":
            self.df_subset =self.df
        else:
            df_train, df_test = train_test_split(self.df, test_size=0.25, random_state=42, stratify=self.df.song)
            if self.subset == "train":
                self.df_subset = df_train
            elif self.subset == "test" or self.subset == "valid":
                self.df_subset = df_test

    def file_path(self, n: int) -> str:
        pass

    def set_transform(self, transform):
        self.transform = Compose(transform)
        print(self.transform)

    def set_other_parameters(self, eeg_clip_length, audio_clip_length, split_seed, class_song_id,shifting_time, start_position, evaluation_length):
        self.eeg_clip_length = eeg_clip_length
        self.audio_clip = audio_clip_length
        self.audio_length = audio_clip_length* self.audio_sample_rate
        self.class_song_id = class_song_id
        self.shifting_time = shifting_time
        self.start_position = start_position
        self.evaluation_length=evaluation_length
        self.df = get_file_list(self._base_dir, self.class_song_id).reset_index(drop=True)

        if self.train_test_splitting=='random_split_5s':
            self.df = get_5s_file(self.df)

        if self.train_test_splitting=='random_split_30s':
            self.df = get_30s_file(self.df)

        if self.subset == "CV":
            self.df_subset =self.df

        else:
            df_train, df_test = train_test_split(self.df, test_size=0.25, random_state=split_seed, stratify=self.df.song)
            if self.subset == "train":
                self.df_subset = df_train
            elif self.subset == "test" or self.subset == "valid":
                self.df_subset = df_test

        df_train, df_test = train_test_split(self.df, test_size=0.25, random_state=split_seed, stratify=self.df.song)
        new_valid_df=get_window(df_test,self.chunk_length,self.window_size,self.stride)

        if self.subset == "SW_train":
            self.df_subset = df_train
        elif self.subset == "SW_test" or self.subset == "SW_valid":
            self.df_subset = new_valid_df
            self.df_subset = self.df_subset[self.df_subset.iloc[:, 6] >= (self.audio_clip-3)/2]
            self.accessed_data=check_accessed_data(self.chunk_length,self.window_size,self.stride)
        if self.subset == "train":
            self.df_subset = df_train
        elif self.subset == "test" or self.subset == "valid":
            self.df_subset = df_test


    def set_random_numbers(self, random_numbers):
        self.random_numbers = random_numbers

    def set_sliding_window_parameters(self, window_size, stride):
        self.window_size = window_size
        self.stride = stride

    def labels(self):
        num_label = len(self.df_subset.song.unique())
        return num_label

    def set_eeg_normalization(self, eeg_normalization, clamp_value=None):
        self.eeg_normalization = eeg_normalization
        self.clamp_value = clamp_value
    
    def K_split(self, k=4, random_state=42):
        train_list = [[] for _ in range(k)]
        valid_list = [[] for _ in range(k)]

        for song_id in self.df_subset.song.drop_duplicates().values:
            df_song = self.df_subset[self.df_subset.song == song_id]
            all_list_song = df_song.index.tolist()
            
            for i in range(k):
                valid_list_song_fold = df_song.subject.drop_duplicates().sample(6, random_state=random_state).index.tolist()
                train_list_song_fold = [x for x in all_list_song if x not in valid_list_song_fold]
                valid_list[i].extend(valid_list_song_fold)
                train_list[i].extend(train_list_song_fold)

                df_song = df_song.drop(valid_list_song_fold)

        Kfold_list = [[np.array(train_fold), np.array(valid_fold)] for train_fold, valid_fold in zip (train_list, valid_list)]
        
        return Kfold_list

    def K_split_random(self, k=4, random_state=42):
        kf = KFold(n_splits=k, shuffle=True, random_state=random_state)
        return (kf.split(self.df_subset))
    
    def getitem(self, n, isClip=True):
        p=self.start_position
        eeg_path = self.df_subset.iloc[n, 3]
        with open(eeg_path, 'rb') as f:
            eeg = pickle.load(f)

        if self.train_test_splitting == 'random_split_5s':
            chunk = self.df_subset.iloc[n, 5]
            eeg = eeg[:, int(int(chunk)*(125*5)+(0.125*self.shifting_time)):int(int(chunk+1)*(125*5)+(0.125*self.shifting_time))]
        if self.train_test_splitting == 'random_split_30s':   
            chunk = self.df_subset.iloc[n, 5]
            eeg = eeg[:, int(int(chunk)*(125*30)+(0.125*self.shifting_time)):int(int(chunk+1)*(125*30)+(0.125*self.shifting_time))]
        if self.subset == "SW_valid":   
            window = self.df_subset.iloc[n, 6]
            eeg = eeg[:, int((window)*self.stride)+p:int((window)*self.stride+self.window_size)+p] 
        
        audio_path = self.df_subset.iloc[n, 4]
        with open(audio_path, 'rb') as f:
            whole_audio, audio_sample_rate = torchaudio.load(f)

        if self.train_test_splitting == 'random_split_5s':
            chunk = self.df_subset.iloc[n, 5]
            audio = whole_audio[:, int(chunk)*(44100*5)+p:int(chunk+1)*(44100*5)+p]

        if self.train_test_splitting == 'random_split_30s':   
            chunk = self.df_subset.iloc[n, 5]
            audio = whole_audio[:, int(chunk)*(44100*30)+p:int(chunk+1)*(44100*30)+p]

        if self.subset == "SW_valid":   
            window = self.df_subset.iloc[n, 6]
            audio = audio[:, int(window*self.stride*44100/125)-int((self.audio_clip-3)/2*44100)+p:int((window*self.stride+self.window_size)*44100/125)+int((self.audio_clip+3)/2*44100)+p] 
        
        label = self.df_subset.iloc[n,1]
        subject = self.df_subset.iloc[n, 0]

        EEG_list=[]
        Audio_list=[]
        
        if isClip:  
            eeg_length = eeg.size(1)
            eeg_start = random.randint(int((self.audio_clip-3)/2*125), int(eeg_length-self.evaluation_length-(self.audio_clip-3)/2*125-1))
            audio_start = int(eeg_start/125 * 44100)+int((self.audio_clip-3)/2*125)
            evaluation_length=self.evaluation_length
            
            for i in range(int((evaluation_length-375)/125+1)):
                eeg_data = eeg[:, eeg_start+int(self.eeg_sample_rate*i*1) : eeg_start + self.eeg_clip_length+int(self.eeg_sample_rate*i*1)].clone()         
                audio_data = audio[:, int(int(audio_start)+44100*i*1) : int(int(audio_start + self.audio_clip * 44100)+44100*i*1)].clone()
                debug_logger.debug(f"audio, {audio_data.shape}")
                
                if self.transform != None:
                    eeg_data = eeg_data.to('cpu').detach().numpy().copy()
                    eeg_data = self.transform(eeg_data, sample_rate=self.eeg_sample_rate)
                    eeg_data = torch.from_numpy(eeg_data.astype(np.float32)).clone()

                    audio_data = audio_data.to('cpu').detach().numpy().copy()
                    audio_data = self.transform(audio_data, sample_rate=self.eeg_sample_rate)
                    audio_data = torch.from_numpy(audio_data.astype(np.float32)).clone()
                
                if self.eeg_normalization == "channel_mean":
                    eeg_data = self.normalize_EEG(eeg_data)
                elif self.eeg_normalization == "all_mean":
                    eeg_data = self.normalize_EEG_2(eeg_data)
                elif self.eeg_normalization == "constant_multiple":
                    eeg_data = self.normalize_EEG_3(eeg_data)
                elif self.eeg_normalization == "MetaAI":
                    eeg_data = self.normalize_EEG_4(eeg_data, self.clamp_value)
                
                if self.eeg_clip_length == 375:
                    padding = (int((3**6-self.eeg_clip_length)/2), int((3**6-self.eeg_clip_length)/2))
                    eeg_data = F.pad(eeg_data, padding)
                    if eeg_data.shape[1] != 3**6:
                        padding = (0, 3**6-eeg_data.shape[1])
                        eeg_data = F.pad(eeg_data, padding)
                    if self.audio_clip <= 4:
                        padding2 = (int((3**11-self.audio_length)/2), int((3**11-self.audio_length)/2))
                        audio_data = F.pad(audio_data, padding2)
                        if audio_data.shape[1] != 3**11:
                            padding2 = (0, 3**11-audio_data.shape[1])
                            audio_data = F.pad(audio_data, padding2)
                    if self.audio_clip >= 4:
                        padding2 = (int((3**12-self.audio_length)/2), int((3**12-self.audio_length)/2))
                        audio_data = F.pad(audio_data, padding2)
                        if audio_data.shape[1] != 3**12:
                            padding2 = (0, 3**12-audio_data.shape[1])
                            audio_data = F.pad(audio_data, padding2)
                if self.eeg_clip_length == 125:
                    padding = (int((3**5-self.eeg_clip_length)/2), int((3**5-self.eeg_clip_length)/2))
                    eeg_data = F.pad(eeg_data, padding)
                    if eeg_data.shape[1] != 3**5:
                        padding = (0, 3**5-eeg_data.shape[1])
                        eeg_data = F.pad(eeg_data, padding)
                    padding2 = (int((3**10-self.audio_length/3)/2), int((3**10-self.audio_length/3)/2))
                    audio_data = F.pad(audio_data, padding2)
                    if audio_data.shape[1] != 3**10:
                        padding2 = (0, 3**10-audio_data.shape[1])
                        audio_data = F.pad(audio_data, padding2)
                if self.eeg_clip_length == 1125:
                    padding = (int((3**7-self.eeg_clip_length)/2), int((3**7-self.eeg_clip_length)/2))
                    eeg_data = F.pad(eeg_data, padding)
                    if eeg_data.shape[1] != 3**7:
                        padding = (0, 3**7-eeg_data.shape[1])
                        eeg_data = F.pad(eeg_data, padding)
                    padding2 = (int((3**12-self.audio_length)/2), int((3**12-self.audio_length)/2))
                    audio_data = F.pad(audio_data, padding2)
                    if audio_data.shape[1] != 3**12:
                        padding2 = (0, 3**12-audio_data.shape[1])
                        audio_data = F.pad(audio_data, padding2)
                else:
                    assert "eeg_clip_length error"
                EEG_list.append(eeg_data)
                Audio_list.append(audio_data)
           
        return EEG_list, Audio_list, label, subject


    def __getitem__(self, n: int) -> Tuple[Tensor, Tensor]:
        eeg_list, audio_list, song_id, subjects = self.getitem(n)
        return eeg_list, audio_list, song_id, subjects
        
    def __len__(self) -> int:
        return len(self.df_subset)

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
        for idx, ch_eeg in enumerate(eeg):
            transformer = RobustScaler().fit(ch_eeg.view(-1,1))
            ch_eeg = transformer.transform(ch_eeg.view(-1,1))
            ch_eeg = torch.from_numpy(ch_eeg.astype(np.float32)).clone()
            eeg[idx] = ch_eeg.view(1,-1)

        eeg = torch.clamp(eeg, min=int(-1*clamp_value), max=int(clamp_value))

        return eeg


    def get_last_iteration_value(self):
        if self.subset == "SW_valid": 
            
            return self.start_value

    def check_access(self,n):
        if self.subset == "SW_valid": 
            self.accessed_data[n][self.eeg_start:self.eeg_start + self.eeg_clip_length] = 1
            
            if np.all(self.accessed_data[n] == 1):
                debug_logger.debug(f"All data in item {n} has been accessed.")
            full_ones_rows = np.all(self.accessed_data == 1, axis=1).sum()
            percentage = (full_ones_rows / len(self.df_subset)) * 100
            if n==len(self.df_subset)-1:
                ones_ratio = np.mean(self.accessed_data[n] == 1)
                self.start.append(self.eeg_start)
                debug_logger.debug(f"The start array is:{self.start}")
                debug_logger.debug(f"The data is:{self.df_subset.iloc[n]}")
                debug_logger.debug(f"The proportion of 1 is:{ones_ratio}")
                debug_logger.debug(f"The percentage of rows in 'accessed_data' where all elements are 1 is: {percentage}%")

        return self.start




