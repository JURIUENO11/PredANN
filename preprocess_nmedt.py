import os
import numpy as np
import mne
import matplotlib.pyplot as plt
from scipy import interpolate
import tarfile
import glob
#import re
import regex as re
import pandas as pd
from joblib import Parallel, delayed
import pickle
import soundfile as sf
import argparse
import scipy.io
import torch
import mat73
from scipy import signal

def parse_din(din):
    triggers = din[0]
    all_trigger = []
    for trigger in triggers:
        
        trigger = re.sub(r"\D", "", trigger)
        all_trigger.append(trigger)

    # Extract audio onsets and Divide by 8 according to the sampling frequency
    onsets = din[1]
    all_onsets=[int(int(y)) for y in onsets]

    return all_trigger, all_onsets

def ExtractEEG(path, base_folder, original_song_list,ds):
    # basename is 'S{subject_id}_{trial_id}.mat'
    subject_id=re.findall(r"\d+", path)[-2]
    subject_id=int(subject_id)
    trial_id=re.findall(r"\d+", path)[-1]
    trial_id=int(trial_id)

    Dictionary = mat73.loadmat(path)
    din = Dictionary['DIN_1']
    all_trigger, all_onsets = parse_din(din)
    sfreq = int(Dictionary['fs'])

    for idx, song_id in enumerate(original_song_list[0]):
        if song_id in all_trigger:
            # The same song is not played more than once in a file.
            trigger_idx = all_trigger.index(song_id)
            # The accurate time stamp of a stimulus onset can be computed by subtracting 1 second (125 time samples) 
            # from the 128 trigger immediately following a 21:36 trigger.
            start_idx = trigger_idx+1  
            assert all_trigger[start_idx]==128, "Trigger Error"
            # Refer https://stacks.stanford.edu/file/druid:jn859kj8079/NMED-T_README.pdf
            onset = all_onsets[start_idx]-1000

            # Extract EEG listening target audio
            audio_length = original_song_list[3][idx]
            # Audio sampling rate is 44100Hz
            EEG_length = int(audio_length/44100*sfreq)
            original_eeg = Dictionary['X']
            # Remove vertex reference row of zeros in electrode 129
            original_eeg_del = np.delete(original_eeg, 128, 0)
            EEG = original_eeg_del[:, onset:onset+EEG_length-1]

            # Save EEG as a pickle file.
            new_file=base_folder+f"/{subject_id}_{song_id}_1.pkl"

            if ds == 'yes':
               signal_size = EEG.shape
               EEG = signal.resample(EEG, int(signal_size[1]/8), axis=1) 
               new_file=base_folder+f"/{subject_id}_{song_id}_1.pkl"

            EEG = torch.from_numpy(EEG.astype(np.float32)).clone()

            with open(new_file,"wb") as f:
                pickle.dump(EEG,f)
            print(new_file,"is saved")


def ExtractAUDIO(path, new_audio_folder, original_song_list):
    basename = os.path.basename(path)
    if basename in original_song_list[4]:
        idx = original_song_list[4].index(basename)
        song_id = original_song_list[0][idx]
        # Convert audio to monoral
        data, sfreq = sf.read(path)
        #data = np.mean(data, axis=1)

        audio_onset = original_song_list[1][idx]
        audio_length = original_song_list[3][idx]
        data = data[audio_onset: audio_onset+audio_length-1]
        
        # Save audio 
        new_file=new_audio_folder+f"/{song_id}.wav"
        sf.write(new_file, data, sfreq)
        print(new_file, "is saved")
    else:
        print(basename, 'is not supported')

def ExtractCleanEEG(path, new_eeg_base):
    song_id = re.findall(r"\d+", path)[0]

    Dictionary = scipy.io.loadmat(path)
    subject_key = list(Dictionary.keys())[-1]
    data_key = list(Dictionary.keys())[-3]
    subject_list = Dictionary[subject_key][0]
    data_list = Dictionary[data_key]

    for idx, sub in enumerate(subject_list):
        sub = np.array2string(sub[0])  
        subject_id = re.findall(r"\d+", sub)[0]
        subject_id = int(subject_id)

        EEG = data_list[:,:,idx]
        EEG = torch.from_numpy(EEG.astype(np.float32)).clone()

        # Save EEG as a pickle file.
        #new_file=new_eeg_base+f"/{subject_id}_{song_id}_{trial_id}.pkl"
        new_file=new_eeg_base+f"/{subject_id}_{song_id}_1.pkl"
        with open(new_file,"wb") as f:
            pickle.dump(EEG,f)
        print(new_file,"is saved")

if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=str, default='H_raw')
    parser.add_argument("--downsample", type=str, default='no')
    args = parser.parse_args()
    
    if args.dataset == 'H_raw' or 'H_audio':
        # original song information table
        # https://stacks.stanford.edu/file/druid:jn859kj8079/NMED-T_README.pdf
        trigger_id = [21,22,23,24,
                    25,26,27,28,
                    29,30]
        start_data = [0,0,0,0,
                    0,0,0,0,
                    0,0]
        end_data = [278*44100, 271*44100, 276*44100, 294*44100,
                    289*44100, 276*44100, 292*44100, 292*44100,
                    292*44100, 298*44100]
        audio_length = [278*44100, 271*44100, 276*44100, 294*44100,
                        289*44100, 276*44100, 292*44100, 292*44100,
                        293*44100, 298*44100]
        file_names = ['FirstFires.wav','Oino.wav','Tiptoes.wav','CarelessLove.wav',
                    'LebaneseBlonde.wav','Canopee.wav','DoingYoga.wav','UntilTheSunNeedsToRise.wav',
                    'SilentShout.wav','TheLastThingYouShouldDo.wav']
        original_song_list = [trigger_id, start_data, end_data, audio_length, file_names]
        
        if args.dataset == "H_raw":
            # Preprocess raw EEG
            eeg_base = '/workdir/share/NMED/NMED-T/Raw_EEG'
            eeg_file_list = [p for p in glob.glob(os.path.join(eeg_base,'*.mat')) if os.path.isfile(p)]
            new_eeg_base = '/workdir/share/NMED/NMED-T/NMED-T_dataset/EEG_pkl'
            os.makedirs(new_eeg_base, exist_ok=True)
            ds=args.downsample
            if ds == 'yes':
                new_eeg_base = '/workdir/share/NMED/NMED-T/NMED-T_dataset/DS_EEG_pkl'

            for path in eeg_file_list:
                ExtractEEG(path, new_eeg_base, original_song_list,ds)

        elif args.dataset == "H_audio":
            # Preprocess Audio
            audio_base = '/workdir/share/NMED/NMED-T/Raw_audio/'
            audio_file_list = [p for p in glob.glob(os.path.join(audio_base,'*.wav')) if os.path.isfile(p)]
            new_audio_base = '/workdir/share/NMED/NMED-T/NMED-T_dataset/audio/'
            os.makedirs(new_audio_base, exist_ok=True)

            for path in audio_file_list:
                ExtractAUDIO(path, new_audio_base, original_song_list)

        for path in eeg_file_list:
            ExtractCleanEEG(path, new_eeg_base)
