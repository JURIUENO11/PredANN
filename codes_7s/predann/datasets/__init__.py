import os
from .dataset import Dataset
from .nmed_dataset import NMED_dataset
from .preprocessing_eegmusic_dataset_7s import Preprocessing_EEGMusic_dataset


def get_dataset(dataset, dataset_dir, subset ,download=True):

    if not os.path.exists(dataset_dir):
        os.makedirs(dataset_dir)

    if dataset == "audio":
        d = AUDIO(root=dataset_dir)
    elif dataset == "librispeech":
        d = LIBRISPEECH(root=dataset_dir, download=download, subset=subset)
    elif dataset == "gtzan":
        d = GTZAN(root=dataset_dir, download=download, subset=subset)
    elif dataset == "magnatagatune":
        d = MAGNATAGATUNE(root=dataset_dir, download=download, subset=subset)
    elif dataset == "magnatagatune_ssa":
        d = MAGNATAGATUNE_SSA(root=dataset_dir, download=download, subset=subset)
    elif dataset == "magnatagatune_add_non_tag":
        d = MAGNATAGATUNE_ADD_NONTAG(root=dataset_dir, download=download, subset=subset)
    elif dataset == "msd":
        d = MillionSongDataset(root=dataset_dir, subset=subset)
    elif dataset == "msd2":
        d = MillionSongDataset2(root=dataset_dir, subset=subset)
    elif dataset == "msd2mel":
        d = MillionSongDataset2Mel(root=dataset_dir, subset=subset)
    elif dataset == "msd2_add_non_tag":
        d = MillionSongDataset2_ADD_NONTAG(root=dataset_dir, subset=subset)
    elif dataset == "msd2mel_add_non_tag":
        d = MillionSongDataset2Mel_ADD_NONTAG(root=dataset_dir, subset=subset)
    elif dataset == "msd3":
        d = MillionSongDataset3(root=dataset_dir, subset=subset)
    elif dataset == "msd3mel":
        d = MillionSongDataset3Mel(root=dataset_dir, subset=subset)
    elif dataset == "msd3_add_non_tag":
        d = MillionSongDataset3_ADD_NONTAG(root=dataset_dir, subset=subset)
    elif dataset == "msd3mel_add_non_tag":
        d = MillionSongDataset3Mel_ADD_NONTAG(root=dataset_dir, subset=subset)
    elif dataset == "fma":
        d = FMADataset(root=dataset_dir, subset=subset)
    elif dataset == "mtgj":
        d = MtgJDataset(root=dataset_dir, subset=subset)
    elif dataset == "mtgj_ssa":
        d = MtgJDataset_SSA(root=dataset_dir, subset=subset)
    elif dataset == "mtgj_add_non_tag":
        d = MtgJDataset_ADD_NONTAG(root=dataset_dir, subset=subset)
    elif dataset == "magnatagatune_mel":
        d = MAGNATAGATUNE_MEL(root=dataset_dir, download=download, subset=subset)
    elif dataset == "magnatagatune_mel_add_non_tag":
        d = MAGNATAGATUNE_MEL_ADD_NONTAG(root=dataset_dir, download=download, subset=subset)
    elif dataset =="mtgjmel":
        d = MtgJDatasetMel(root=dataset_dir, subset=subset)
    elif dataset =="mtgjmel_add_non_tag":
        d = MtgJDatasetMel_ADD_NONTAG(root=dataset_dir, subset=subset)
    elif dataset == "magnatagatune_triplet":
        d = MAGNATAGATUNE_TRIPLET(root=dataset_dir, download=download, subset=subset)
    elif dataset == "lmd":
        d = LMD_contrastive_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "openmiir":
        d = OpenMIIR_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "openmiir_classification":
        d = OpenMIIR_classification_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "openmiir_classification2":
        d = OpenMIIR_classification_dataset2(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmed":
        d = NMED_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmed_classification":
        d = NMED_classification_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmed_clean_classification":
        d = NMED_clean_classification_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmed_audio":
        d = NMED_audio_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmedt_audio":
        d = NMED_audio_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "preprocessing_eeg":
        d = Preprocessing_EEG_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "preprocessing_eegmusic":
        d = Preprocessing_EEGMusic_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmed_2d_classification":
        d = NMED_2Dclassification_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmed_clean_2d_classification":
        d = NMED_clean_2Dclassification_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmedt_SW_2d_classification":
        d = NMEDT_SW_2Dclassification_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmedt":
        d = NMEDT_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmedt_downsample":
        d = NMEDT_DS_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmedt_downsample_classification":
        d = NMEDT_DS_classification_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmedt_clean_classification":
        d = NMEDT_clean_classification_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "nmedt_slidingWindow_classification":
        d = NMEDT_SW_classification_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "preprocessing_NMEDH":
        d = Preprocessing_NMEDH_dataset(root=dataset_dir, download=download, subset=subset)
    elif dataset == "preprocessing_2deegmusic":
        d = Preprocessing_2DEEGMusic_dataset(root=dataset_dir, download=download, subset=subset)    
    else:
        raise NotImplementedError("Dataset not implemented")
    return d
