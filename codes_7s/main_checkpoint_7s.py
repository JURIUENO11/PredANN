import argparse
import torch
from torch.utils.data import DataLoader
import pytorch_lightning as pl
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
from audiomentations import AddGaussianNoise, Gain
from predann.datasets import get_dataset
from predann.models import SampleCNN2DEEG
from predann.modules import EEGContrastiveLearning
from predann.utils import yaml_config_hook
import pandas as pd
import datetime
import random

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="PredANN")

    config = yaml_config_hook("./config/config.yaml")
    for k, v in config.items():
        parser.add_argument(f"--{k}", default=v, type=type(v))
    parser.add_argument('--start_position', type=int)
    parser.add_argument('--evaluation_length', type=int)

    args = parser.parse_args()
    pl.seed_everything(args.seed, workers=True)

    train_transform = {}
    if args.openmiir_augmentation == "gaussiannoise":
        train_transform = [
            AddGaussianNoise(min_amplitude=args.min_amplitude, max_amplitude=args.max_amplitude, p=0.5),
        ]
        print("augematation is gaussiannoise")

    elif args.openmiir_augmentation == "gain":
        train_transform = [
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5)
        ]
        print("augematation is gain")

    elif args.openmiir_augmentation == "gaussiannoise+gain":
        train_transform = [
            AddGaussianNoise(min_amplitude=args.min_amplitude, max_amplitude=args.max_amplitude, p=0.5),
            Gain(min_gain_in_db=-12, max_gain_in_db=12, p=0.5)
        ]
        print("augematation is gaussiannoise+gain")

    else:
        print("no augmentation")
        
    train_log = pd.DataFrame(columns=["Loss/train", "Accuracy/train_eeg", "Accuracy/train_audio"])
    valid_log = pd.DataFrame(columns=["Loss/valid", "Accuracy/valid_eeg", "Accuracy/valid_audio"])

    train_dataset = get_dataset(args.dataset, args.dataset_dir, subset="SW_train", download=False)
    train_dataset.set_sliding_window_parameters(args.window_size, args.stride)
    train_dataset.set_eeg_normalization(args.eeg_normalization, args.clamp_value)
    train_dataset.set_other_parameters(args.eeg_length, args.audio_clip_length,args.split_seed, args.class_song_id,args.shifting_time, args.start_position,args.evaluation_length)
    random.seed(42)
    train_random_numbers = [random.randint(0, 125 * 30 - 375 - 1) for _ in range(1200)]
    train_dataset.set_random_numbers(train_random_numbers)


    if args.openmiir_augmentation != "no_augmentation":
        train_dataset.set_transform(train_transform)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=True,
    )

    valid_dataset = get_dataset(args.dataset, args.dataset_dir, subset="SW_valid", download=False)
    valid_dataset.set_sliding_window_parameters(args.window_size, args.stride)
    valid_dataset.set_eeg_normalization(args.eeg_normalization, args.clamp_value)
    valid_dataset.set_other_parameters(args.eeg_length,args.audio_clip_length, args.split_seed, args.class_song_id,args.shifting_time, args.start_position,args.evaluation_length)
    random.seed(42)
    valid_random_numbers = [random.randint(0, args.window_size - 375 - 1) for _ in range(1200)]
    valid_dataset.set_random_numbers(valid_random_numbers)  

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=args.batch_size,
        num_workers=args.workers,
        drop_last=True,
        shuffle=False,
    )
    print(f"Size of train dataset: {len(train_dataset)}")
    print(f"Size of valid dataset: {len(valid_dataset)}")


    if args.dataset == "preprocessing_eegmusic":
        encoder_eeg = SampleCNN2DEEG(
            out_dim=train_dataset.labels(),
            kernal_size=3,
        ) 
        encoder_audio = SampleCNN2DEEG(
            out_dim=train_dataset.labels(),
            kernal_size=3,
        )
    print('EEG Contrastive learning')
    module = EEGContrastiveLearning(valid_dataset, args, encoder_eeg, encoder_audio)
    logger = TensorBoardLogger("runs/{}".format(args.training_date), name="nmed-CL-{}".format(args.dataset))


    early_stop_callback = EarlyStopping(
        monitor="Valid/loss", patience=10
    )
    trainer = Trainer.from_argparse_args(
        args,
        logger=logger,
        sync_batchnorm=True,
        max_epochs=args.max_epochs,
        deterministic=True,
        log_every_n_steps=1,
        check_val_every_n_epoch=1,
        accelerator=args.accelerator,
        resume_from_checkpoint="checkpoint_example.ckpt"
    )
    print('[[[ START ]]]',datetime.datetime.now())
   
    checkpoint_path = "checkpoint_example.ckpt"
    checkpoint = torch.load(checkpoint_path)
    module.load_state_dict(checkpoint['state_dict'])
    trainer.validate(module,dataloaders=valid_loader)
    print('[[[ FINISH ]]]',datetime.datetime.now())
