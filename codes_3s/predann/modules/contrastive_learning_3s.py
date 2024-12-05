import torch
from pytorch_lightning import LightningModule
from . import CLIP_Loss
import torch.nn.functional as F
import pandas as pd
import logging
import os
from itertools import chain

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

class EEGContrastiveLearning(LightningModule):
    def __init__(self, preprocess_dataset, args, encoder_raw_e, encoder_clean_e):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder_raw_e = encoder_raw_e
        self.encoder_clean_e = encoder_clean_e
        self.criterion = self.configure_criterion()

        self.last_epoch_train_embeddings = []
        self.last_epoch_train_labels = []
        self.last_epoch_valid_embeddings = []
        self.last_epoch_valid_labels = []

        self.train_log_df = pd.DataFrame(columns=["Loss/train", "Accuracy/train_raw", "Accuracy/train_clean"])
        self.valid_log_df = pd.DataFrame(columns=["Loss/valid", "Accuracy/valid_raw", "Accuracy/valid_clean"])
        
        self.validation_end_values = []
        self.preprocess_dataset = preprocess_dataset
        self.batch_accuracies = []
        self.batch_accuracies_c = []
        self.label_accuracy_count = {label: {'correct': 0, 'total': 0} for label in range(10)}
        self.subject_accuracy_count = {subject: {'correct': 0, 'total': 0} for subject in range(24)}

    def forward(self, r, c):
        y_r, z_r = self.encoder_raw_e(r)
        y_c, z_c = self.encoder_clean_e(c)
        return y_r, y_c, z_r, z_c
    

    def training_step(self, batch, batch_idx):
        raw_eeg, clean_eeg = batch[:2]
        y_r, y_c, z_r, z_c = self.forward(raw_eeg, clean_eeg)
        if self.hparams.detach_z_c:
            z_c = z_c.detach()
        clip_loss = self.criterion(z_r, z_c)

        label = batch[2]
        ce_loss_r, acc_r = self._shared_step(label, y_r)
        ce_loss_c, acc_c = self._shared_step(label, y_c)

        loss = self.hparams.weight_clip*clip_loss + self.hparams.weight_r*ce_loss_r + self.hparams.weight_c*ce_loss_c
        self.log("Loss/train", loss)
        self.log("Accuracy/train_raw", acc_r)
        self.log("Accuracy/train_clean", acc_c)

        new_train_log = pd.DataFrame({"Loss/train": [loss.item()], "Accuracy/train_raw": [acc_r.item()], "Accuracy/train_clean": [acc_c.item()]})
        self.train_log_df = pd.concat([self.train_log_df, new_train_log ], ignore_index=True)

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.last_epoch_train_embeddings.append(z_r.cpu().detach().numpy())
            self.last_epoch_train_labels.extend(label.cpu().numpy())
            
        return loss


    def on_validation_start(self):
        self.eval()

    def validation_step(self, batch, batch_idx):
        raw_eeg, clean_eeg = batch[:2]

        y_r, y_c, z_r, z_c = self.forward(raw_eeg, clean_eeg)
        if self.hparams.detach_z_c:
            z_c = z_c.detach()
        clip_loss = self.criterion(z_r, z_c)
        
        label = batch[2]


        ce_loss_r, acc_r = self._shared_step(label, y_r)
        ce_loss_c, acc_c = self._shared_step(label, y_c)

        self.batch_accuracies.append(acc_r.item())
        self.batch_accuracies_c.append(acc_c.item())

        loss = self.hparams.weight_clip*clip_loss + self.hparams.weight_r*ce_loss_r + self.hparams.weight_c*ce_loss_c

        self.log("Loss/valid", loss)
        self.log("Accuracy/valid_raw", acc_r)
        self.log("Accuracy/valid_clean", acc_c)

        new_valid_log = pd.DataFrame({"Loss/valid": [loss.item()], "Accuracy/valid_raw": [acc_r.item()], "Accuracy/valid_clean": [acc_c.item()]})
        self.valid_log_df = pd.concat([self.valid_log_df, new_valid_log ], ignore_index=True)
        return loss
    
    def configure_criterion(self):
        if self.hparams.accelerator == "dp" and self.hparams.gpus:
            batch_size = int(self.hparams.batch_size / self.hparams.gpus)
        else:
            batch_size = self.hparams.batch_size
        
        
        if self.hparams.loss_function == "clip_loss":
            print("use CLIP_loss as criterion")
            criterion = CLIP_Loss(batch_size, self.hparams.temperature, world_size=1)
        else:
            print('error')
           
        return criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.encoder_raw_e.parameters(),self.encoder_clean_e.parameters()), self.hparams.learning_rate)
        return {"optimizer": optimizer}


   
    def _shared_step(self, label, y):
        y_hat = y
        y = label

        loss = F.cross_entropy(y_hat, y)
        preds = torch.argmax(y_hat, dim=1)
        acc = (y == preds).sum() / y.size(0)
        return loss, acc
   
        

    def Kfold_log(self):
        return self.train_log_df, self.valid_log_df

    def save_checkpoint(self, filepath):
        torch.save({
            'module_state_dict': self.state_dict(),
            'encoder_raw_e_state_dict': self.encoder_raw_e.state_dict(),
            'encoder_clean_e_state_dict': self.encoder_clean_e.state_dict(),
            'optimizer_state_dict': self.trainer.optimizers[0].state_dict()
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['module_state_dict'])
        self.encoder_raw_e.load_state_dict(checkpoint['encoder_raw_e_state_dict'])
        self.encoder_clean_e.load_state_dict(checkpoint['encoder_clean_e_state_dict'])
     
        
        optimizer_config = self.configure_optimizers()
        optimizer = optimizer_config['optimizer']
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return optimizer

   