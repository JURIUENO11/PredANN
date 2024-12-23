import torch
from pytorch_lightning import LightningModule
from . import PredANN_Loss
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
    def __init__(self, preprocess_dataset, args, encoder_eeg, encoder_audio):
        super().__init__()
        self.save_hyperparameters(args)

        self.encoder_eeg = encoder_eeg
        self.encoder_audio = encoder_audio
        self.criterion = self.configure_criterion()

        self.last_epoch_train_embeddings = []
        self.last_epoch_train_labels = []
        self.last_epoch_valid_embeddings = []
        self.last_epoch_valid_labels = []

        self.train_log_df = pd.DataFrame(columns=["Loss/train", "Accuracy/train_eeg", "Accuracy/train_audio"])
        self.valid_log_df = pd.DataFrame(columns=["Loss/valid", "Accuracy/valid_eeg", "Accuracy/valid_audio"])
        
        self.validation_end_values = []
        self.preprocess_dataset = preprocess_dataset
        self.batch_accuracies = []
        self.batch_accuracies_c = []
        self.label_accuracy_count = {label: {'correct': 0, 'total': 0} for label in range(10)}
        self.subject_accuracy_count = {subject: {'correct': 0, 'total': 0} for subject in range(24)}

    def forward(self, r, c):
        y_eeg, z_eeg = self.encoder_eeg(r)
        y_audio, z_audio = self.encoder_audio(c)
        return y_eeg, y_audio, z_eeg, z_audio
    
    def on_validation_start(self):
        self.eval()

    def validation_step(self, batch, batch_idx):
        
        eeg_data, audio_data = batch[:2]
        label = batch[2]
        subject = batch[3]
        subject = [int(sub) for sub in subject]
        subject = torch.tensor(subject, device=label.device)
 

        y_eeg_list=[]
        y_audio_list=[]
        z_eeg_list=[]
        z_audio_list=[]

       
        for eeg, audio in zip(eeg_data, audio_data):
            y_eeg, y_audio, z_eeg, z_audio = self.forward(eeg, audio)
            y_eeg_list.append(y_eeg)
            y_audio_list.append(y_audio)
            z_eeg_list.append(z_eeg)
            z_audio_list.append(z_audio)
            
        y_audio=self.tensor_calculate(y_audio_list)
        y_eeg=self.tensor_calculate(y_eeg_list)
        z_audio=self.tensor_calculate(z_audio_list)
        z_eeg=self.tensor_calculate(z_eeg_list)
      
        predann_loss = self.criterion(z_eeg, z_audio)
        ce_loss_r, acc_r_av = self._shared_step(label, y_eeg)
        ce_loss_c, acc_c_av = self._shared_step(label, y_audio)
        loss = self.hparams.weight_predann*predann_loss + self.hparams.weight_r*ce_loss_r + self.hparams.weight_c*ce_loss_c
        
        self.log("Loss/valid", loss)
        self.log("Accuracy/valid_eeg", acc_r_av)
        self.batch_accuracies.append(acc_r_av.item())
        self.log("Accuracy/valid_audio", acc_c_av)
        self.batch_accuracies_c.append(acc_c_av.item())

        new_valid_log = pd.DataFrame({"Loss/valid": [loss.item()], "Accuracy/valid_eeg": [acc_r_av.item()], "Accuracy/valid_audio": [acc_c_av.item()]})
        self.valid_log_df = pd.concat([self.valid_log_df, new_valid_log ], ignore_index=True)
  

        if self.current_epoch == self.trainer.max_epochs - 1:
            self.last_epoch_valid_embeddings.append(z_eeg.cpu().detach().numpy())
            self.last_epoch_valid_labels.extend(label.cpu().numpy())

        return loss

    
    
    
    def configure_criterion(self):
        if self.hparams.accelerator == "dp" and self.hparams.gpus:
            batch_size = int(self.hparams.batch_size / self.hparams.gpus)
        else:
            batch_size = self.hparams.batch_size
        
       
        if self.hparams.loss_function == "predann_loss":
            print("use PredANN_loss as criterion")
            criterion = PredANN_Loss(batch_size, self.hparams.temperature,self.hparams.detach_z_audio, world_size=1)
        else:
            print('error')
        return criterion

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(chain(self.encoder_eeg.parameters(),self.encoder_audio.parameters()), self.hparams.learning_rate)
        return {"optimizer": optimizer}


    def tensor_calculate(self,list):
        softmax_list = [torch.nn.functional.softmax(tensor, dim=1) for tensor in list]
        tensor = torch.stack(softmax_list).mean(dim=0)
        return tensor
   
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
            'encoder_eeg_state_dict': self.encoder_eeg.state_dict(),
            'encoder_audio_state_dict': self.encoder_audio.state_dict(),
            'optimizer_state_dict': self.trainer.optimizers[0].state_dict()
        }, filepath)

    def load_checkpoint(self, filepath):
        checkpoint = torch.load(filepath)
        self.load_state_dict(checkpoint['module_state_dict'])
        self.encoder_eeg.load_state_dict(checkpoint['encoder_eeg_state_dict'])
        self.encoder_audio.load_state_dict(checkpoint['encoder_audio_state_dict'])
    
        
        optimizer_config = self.configure_optimizers()
        optimizer = optimizer_config['optimizer']
        
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        return optimizer

    