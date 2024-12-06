# Predicting Artificial Neural Network Representations to Learn Recognition Model for Music Identification from Brain Recordings

## Neural Network Architecture
Two separate 2D CNNs are employed to process music and EEG data independently. The outputs include individual losses for music and EEG, along with a contrastive loss for learning the relationship between the two modalities.
![image](https://github.com/Mind-Music-PJ/paper/blob/main/Picture9.png?raw=true) 
## Code Structure and Files
### How to run codes
To execute the experiments, follow the steps below:

1. Navigate to the appropriate folder based on your experiment.
   - For instance, for a 3-second experiment, go to the **codes_3s** directory.

2. Execute the sequential script in the terminal using the command:

```nohup sh sequential_3s.sh > log/log.txt &```

4. The progress will be logged in ***log/log.txt***.
   - You can review this file for detailed outputs.
### Dataset Path Configuration
In both the 3-second (3s) and 7-second (7s) experiments, it is essential to modify the dataset path to match your local configuration. Specifically, in the files ***preprocessing_eegmusic_dataset_3s.py*** and ***preprocessing_eegmusic_dataset_7s.py***, the class ***Preprocessing_EEGMusic*** utilizes the dataset path to load data. By default, the path is set to ***_base_dir = "/workdir/share/NMED/NMED-T/NMED-T_dataset"***, which must be updated to reflect your local directory structure.

Sequential scripts (***sequential_3s.sh*** and ***sequential_7s.sh***) are provided to execute the experiments. These scripts include the necessary parameters for running the respective experiments.
#### Code Structure for 3-Second Training and Evaluation
```
predann/
├── datasets/                      
│   ├── __init__.py                # Initialization file for the datasets module for 3s experiments
│   ├── dataset.py                 # Base class for datasets
│   ├── nmed_dataset.py            # Script for handling the NMED dataset
│   └── preprocessing_eegmusic_dataset_3s.py # Preprocessing script for EEG and music data (3-second segments)
│
├── models/                        
│   ├── __init__.py                # Initialization file for the models module
│   ├── model.py                   # Base class for model definitions
│   └── sample_cnn2d_eeg.py        # Implementation of a 2D CNN model for EEG data
│
├── modules/                       
│   ├── __init__.py                # Initialization file for the modules in 3s experiments
│   ├── clip_loss.py               # Implementation of the Clip Loss function
│   └── contrastive_learning_3s.py # Script for contrastive learning using 3-second segments
│
├── utils/                         
│   ├── __init__.py                # Initialization file for the utilities module
│   ├── checkpoint.py              # Utility script for saving and loading model checkpoints
│   └── yaml_config_hook.py        # Utility script for loading YAML configuration files

config/                        
└── config.yaml                # Main configuration file for the project

log/                           
└── log.txt                    # Log file for recording training and evaluation progress

LICENSE                        # License information
main_3s.py                     # Main script for training and evaluating 3-second segments
requirements.txt               # Required Python packages   
sequential_3s.sh               # Script for running the 3s experiments
```
#### Code Structure for 7-Second Evaluation
```
predann/
├── datasets/                      
│   ├── __init__.py                # Initialization file for the datasets module for 7s experiments
│   ├── dataset.py                 
│   ├── nmed_dataset.py            
│   └── preprocessing_eegmusic_dataset_7s.py # Preprocessing script for EEG and music data (7-second segments)
│
├── models/                        
│   ├── __init__.py                
│   ├── model.py                   
│   └── sample_cnn2d_eeg.py        
│
├── modules/                      
│   ├── __init__.py                # Initialization file for the modules in 7s experiments
│   ├── clip_loss.py               
│   ├── evaluation_majority_7s.py  # Evaluation script using majority voting for 7-second segments
│   ├── evaluation_max_7s.py       # Evaluation script using maximum values for 7-second segments
│   └── evaluation_mean_7s.py      # Evaluation script using mean values for 7-second segments
│
├── utils/                         
│   ├── __init__.py                
│   ├── checkpoint.py             
│   └── yaml_config_hook.py        

config/                        
└── config.yaml               

log/                          
└── log.txt                   
                      
epoch-5999-step-149999.ckpt    # Saved model checkpoint
LICENSE  
main_checkpoint_7s.py          # Main script for evaluating 7-second segments
requirements.txt                     
sequential_7s.sh               # Script for running the 7s experiments
```
Please note that in the file ***main_checkpoint_7s.py***, the checkpoint path must be updated to correspond to the specific location of your checkpoint file. Additionally, the parameter ***evaluation_length*** can be modified to other durations, but it must not exceed 8 seconds (1000). Evaluations for 4s, 5s, 6s, and 7s have also been conducted using these codes by adjusting this parameter accordingly.

### Notes
- Ensure that the dataset is preprocessed before running experiments.
- Checkpoint paths must be correctly configured in the evaluation scripts to avoid runtime errors.
- The evaluation segment length parameter should not exceed 8 seconds to maintain consistency with the experimental design.
- The provided Sequential scripts (e.g., ***sequential_3s.sh ***,  ***sequential_7s.sh ***) are configured with default parameters. Users may adjust these parameters according to their specific experimental requirements.

## License
This project is under the CC-BY-SA 4.0 license. See [LICENSE](LICENSE) for details.

## Copyright
Copyright (c) 2024 Sony Computer Science Laboratories, Inc., Tokyo, Japan. All rights reserved. This source code is licensed under the [LICENSE](LICENSE).
