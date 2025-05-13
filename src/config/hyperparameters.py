"""
Hyper parameter configuration for the MNIST Digit Recognition model.
"""

import torch

# MODEL ARCHITECTURE PARAMETERS
INPUT_SIZE = 784
HIDDEN_SIZE = 128
OUTPUT_SIZE = 10
CONV_CHANNELS1 = 64
CONV_CHANNELS2 = 128
CONV_CHANNELS3 = 256
KERNEL_SIZE = 3
FC_SIZE = 128

# TRAINING PARAMETERS
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 5e-6
DROPOUT_RATE = 0.3
MAX_EPOCHS = 40
BATCH_SIZE = 64
VAL_BATCH_SIZE = BATCH_SIZE * 2

# DATA PARAMETERS
VAL_SPLIT = 0.1667

# PATHS AND OTHER SETTINGS
CHECKPOINTS_DIR = "checkpoints"
WANDB_PROJECT = "pytorch-mnist-ocr"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# HYPERPARAMETER TUNING
OPTUNA_TRIALS = 20
STUDY_NAME = "optuna_study"
STORAGE = "sqlite:///optuna_studies.db"

# Imported into the notebook
hyperparameters = {
    "input_size": INPUT_SIZE,
    "hidden_size": HIDDEN_SIZE,
    "output_size": OUTPUT_SIZE,
    "conv_channels1": CONV_CHANNELS1,
    "conv_channels2": CONV_CHANNELS2,
    "conv_channels3": CONV_CHANNELS3,
    "kernel_size": KERNEL_SIZE,
    "fc_size": FC_SIZE,
    "learning_rate": LEARNING_RATE,
    "weight_decay": WEIGHT_DECAY,
    "dropout_rate": DROPOUT_RATE,
    "max_epochs": MAX_EPOCHS,
    "batch_size": BATCH_SIZE,
    "val_batch_size": VAL_BATCH_SIZE,
    "val_split": VAL_SPLIT,
    "checkpoints_dir": CHECKPOINTS_DIR,
    "wandb_project": WANDB_PROJECT,
    "device": DEVICE,
    "optuna_trials": OPTUNA_TRIALS,
    "study_name": STUDY_NAME,
    "storage": STORAGE,
}
