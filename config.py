import os

# Directories
DATASET_DIR = os.path.join("data", "dataset")
OUTPUT_DIR = os.path.join("data", "preprocessed_dataset")
H5_FILE = "data/preprocessed_dataset/preprocessed_data.h5"
DATASET_PATH = "data/preprocessed_dataset/preprocessed_data.h5"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Dataset parameters
BATCH_SIZE = 1
LABEL_MAPPING = {"HANDS": 0, "RUNNING": 0, "WALKING": 1, "SITTING": 1}

# Preprocessing parameters
PADDING_MULTIPLE = 8
TRAIN_TEST_SPLIT = 0.8

# Hyperparameters for the model
MODEL_HYPERPARAMETERS = {
    "batch_size": 1,
    "epochs": 10,
    "noise_factor": 0.01,
}
