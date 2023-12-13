#: The directory where audio data is stored.
DATA_DIRECTORY = "/data/DHD/audio/"  # Replace with your relative data path

#: The directory where images are stored.
IMAGES_DIRECTORY = "/images/"  # Replace with your relative data path

#: The Hugging Face dataset path used to load the dataset.
HF_DS_PATH = "aatherton2024/heartbeat_images_final_project"

#: The batch size used during training.
BATCH_SIZE = 64

#: The number of epochs for training.
NUM_EPOCHS = 50

#: The number of folds used in cross-validation.
NUM_FOLDS = 10

#: Indicates whether the dataset exists or not. If false Hugginface dataset will be regenerated from .wav files
DATASET_EXISTS = True
