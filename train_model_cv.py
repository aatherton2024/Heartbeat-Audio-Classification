from constants import NUM_EPOCHS, HF_DS_PATH, BATCH_SIZE, DATASET_EXISTS, NUM_FOLDS
from pytorch_cnn import Net
from datasets import load_dataset
from train_utils import preprocess_data, train_model_with_cv
from create_dataset import create_image_dataset

"""
File to train models using cross validation
"""

#Create dataset if not already created
if not DATASET_EXISTS: create_image_dataset()

#Load dataset from huggingface
dataset = load_dataset(HF_DS_PATH)
dataset, _, testloader = preprocess_data(dataset, BATCH_SIZE)
train_model_with_cv(dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, num_folds=NUM_FOLDS)


