from constants import NUM_EPOCHS, HF_DS_PATH, BATCH_SIZE, DATASET_EXISTS, NUM_FOLDS
from datasets import load_dataset
from train_utils import preprocess_data, train_model_with_cv
from create_dataset import create_image_dataset
from create_graphs import cnn_plot

"""
Train models using cross-validation for heartbeat classification.

If the dataset does not exist, create it using create_image_dataset().
Load the dataset from Hugging Face, preprocess the data, and obtain data loaders.
Train models with cross-validation using train_model_with_cv().

Parameters:
None

Returns:
None
"""
# Create dataset if not already created
if not DATASET_EXISTS:
    create_image_dataset()

# Load dataset from Hugging Face
dataset = load_dataset(HF_DS_PATH)
dataset, _, testloader = preprocess_data(dataset, BATCH_SIZE)

# Train models with cross-validation
results = train_model_with_cv(dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, num_folds=NUM_FOLDS)
cnn_plot(results)
