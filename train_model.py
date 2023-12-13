from constants import NUM_EPOCHS, HF_DS_PATH, BATCH_SIZE, DATASET_EXISTS
from pytorch_cnn import Net
from datasets import load_dataset
from train_utils import train_model, save_model, preprocess_data, test_model_conf_mat
from create_dataset import create_image_dataset

"""
Train a Convolutional Neural Network (CNN) for heartbeat classification.

If the dataset does not exist, create it using create_image_dataset().
Load the dataset from Hugging Face, preprocess the data, and obtain data loaders.
Create a CNN, train it, test it, and save the model.
"""
# Create dataset if not already created
if not DATASET_EXISTS:
    create_image_dataset()

# Load dataset from Hugging Face
dataset = load_dataset(HF_DS_PATH)

# Preprocess data and get data loaders
_, trainloader, testloader = preprocess_data(dataset, BATCH_SIZE)

# Create, train, test, and save CNN
net = Net()
train_model(net, trainloader, NUM_EPOCHS)
test_model_conf_mat(net, testloader)
save_model(net)