from constants import NUM_EPOCHS, HF_DS_PATH, BATCH_SIZE, DATASET_EXISTS
from pytorch_cnn import Net
from datasets import load_dataset
from train_utils import train_model, test_model, save_model, preprocess_data, test_model_conf_mat
from create_dataset import create_image_dataset

"""
File to train CNN
"""

#Create dataset if not already created
if not DATASET_EXISTS: create_image_dataset()

#Load dataset from huggingface
dataset = load_dataset(HF_DS_PATH)

#Preprocess data and get dataloaders
trainloader, testloader = preprocess_data(dataset, BATCH_SIZE)

#Create, train, test, and save cnn
net = Net()
train_model(net, trainloader, NUM_EPOCHS)
test_model_conf_mat(net, testloader)
save_model(net)
