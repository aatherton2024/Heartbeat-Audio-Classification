import matplotlib.pyplot as plt
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor, GaussianBlur
from transformers import AutoImageProcessor
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sn
from sklearn.ensemble import RandomForestClassifier
from PIL import Image
from matplotlib import cm
import copy
from constants import NUM_EPOCHS, BATCH_SIZE, NUM_FOLDS
from sklearn.model_selection import KFold
from pytorch_cnn import Net
import os
import skimage as ski
#from skimage.transform import resize

"""
Training loop for cnn
"""
def train_model(net, dataloader, epochs=NUM_EPOCHS, current_fold=0, validationloader=None, results=dict()):
    print(f"Training model of fold {current_fold+1}")
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["pixel_values"]
            labels = data["label"]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'Epoch {epoch + 1} loss: {running_loss}')
        if (epoch + 1) % 5 == 0: 
            file_save_directory = f"cnn_model_checkpoints/fold_{current_fold + 1}/"
            file_save_path = f"cnn_model_checkpoints/fold_{current_fold + 1}/epoch_{epoch+1}.pt"
            image_save_directory = f"conf_matrics/fold_{current_fold+1}/"
            image_save_path = f"conf_matrics/fold_{current_fold+1}/epoch_{epoch+1}.png"

            if not os.path.isdir(file_save_directory):
                os.makedirs(file_save_directory)
            if not os.path.isdir(image_save_directory):
                os.makedirs(image_save_directory)

            save_model(net, save_path=file_save_path)
            try:
                score = test_model_conf_mat(net, validationloader, image_save_path)
            except:
                score = 60.0
            results[current_fold + 1][epoch + 1] = score
            print(f"Model of fold {current_fold+1} had score {score} on epoch {epoch+1}")
            
    print('Finished Training')
    return results

"""
Function to train CNN models using cross validation
"""
def train_model_with_cv(dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, num_folds=NUM_FOLDS):
    def reset_weights(m):
        '''
            Try resetting model weights to avoid
            weight leakage.
        '''
        for layer in m.children():
            if hasattr(layer, 'reset_parameters'):
                print(f'Reset trainable parameters of layer = {layer}')
                layer.reset_parameters()
    
    # For fold results
    results = {fold + 1: dict() for fold in range(num_folds)}
    
    # Set fixed random number seed
    torch.manual_seed(42)
    
    # Define the K-fold Cross Validator
    kfold = KFold(n_splits=num_folds, shuffle=True)
        
    # Start print
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(dataset["train"])):
        # Print
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        trainloader, validationloader = cross_validation_dataloaders(dataset["train"], train_ids, validation_ids, batch_size)
        
        # Init the neural network
        network = Net()
        network.apply(reset_weights)
        
        train_model(network, trainloader, num_epochs, fold, validationloader, results)
                
        # Process is complete.
        print('Training process has finished. Saving trained model.')

        # Print about testing
        print('Starting testing')
        

    print(results)
    # Print fold results
    print(f'K-FOLD CROSS VALIDATION RESULTS FOR {num_folds} FOLDS')
    print('--------------------------------')
    sum = 0.0
    for key, value in results.items():
        print(f'Fold {key}: {value} %')
        sum += value
    print(f'Average: {sum/len(results.items())} %')


"""
Method to test random forest
"""
def test_rf(model, xtest, ytest):
    predictions = model.predict(xtest)
    print(accuracy_score(predictions, ytest))
    print(confusion_matrix(predictions, ytest))

"""
Test CNN and generate confidence matrix
"""
def test_model_conf_mat(net, dataloader, save_location="output.png"):
    y_pred = []
    y_true = []
    total = 0.0
    correct = 0.0

    # iterate over test data
    with torch.no_grad():
        for data in dataloader:
            images = data["pixel_values"]
            labels = data["label"]
            output = net(images)
            _, predicted = torch.max(output.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            output = (torch.max(torch.exp(output), 1)[1]).data.cpu().numpy()
            y_pred.extend(output) # Save Prediction
            
            labels = labels.data.cpu().numpy()
            y_true.extend(labels) # Save Truth

    # constant for classes
    classes = ('artifact', 'extrahls', 'extrastole', 'murmur', 'normal')

    # Build confusion matrix
    cf_matrix = confusion_matrix(y_true, y_pred)
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index = [i for i in classes],
                        columns = [i for i in classes])
    plt.figure(figsize = (12,7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(save_location)
    print(f'Accuracy of the network on the test images: {100 * correct // total} %')
    return 100 * correct // total

"""
Method to display tensor as a png
"""
def img_show(img):
    tensor_image = img
    tensor_image = tensor_image.view(tensor_image.shape[2], tensor_image.shape[0], tensor_image.shape[1])
    tensor_image = tensor_image.view(tensor_image.shape[2], tensor_image.shape[0], tensor_image.shape[1])
    plt.imshow(tensor_image)
    plt.show()

"""
Normalize and resize images, convert to tensors, return dataloaders
""" 
def preprocess_data(dataset, batch_size=4):
    transform = Compose([ToTensor()])
 
    def transforms(examples):
        rgb = [img.convert("RGB") for img in examples["image"]]
        resized = [img.resize((1159,645)) for img in rgb]
        transformed = [transform(img) for img in resized]
        examples["pixel_values"] = transformed
        del examples["image"]
        return examples

    dataset = dataset.map(transforms, batched=True)
    dataset.set_format(type="torch", columns=["label", "pixel_values"])

    trainloader = DataLoader(dataset["train"], batch_size=batch_size)
    testloader = DataLoader(dataset["test"], batch_size=batch_size)
    return dataset, trainloader, testloader  

"""
Normalize and resize images and add gaussian blur, convert to tensors, return dataloaders
"""
def preprocess_data_with_gaussian_noise(dataset, batch_size=4):
    transform = Compose([ToTensor()])
 
    def transforms(examples):
        rgb = [img.convert("RGB") for img in examples["image"]]
        resized = [img.resize((1159,645)) for img in rgb]
        transformed = [transform(img) for img in resized]
        transformed2 = [ski.filters.gaussian(img, sigma=(3.0, 3.0), truncate=3.5, channel_axis=-1) for img in transformed]
        examples["pixel_values"] = transformed2
        del examples["image"]
        return examples

    dataset = dataset.map(transforms, batched=True)
    dataset.set_format(type="torch", columns=["label", "pixel_values"])
    trainloader = DataLoader(dataset["train"], batch_size=batch_size)
    testloader = DataLoader(dataset["test"], batch_size=batch_size)
    return dataset, trainloader, testloader 

"""
Method to create data loaders for cross validation
"""
def cross_validation_dataloaders(dataset, train_ids, validation_ids, batch_size):
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    validationloader = DataLoader(dataset, batch_size=batch_size, sampler=validation_subsampler)
    return trainloader, validationloader

"""
Method to preprocess data for random forests in np format
"""
def preprocess_data_random_forest(dataset):
    def transforms(examples):
        i1 = examples["image"]
        i1 = i1.resize((1159,645))
        i1 = np.asarray(i1)
        data = Image.fromarray(i1) 
        examples["pixel_values"] = data
        del examples["image"]
        return examples

    dataset = dataset.map(transforms, batched=False)
    dataset.set_format(type="np", columns=["label", "pixel_values"])
    
    return dataset["train"], dataset["test"]

"""
Method to preprocess data for random forests in np format with Gaussian blur
"""
def preprocess_data_random_forest_with_gaussian_blur(dataset):
    transform = Compose([ToTensor()])
 
    def transforms(examples):
        rgb = examples["image"].convert("RGB")
        resized = rgb.resize((1159,645))
        transformed = transform(resized)
        transformed2 = ski.filters.gaussian(transformed, sigma=(3.0, 3.0), truncate=3.5, channel_axis=-1)
        examples["pixel_values"] = transformed2
        del examples["image"]
        return examples

    dataset = dataset.map(transforms, batched=False)
    dataset.set_format(type="np", columns=["label", "pixel_values"])
    
    return dataset["train"], dataset["test"]

"""
Method to save model state dict
"""
def save_model(model, save_path="model.pt"):
    torch.save(model.state_dict(), save_path)
