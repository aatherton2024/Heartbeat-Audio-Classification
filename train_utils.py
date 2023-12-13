import matplotlib.pyplot as plt
from torchvision.transforms import Compose, ToTensor
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score
import pandas as pd
import seaborn as sn
from PIL import Image
from constants import NUM_EPOCHS, BATCH_SIZE, NUM_FOLDS
from sklearn.model_selection import KFold
from pytorch_cnn import Net
import os
import skimage as ski
from tqdm import tqdm

def train_model(net, dataloader, epochs=NUM_EPOCHS, current_fold=-1, validationloader=None, results=dict()):
    """
    Train a CNN model.

    Parameters:
    - net (nn.Module): The CNN model to be trained.
    - dataloader (DataLoader): DataLoader for training data.
    - epochs (int): Number of training epochs.
    - current_fold (int): Current fold number for cross-validation.
    - validationloader (DataLoader): DataLoader for validation data.
    - results (dict): Dictionary to store training results.

    Returns:
    - results (dict): Updated dictionary with training results.
    """
    if current_fold >= 0: print(f"Training model of fold {current_fold+1}")
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
            image_save_directory = f"conf_matrices/fold_{current_fold+1}/"
            image_save_path = f"conf_matrices/fold_{current_fold+1}/epoch_{epoch+1}.png"

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

def train_model_with_cv(dataset, num_epochs=NUM_EPOCHS, batch_size=BATCH_SIZE, num_folds=NUM_FOLDS):
    """
    Train CNN models using cross-validation.

    Parameters:
    - dataset (Dataset): The dataset containing training and validation data.
    - num_epochs (int): Number of training epochs.
    - batch_size (int): Batch size for training.
    - num_folds (int): Number of folds for cross-validation.

    Returns:
    - results (dict): Updated dictionary with training results for every checkpoint and fold.
    """
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
        
    print('--------------------------------')

    # K-fold Cross Validation model evaluation
    for fold, (train_ids, validation_ids) in enumerate(kfold.split(dataset["train"])):
        print(f'FOLD {fold}')
        print('--------------------------------')
        
        trainloader, validationloader = cross_validation_dataloaders(dataset["train"], train_ids, validation_ids, batch_size)
        
        # Init the neural network
        network = Net()
        network.apply(reset_weights)
        
        train_model(network, trainloader, num_epochs, fold, validationloader, results)
                
        print('Training process has finished. Saving trained model.')
        print('Starting testing')

    return results

def test_rf(model, xtest, ytest):
    """
    Test a random forest model.

    Parameters:
    - model: The trained random forest model.
    - xtest: Test data.
    - ytest: True labels for test data.

    Returns:
    np confusion matrix
    """
    predictions = model.predict(xtest)
    print(accuracy_score(predictions, ytest))
    mat = confusion_matrix(predictions, ytest)
    mat.tolist()
    return mat

def test_model_conf_mat(net, dataloader, save_location="output.png"):
    """
    Test a CNN model and generate a confusion matrix.

    Parameters:
    - net (nn.Module): The trained CNN model.
    - dataloader (DataLoader): DataLoader for test data.
    - save_location (str): Path to save the generated confusion matrix image.

    Returns:
    - score (float): Accuracy score of the CNN model on the test data.
    """
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

def img_show(img):
    """
    Display a tensor as a PNG image.

    Parameters:
    - img: The input tensor.

    Returns:
    None
    """
    tensor_image = img
    tensor_image = tensor_image.view(tensor_image.shape[2], tensor_image.shape[0], tensor_image.shape[1])
    tensor_image = tensor_image.view(tensor_image.shape[2], tensor_image.shape[0], tensor_image.shape[1])
    plt.imshow(tensor_image)
    plt.show()

def preprocess_data(dataset, batch_size=4):
    """
    Preprocess data by normalizing, resizing, and converting to tensors.

    Parameters:
    - dataset: The input dataset.
    - batch_size (int): Batch size for data loaders.

    Returns:
    - dataset (Dataset): Preprocessed dataset.
    - trainloader (DataLoader): DataLoader for training data.
    - testloader (DataLoader): DataLoader for test data.
    """
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

def preprocess_data_with_gaussian_noise(dataset, batch_size=4):
    """
    Preprocess data by normalizing, resizing, adding Gaussian blur, and converting to tensors.

    Parameters:
    - dataset: The input dataset.
    - batch_size (int): Batch size for data loaders.

    Returns:
    - dataset (Dataset): Preprocessed dataset.
    - trainloader (DataLoader): DataLoader for training data.
    - testloader (DataLoader): DataLoader for test data.
    """
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

def cross_validation_dataloaders(dataset, train_ids, validation_ids, batch_size):
    """
    Create data loaders for cross-validation.

    Parameters:
    - dataset: The input dataset.
    - train_ids: Indices for training data.
    - validation_ids: Indices for validation data.
    - batch_size (int): Batch size for data loaders.

    Returns:
    - trainloader (DataLoader): DataLoader for training data.
    - validationloader (DataLoader): DataLoader for validation data.
    """
    train_subsampler = torch.utils.data.SubsetRandomSampler(train_ids)
    validation_subsampler = torch.utils.data.SubsetRandomSampler(validation_ids)
    
    # Define data loaders for training and testing data in this fold
    trainloader = DataLoader(dataset, batch_size=batch_size, sampler=train_subsampler)
    validationloader = DataLoader(dataset, batch_size=batch_size, sampler=validation_subsampler)
    return trainloader, validationloader

def preprocess_data_random_forest(dataset):
    """
    Preprocess data for random forests in np format.

    Parameters:
    - dataset: The input dataset.

    Returns:
    - train_data (numpy.ndarray): Preprocessed training data.
    - test_data (numpy.ndarray): Preprocessed test data.
    """
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

def preprocess_data_random_forest_with_gaussian_blur(dataset):
    """
    Preprocess data for random forests in np format with Gaussian blur.

    Parameters:
    - dataset: The input dataset.

    Returns:
    - train_data (numpy.ndarray): Preprocessed training data.
    - test_data (numpy.ndarray): Preprocessed test data.
    """
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

def save_model(model, save_path="model.pt"):
    """
    Save the state dict of a model.

    Parameters:
    - model: The model to be saved.
    - save_path (str): Path to save the model.

    Returns:
    None
    """
    torch.save(model.state_dict(), save_path)