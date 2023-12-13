from constants import HF_DS_PATH, BATCH_SIZE
from datasets import load_dataset
import matplotlib as mpl
import matplotlib.pyplot as plt
import os
import torch
from training_files.train_utils import test_model_conf_mat, preprocess_data, preprocess_data_with_gaussian_noise
from pytorch_cnn import Net
import pandas as pd
import seaborn as sn
import numpy as np

# Load dataset from Hugging Face
dataset = load_dataset(HF_DS_PATH)
dataset, train_loader, test_loader = preprocess_data(dataset, BATCH_SIZE)

#to load noisy data
#dataset, train_loader, test_loader = preprocess_data_with_gaussian_noise(dataset, BATCH_SIZE)

def class_dist():
    """
    Create a barplot visualizing class distribution.

    Parameters:
    None

    Returns:
    None
    """
    mpl.style.use('seaborn-v0_8')
    d = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    for split in ["train", "test"]:
        for entry in dataset[split]:
            label = int(entry["label"])
            d[label] += 1

    updated_counts = {"normal": d[4], "murmur": d[3], "extrastole": d[0], "artifact": d[2], "extrahls": d[1]}

    plt.bar(range(len(updated_counts)), list(updated_counts.values()), align='center')
    plt.xticks(range(len(updated_counts)), list(updated_counts.keys()))
    plt.xlabel("Heartbeat classification")
    plt.ylabel("Count")
    plt.savefig("classdist_barplot.png")

def cnn_plot(fold):
    """
    Create a confusion matrix for CNN on test data.

    Parameters:
    - fold (int): The fold number.

    Returns:
    None
    """
    for save_path in os.listdir(f"cnn_model_checkpoints/fold_{fold}/"):
        full_path = f"cnn_model_checkpoints/fold_{fold}/{save_path}"
        model = Net()
        model.load_state_dict(torch.load(full_path))
        model.eval()
        save_location = f"test_conf_matrices/fold_{fold}/{save_path[:-3]}.png"
        if not os.path.isdir(f"test_conf_matrices/fold_{fold}/"):
                os.makedirs(f"test_conf_matrices/fold_{fold}/")
        score = test_model_conf_mat(model, test_loader, save_location)
        print(f"{save_location} score: {score}")

def plot_folds(cnn_dict):
    """
    Plot cross-validation results.

    Parameters:
    - cnn_dict (dict): Dictionary containing fold-wise accuracy values.

    Returns:
    None
    """
    mpl.style.use('seaborn-v0_8')
    x = [key for key in cnn_dict[1].keys()]
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf']
    for fold, val_dict in cnn_dict.items():
        values = [value for value in val_dict.values()]
        plt.plot(x, values, color=colors[fold-1], label=f"Fold_{fold}", marker=".")
    plt.legend(loc="upper left", ncol=2)
    plt.ylim(40.0, 90.0)
    plt.xlabel("Number of Epochs")
    plt.ylabel("Cross Validation Fold Accuracy")
    plt.savefig("folds.png")

def create_confusion_matrix_rf(mat, save_location="output.png"):
    """
    Create a confusion matrix from the output of RandomForestClassifier in scikit-learn.

    Parameters:
    - mat (numpy.ndarray): The confusion matrix.

    Returns:
    None
    """
    cf_matrix = [[row[i] for row in mat] for i in range(len(mat[0]))]

    classes = ('artifact', 'extrahls', 'extrastole', 'murmur', 'normal')
    df_cm = pd.DataFrame(cf_matrix / np.sum(cf_matrix, axis=1)[:, None], index=[i for i in classes],
                        columns=[i for i in classes])
    plt.figure(figsize=(12, 7))
    sn.heatmap(df_cm, annot=True)
    plt.savefig(save_location)

def evaluate_model_with_noise(save_location):
    """
    Load in CNN from a checkpoint and evaluate it using noisy data.

    Parameters:
    - save_location (str): The path to the saved model checkpoint.

    Returns:
    None
    """
    model = Net()
    model.load_state_dict(torch.load(save_location))
    model.eval()
    score = test_model_conf_mat(model, test_loader, "cnn_noise.png")
    print(f"{save_location} score: {score}")
