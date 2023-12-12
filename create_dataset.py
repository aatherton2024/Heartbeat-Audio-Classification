import os
from tqdm import tqdm
from constants import DATA_DIRECTORY, IMAGES_DIRECTORY, HF_DS_PATH
from create_images import make_spectogram
from datasets import load_dataset
import librosa
import librosa.display
import matplotlib.pyplot as plt
import numpy as np
from constants import DATA_DIRECTORY

"""
Method to create spectrograms for all audio files
"""
def generate_all_images(directory):

    for filename in tqdm(os.listdir(directory)):
        if filename.startswith("murmur"): label = "murmur"
        elif filename.startswith("extrahls"): label = "extrahls"
        elif filename.startswith("extrastole"): label = "extrastole"
        elif filename.startswith("artifact"): label = "artifact"
        else: label = "normal"

        file_path = f"{DATA_DIRECTORY}{filename}"
        image_folder = f"{IMAGES_DIRECTORY}train/{label}/"
        image_path = f"{IMAGES_DIRECTORY}train/{label}/{filename[:-4]}.png"

        if not os.path.isfile(image_path):
            print("creating new image file")
            if not os.path.isdir(image_folder):
                os.makedirs(image_folder)
            make_spectogram(file_path, image_path)

"""
Method to make a HF dataset from image folder
"""
def create_image_dataset():

    dataset = load_dataset("imagefolder", data_dir="images/")
    dataset = dataset["train"].train_test_split(test_size=0.2, stratify_by_column="label")
    dataset.push_to_hub(HF_DS_PATH)
