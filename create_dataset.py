from datasets import Dataset
import pandas as pd
import os
from tqdm import tqdm
from contstants import DATA_DIRECTORY, IMAGES_DIRECTORY
from create_images import plotstft
from datasets import load_dataset

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
            plotstft(file_path, plotpath=image_path) #generate image

def create_image_dataset():

    dataset = load_dataset("imagefolder", data_dir="images/")
    dataset = dataset["train"].train_test_split(test_size=0.2, stratify_by_column="label")
    dataset.push_to_hub("aatherton2024/heartbeat_images_final_project")

ds = create_image_dataset()