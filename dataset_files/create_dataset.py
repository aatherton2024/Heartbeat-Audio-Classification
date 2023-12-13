import os
from tqdm import tqdm
from constants import DATA_DIRECTORY, IMAGES_DIRECTORY, HF_DS_PATH
from create_images import make_spectogram
from datasets import load_dataset
from constants import DATA_DIRECTORY

def generate_all_images(directory):
    """
    Generate spectrograms for all audio files in the given directory.

    Parameters:
    - directory (str): The path to the directory containing audio files.

    Returns:
    None
    """
    for filename in tqdm(os.listdir(directory)):
        if filename.startswith("murmur"):
            label = "murmur"
        elif filename.startswith("extrahls"):
            label = "extrahls"
        elif filename.startswith("extrastole"):
            label = "extrastole"
        elif filename.startswith("artifact"):
            label = "artifact"
        else:
            label = "normal"

        file_path = f"{DATA_DIRECTORY}{filename}"
        image_folder = f"{IMAGES_DIRECTORY}train/{label}/"
        image_path = f"{IMAGES_DIRECTORY}train/{label}/{filename[:-4]}.png"

        if not os.path.isfile(image_path):
            print("creating new image file")
            if not os.path.isdir(image_folder):
                os.makedirs(image_folder)
            make_spectogram(file_path, image_path)

def create_image_dataset():
    """
    Create a HuggingFace (HF) dataset from the image folder.

    Parameters:
    None

    Returns:
    None
    """
    dataset = load_dataset("imagefolder", data_dir="images/")
    dataset = dataset["train"].train_test_split(test_size=0.2, stratify_by_column="label")
    dataset.push_to_hub(HF_DS_PATH)
