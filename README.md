# Hearbeat Audio Computer Vision Project

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Folder Structure](#Structure)

## Introduction
This is a repository containing code to train and test a Convolutional Neural Network and a Random Forest Classifier to perform classification on heartbeat audio. The goal of this project was to accurately classify hearbeat audio within five classes: artifact, extrahls, extrastole, murmur, and normal. The main contributions of this project are three fold. We first created a framework to convert a Kaggle dataset of hearbeat .wav audio files into a publicy-accessible Huggingface dataset of spectrograms and their corresponding class labels. We then provided the framework to train and test a CNN and Random Forest on the data using cross-validation. We finally provide the framework to evaluate model performance on test data and test data with added noise using confusion matrices.

## Installation
To run everything from scratch, you must first download the data from Kaggle and store it in the data folder:
https://www.kaggle.com/datasets/mersico/dangerous-heartbeat-dataset-dhd

You then need to change DATASET_EXISTS to false in constants.py
You will also need to create your own Huggingface dataset and update HF_DS_PATH in constants.py

Otherwise, you can use our Huggingface dataset(highly reccommended)
```bash
git clone https://github.com/aatherton2024/Heartbeat-Audio-Classification.git

cd Heartbeat-Audio-Classification

pip install -r requirements.txt
```

## Usage
To train a CNN:
```bash
cd training_files
python train_model.py
```

To train CNNs using cross validation:
```bash
cd training_files
python train_model_cv.py
```

To train Random Forests:
```bash
cd training_files
python train_randf.py
```

## Structure
```
Heartbeat-Audio-Classification/
│
├── data/
│   └── # Hearbeat Audio .wav files
│
├── models/
│   └── # Trained models
│
├── images/
│   └── # Spectrograms made from audio .wav files
|
├── README.md
├── constants.py
└── requirements.txt
└── create_dataset.py
└── create_images.py
└── create_graphs.py
└── pytorch_cnn.py
└── train_model.py
└── train_model_cv.py
└── train_randf.py
└── train_utils.py
```
