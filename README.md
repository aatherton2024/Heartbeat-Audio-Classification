# CSCI-374-Final-Project
Computer Vision ML Task on Hearbeat Noise Dataset

Project Flow:
1. Generate images ✅
2. Create hugginface dataset with train test splits ✅
3. Make model class from generic PyTorch framework 🟡
4. Test model 🔴
5. Use other model frameworks and play around with hyperparameters 🔴
6. Write paper 🔴
7. Win CS project of the year 🔴

Considerations:
 - Dataset is too small to sufficiently train model, will likely need to bag/boost or provide synthetic or additional data
 - Need to train on several model architectures and on other ML frameworks, such as random forests
 - Dataset will (potentially) need to be regenerated with shorter recordings padded to max recording length

 Folders:
 - Data: folder stores all imported data from kaggle dataset (empty at the moment, will eventually want to redownload data)
 - Model: folder stores all saved model checkpoints
    - subfolders for each model implementation

Files:
 - constants.py: file containing constants allowing for easy model adjustment
 - create_images.py: file to create spectogram image files for each audio file
 - create_dataset.py: file to create hf style dataset from csvs with paths to audio and image files for each entry
 - train_model.py: file to train model and store model class
 - test_model.py: file to test model's ability to classify never seen before heartbeat audio
 - run.sh: bash scripting file to run code on school's GPUs (faster training)

Usage:
 - To load in hf dataset: dataset = load_dataset("aatherton2024/heartbeat_images_final_project")
    - Observe that the data folder will be empty as there is no need for audio recordings at the moment
 - Kaggle data: https://www.kaggle.com/datasets/mersico/dangerous-heartbeat-dataset-dhd

