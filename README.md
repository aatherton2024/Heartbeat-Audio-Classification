# CSCI-374-Final-Project
Computer Vision ML Task on Hearbeat Noise Dataset

Project Flow:
1. Create a huggingface style data set by reading in csv files
    - This may also take the form of creating objects which will then hold image file paths (I like this approach)
2. From this dataset or object list, generate images
3. Observe that train test splits are already provided
4. Import model architecture from pytorch or huggingface, train model
5. Test model

Considerations:
 - Dataset is too small to sufficiently train model, will likely need to bag/boost or provide synthetic or additional data
 - Need to train on several model architectures and on other ML frameworks, such as random forests

 Folders:
 - Data: folder stores all imported data from kaggle dataset
     - set_a: folder stores all heartbeat audio data from stethoscope recordings
     - set_b: folder stores all heartbeat audio data from phone app recordings
 - Images: folder stores all spectogram image files from audio conversions
 - Model: folder stores all saved model checkpoints
    - subfolders for each model implementation

Files:
 - data/set_a.csv: csv file containing classification data for audio files in set_a
 - data/set_b.csv: csv file containing classification data for audio files in set_b
 - constants.py: file containing constants allowing for easy model adjustment
 - create_images.py: file to create spectogram image files for each audio file
 - create_dataset.py: file to create hf style dataset from csvs with paths to audio and image files for each entry
 - train_model.py: file to train model
 - test_model.py: file to test model's ability to classify never seen before heartbeat audio
 - run.sh: bash scripting file to run code on school's GPUs (faster training)

