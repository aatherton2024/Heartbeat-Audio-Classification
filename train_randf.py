from train_utils import preprocess_data_random_forest, test_rf
from datasets import load_dataset
from constants import HF_DS_PATH
from sklearn.ensemble import RandomForestClassifier

dataset = load_dataset(HF_DS_PATH)
train, test = preprocess_data_random_forest(dataset)
# xtrain, ytrain = train["pixel_values"], train["label"]
# xtest, ytest =  test["pixel_values"], test["label"]
# model = RandomForestClassifier()
# model.fit(xtrain, ytrain)
# test_rf(model, xtest, ytest)

