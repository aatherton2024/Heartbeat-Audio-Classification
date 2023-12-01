from train_utils import preprocess_data_random_forest, test_rf
from datasets import load_dataset
from constants import HF_DS_PATH
from sklearn.ensemble import RandomForestClassifier

dataset = load_dataset(HF_DS_PATH)
train, test = preprocess_data_random_forest(dataset)
xtrain, ytrain = train["pixel_values"], train["label"]
nsamples,nx,ny,nrgb = xtrain.shape
x_train2 = xtrain.reshape((nsamples,nx*ny*nrgb))

xtest, ytest =  test["pixel_values"], test["label"]
nsamples,nx,ny,nrgb = xtest.shape
x_test2 = xtest.reshape((nsamples,nx*ny*nrgb))

model = RandomForestClassifier(n_estimators=10, max_depth=10, verbose=1)
print("TRAINING MODEL")
model.fit(x_train2, ytrain)
test_rf(model, x_test2, ytest)

