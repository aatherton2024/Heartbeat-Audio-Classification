from train_utils import preprocess_data_random_forest, test_rf, preprocess_data_random_forest_with_gaussian_blur
from datasets import load_dataset
from constants import HF_DS_PATH
from sklearn.ensemble import RandomForestClassifier

dataset = load_dataset(HF_DS_PATH)

#train, _ = preprocess_data_random_forest(dataset)
#_, test = preprocess_data_random_forest_with_gaussian_blur(dataset)

train, test = preprocess_data_random_forest(dataset)

xtrain, ytrain = train["pixel_values"], train["label"]
nsamples,nx,ny,nrgb = xtrain.shape
x_train2 = xtrain.reshape((nsamples,nx*ny*nrgb))

xtest, ytest =  test["pixel_values"], test["label"]
nsamples,nx,ny,nrgb = xtest.shape
x_test2 = xtest.reshape((nsamples,nx*ny*nrgb))

depths = [1,5,25,50]
for depth in depths:
    model = RandomForestClassifier(n_estimators=100, max_depth=depth, verbose=1)
    print("TRAINING MODEL")
    model.fit(x_train2, ytrain)
    test_rf(model, x_test2, ytest)

