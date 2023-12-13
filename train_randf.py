from train_utils import preprocess_data_random_forest, test_rf, preprocess_data_random_forest_with_gaussian_blur
from datasets import load_dataset
from constants import HF_DS_PATH
from sklearn.ensemble import RandomForestClassifier
from create_graphs import create_confusion_matrix_rf

# Load dataset
dataset = load_dataset(HF_DS_PATH)

# Preprocess data for Random Forest
train, test = preprocess_data_random_forest(dataset)

# Preprocess noisy data for Random Forest
#train, test = preprocess_data_random_forest_with_gaussian_blur(dataset)

# Extract features and labels for training
xtrain, ytrain = train["pixel_values"], train["label"]
nsamples, nx, ny, nrgb = xtrain.shape
x_train2 = xtrain.reshape((nsamples, nx * ny * nrgb))

# Extract features and labels for testing
xtest, ytest = test["pixel_values"], test["label"]
nsamples, nx, ny, nrgb = xtest.shape
x_test2 = xtest.reshape((nsamples, nx * ny * nrgb))

# Define depths for Random Forest
depths = [1, 5, 25, 50]

for depth in depths:
    # Create and train Random Forest model
    model = RandomForestClassifier(n_estimators=100, max_depth=depth, verbose=1)
    print(f"Training Random Forest with depth: {depth}")
    model.fit(x_train2, ytrain)

    # Test Random Forest model
    mat = test_rf(model, x_test2, ytest)
    create_confusion_matrix_rf(mat)
