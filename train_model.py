import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.optim import Adam
from datasets import load_dataset
from torch.autograd import Variable
from contstants import MODEL_DIRECTORY, BATCH_SIZE, NUM_EPOCHS
import os
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np

# Load image dataset
dataset = load_dataset("aatherton2024/heartbeat_images_final_project")
print(dataset)

# Define a convolution neural network
class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        
        self.conv1 = nn.Conv2d(in_channels=3, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(12)
        self.conv2 = nn.Conv2d(in_channels=12, out_channels=12, kernel_size=5, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(12)
        self.pool = nn.MaxPool2d(2,2)
        self.conv4 = nn.Conv2d(in_channels=12, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn4 = nn.BatchNorm2d(24)
        self.conv5 = nn.Conv2d(in_channels=24, out_channels=24, kernel_size=5, stride=1, padding=1)
        self.bn5 = nn.BatchNorm2d(24)
        self.fc1 = nn.Linear(24*10*10, 10)

    def forward(self, input):
        output = F.relu(self.bn1(self.conv1(input)))      
        output = F.relu(self.bn2(self.conv2(output)))     
        output = self.pool(output)                        
        output = F.relu(self.bn4(self.conv4(output)))     
        output = F.relu(self.bn5(self.conv5(output)))     
        output = output.view(-1, 24*10*10)
        output = self.fc1(output)

        return output

# Instantiate a neural network model 
model = Network()
 
# Define the loss function with Classification Cross-Entropy loss and an optimizer with Adam optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

# Function to save the model
def save_model():
    path = f"{MODEL_DIRECTORY}checkpoint.pt"
    if not os.path.exists(MODEL_DIRECTORY):
        os.makedirs(MODEL_DIRECTORY)
    torch.save(model, path)

train_loader = DataLoader(dataset["train"], batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(dataset["test"], batch_size=BATCH_SIZE, shuffle=True)

# Function to test the model with the test dataset and print the accuracy for the test images
def test_accuracy():
    
    model.eval()
    accuracy = 0.0
    total = 0.0
    
    with torch.no_grad():
        for data in test_loader:
            images, labels = data
            # run the model on the test set to predict labels
            outputs = model(images)
            # the label with the highest energy will be our prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            accuracy += (predicted == labels).sum().item()
    
    # compute the accuracy over all test images
    accuracy = (100 * accuracy / total)
    return(accuracy)


# Training function. We simply have to loop over our data iterator and feed the inputs to the network and optimize.
def train(num_epochs):
    
    best_accuracy = 0.0

    # Define your execution device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"The model will be running on {device}")
    # Convert model parameters and buffers to CPU or Cuda
    model.to(device)

    for epoch in range(num_epochs):  # loop over the dataset multiple times
        running_loss = 0.0
        running_acc = 0.0

        for i, (images, labels) in enumerate(train_loader, 0):
            
            # get the inputs
            images = Variable(images.to(device))
            labels = Variable(labels.to(device))

            # zero the parameter gradients
            optimizer.zero_grad()
            # predict classes using images from the training set
            outputs = model(images)
            # compute the loss based on model output and real labels
            loss = loss_fn(outputs, labels)
            # backpropagate the loss
            loss.backward()
            # adjust parameters based on the calculated gradients
            optimizer.step()

            # Let's print statistics for every 1,000 images
            running_loss += loss.item()     # extract the loss value
            if i % 1000 == 999:    
                # print every 1000 (twice per epoch) 
                print('[%d, %5d] loss: %.3f' %
                      (epoch + 1, i + 1, running_loss / 1000))
                # zero the loss
                running_loss = 0.0

        # Compute and print the average accuracy fo this epoch when tested over all 10000 test images
        accuracy = test_accuracy()
        print('For epoch', epoch+1,'the test accuracy over the whole test set is %d %%' % (accuracy))
        
        # we want to save the model if the accuracy is the best
        if accuracy > best_accuracy:
            save_model()
            best_accuracy = accuracy

# Function to show the images
def image_show(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

#region Optional test visualization at batch level
# Function to test the model with a batch of images and show the labels predictions
# def test_batch():
#     # get batch of images from the test DataLoader  
#     images, labels = next(iter(test_loader))

#     # show all images as one image grid
#     image_show(torchvision.utils.make_grid(images))
   
#     # Show the real labels on the screen 
#     print('Real labels: ', ' '.join('%5s' % classes[labels[j]] 
#                                for j in range(BATCH_SIZE)))
  
#     # Let's see what if the model identifiers the  labels of those example
#     outputs = model(images)
    
#     # We got the probability for every 10 labels. The highest (max) probability should be correct label
#     _, predicted = torch.max(outputs, 1)
    
#     # Let's show the predicted labels on the screen to compare with the real ones
#     print('Predicted: ', ' '.join('%5s' % classes[predicted[j]] 
#                               for j in range(BATCH_SIZE)))
#endregion

if __name__ == "__main__":
    
    # Let's build our model
    train(5)
    print('Finished Training')

    # Test which classes performed well
    test_accuracy()
    
    # Let's load the model we just created and test the accuracy per label
    model_save_location = f"{MODEL_DIRECTORY}checkpoint.pt"
    model = torch.load(model_save_location)
    model.eval()

    # # Test with batch of images
    # test_batch()