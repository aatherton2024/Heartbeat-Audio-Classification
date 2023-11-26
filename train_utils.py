import matplotlib.pyplot as plt
from torchvision.transforms import RandomResizedCrop, Compose, Normalize, ToTensor
from transformers import AutoImageProcessor
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim

"""
Training loop for cnn
"""
def train_model(net, dataloader, epochs=20):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    for epoch in range(epochs):  # loop over the dataset multiple times

        running_loss = 0.0
        for i, data in enumerate(dataloader, 0):
            # get the inputs; data is a list of [inputs, labels]
            inputs = data["pixel_values"]
            labels = data["label"]

            # zero the parameter gradients
            optimizer.zero_grad()

            # forward + backward + optimize
            outputs = net(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            # print statistics
            running_loss += loss.item()
        print(f'Epoch {epoch + 1} loss: {running_loss}')

    print('Finished Training')

"""
Test loop for model... currently returns accuracy
#TODO return a better evaluation metric
"""
def test_model(net, dataloader):
    correct = 0
    total = 0
    # since we're not training, we don't need to calculate the gradients for our outputs
    with torch.no_grad():
        for data in dataloader:
            images = data["pixel_values"]
            labels = data["label"]
            # calculate outputs by running images through the network
            outputs = net(images)
            # the class with the highest energy is what we choose as prediction
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    print(f'Accuracy of the network on the test images: {100 * correct // total} %')

"""
Method to display tensor as a png
"""
def img_show(img):
    tensor_image = img
    print(type(tensor_image), tensor_image.shape)
    tensor_image = tensor_image.view(tensor_image.shape[2], tensor_image.shape[0], tensor_image.shape[1])
    print(type(tensor_image), tensor_image.shape)
    tensor_image = tensor_image.view(tensor_image.shape[2], tensor_image.shape[0], tensor_image.shape[1])
    print(type(tensor_image), tensor_image.shape)
    plt.imshow(tensor_image)
    plt.show()

"""
Normalize and resize images, convert to tensors, return dataloaders
"""
def preprocess_data(dataset, batch_size=4):
    checkpoint = "google/vit-base-patch16-224-in21k"
    image_processor = AutoImageProcessor.from_pretrained(checkpoint)

    normalize = Normalize(mean=image_processor.image_mean, std=image_processor.image_std)
    size = (image_processor.size["height"], image_processor.size["width"])
    _transforms = Compose([RandomResizedCrop(size), ToTensor(), normalize])

    def transforms(examples):
        examples["pixel_values"] = [_transforms(img.convert("RGB")) for img in examples["image"]]
        del examples["image"]
        return examples

    dataset = dataset.map(transforms, batched=True)
    dataset.set_format(type="torch", columns=["label", "pixel_values"])

    trainloader = DataLoader(dataset["train"], batch_size=batch_size)
    testloader = DataLoader(dataset["test"], batch_size=batch_size)
    return trainloader, testloader                                                                                           

"""
Method to save model state dict to disk
"""
def save_model(model, save_path="model.pt"):
    torch.save(model.state_dict(), save_path)