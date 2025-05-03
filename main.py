import torch  # a directory with a lot of data utilities(in this project i used raw torch just for GPU usage)
import torch.nn as nn  # didn't use yet because i didnt start the neural network build
import torch.nn.functional as F  # still, honestly doesn't know what is this functional as f

import torch.optim as optim  # this would implement the adam algorithem i guess
import torchvision  # incredible library that handle data manipulations and storage such as transforms or ImageFolder
import matplotlib.pyplot as plt  # another tool that has a platform for image printing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # config the nn calculation to run on GPU(if can)
path_to_data = "C:/Users/user/CNN Dogs_Cats/DATA/CNN First project"  # my data path

image_size = 128  # instance created for resize the images in future(in transforms)

transform = torchvision.transforms.Compose([  # compose mean something like "these adjusts are going to be made:"
    torchvision.transforms.Resize(image_size),  # here we are config the resize to be image size 128
    torchvision.transforms.CenterCrop(128),  # the problem of Resize function is that it changes the horizontal
    # too in ration so we cropping it to get a square witch is easier for the calculations
    torchvision.transforms.ToTensor()  # it's important to transform the PIL image to a tensor for CNN calculations
])
#  now we yet activate the transform, we just set it to the right configurations, we will activate it in ImageFolder

train_dataset = torchvision.datasets.ImageFolder(root=path_to_data + '/train', transform=transform)  # Get images
test_dataset = torchvision.datasets.ImageFolder(root=path_to_data + '/test', transform=transform)  # Get images

#  print(train_dataset.classes)
#  print(train_dataset.class_to_idx)

# for img, lbl in train_dataset:
#     if lbl == 1:
#         dog_img = img
#         break
# plt.imshow(dog_img.permute(1, 2, 0))  # for using plt we nee

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)  # Final preparing
# before sending to the CNN
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)  # Final preparing


# before sending to the CNN


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1)
        self.conv2 = nn.Conv2d(16, 32, 3, 1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28800, 128)
        self.fc2 = nn.Linear(128, 2)

    def forward(self, image_input):
        conv1_result = self.conv1(image_input)
        relu1_result = self.relu(conv1_result)
        pull1_result = self.pool(relu1_result)
        conv2_result = self.conv2(pull1_result)
        relu2_result = self.relu(conv2_result)
        pull2_result = self.pool(relu2_result)
        x = self.flatten(pull2_result)
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x


model = CNN()
model.to(device)
#  hii it's the new branch


# Done:
# - Data loading
# - Transform
# - CNN definition (conv, relu, pool, fc)

# Next:
# - Loss function (maybe CrossEntropyLoss)
# - Optimizer (Adam)
# - Training loop (forward, backward, step)
# - Validation loop
# - Save model
