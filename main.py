import torch  # a directory with a lot of data utilities(in this project i used raw torch just for GPU usage)
import torch.nn as nn  # didn't use yet because i didnt start the neural network build
import torch.optim as optim  # this would implement the adam algorithem i guess
import torchvision  # incredible library that handle data manipulations and storage such as transforms or ImageFolder
import matplotlib.pyplot as plt  # another tool that has a platform for image printing

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # config the nn calculation to run on GPU(if can)
path_to_data = "C:/Users/user/CNN Dogs_Cats/REDUCED DATA/REDUCED DATA SETS"  # my data path

image_size = 128  # instance created for resize the images in future(in transforms)

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.CenterCrop(image_size),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomRotation(10),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
    torchvision.transforms.ToTensor()
])

test_transform = torchvision.transforms.Compose([  # compose mean something like "these adjusts are going to be made:"
    torchvision.transforms.Resize(image_size),  # here we are config the resize to be image size 128
    torchvision.transforms.CenterCrop(image_size),  # the problem of Resize function is that it changes the horizontal
    # too in ration so we cropping it to get a square witch is easier for the calculations
    torchvision.transforms.ToTensor()  # it's important to transform the PIL image to a tensor for CNN calculations
])
#  now we yet activate the transform, we just set it to the right configurations, we will activate it in ImageFolder

train_dataset = torchvision.datasets.ImageFolder(root=path_to_data + '/train', transform=train_transforms)  # Get images
test_dataset = torchvision.datasets.ImageFolder(root=path_to_data + '/test', transform=test_transform)  # Get images

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)  # Final preparing
# before sending to the CNN
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)  # Final preparing


# before sending to the CNN


class CNN(nn.Module):  # Creating a CNN class to set the CNN flow and arrtributes, also it inherits the nn.Module
    # that holds important and usefull CNN utilities such as
    def __init__(self):  # The constructor to create a CNN object
        super().__init__()  # send it to the father initialization
        self.conv1 = nn.Conv2d(3, 16, 3, 1)  # First conv_layer
        self.conv2 = nn.Conv2d(16, 32, 3, 1)  # Second conv_layer
        self.conv3 = nn.Conv2d(32, 64, 3, 1)
        self.relu = nn.ReLU()  # Activation to highlight important patterns
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # pool highest pixel value of 4x4 set to reduce size
        # and improve flow of the calculations.
        self.flatten = nn.Flatten()  # Flattening the tensor for the fc layer
        self.fc1 = nn.Linear(12544, 128)  # First fc_layer
        self.fc2 = nn.Linear(128, 2)  # Second fc_layer

    def forward(self, image_input):  # in the above, we set the utilities that we *will* use, here we set the
        # actual CNN flow and process
        conv1_result = self.conv1(image_input)  # First, we conv the batch
        relu1_result = self.relu(conv1_result)  # then, we highlight important patterns(zeroing negative values)
        pool1_result = self.pool(relu1_result)  # pool the highest pixel values
        conv2_result = self.conv2(pool1_result)  # We conv the batch again
        relu2_result = self.relu(conv2_result)  # we highlight important patterns(zeroing negative values) again
        pool2_result = self.pool(relu2_result)  # pool the highest pixel values again
        conv3_result = self.conv3(pool2_result)
        relu3_result = self.relu(conv3_result)
        pool3_reeult = self.pool(relu3_result)
        x = self.flatten(pool3_reeult)  # Finished the conv, now we flatten the data as preparation for the fc layer
        x = self.fc1(x)  # apply fc layer
        x = self.relu(x)  # apply relu again to highlight important patterns even though we "finished the calculations"
        x = self.fc2(x)  # apply final fc layer
        return x


if __name__ == '__main__':
    model = CNN()  # Creating a CNN object named model
    model.to(device)  # configing the model to run on the gpu (if ther eis one) for faster calculations
    loss_fn = nn.CrossEntropyLoss()  # defining loss_fn to be a CrossEntropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # defining optimizer to change the weights later
    for epoch in range(15):  # running in a loop 5 times (5 epoch that runs upon all the data)
        print(f"Starting Epoch {epoch + 1}/15")  # indication of where we are
        i = 0  # meant to count the number of batchs to calculate avarage loss(why and how?)
        total_epoch_loss = 0  # total of epoch loss to calculate avaraage loss
        model.train()  # why do we activate .train and not .forward??

        for batch_idx, (images, labels) in enumerate(
                train_loader):  # run upon the whole train loader by seperated batches
            images = images.to(device)  # what does that mean
            labels = labels.to(device)  # what does that mean
            optimizer.zero_grad()  # preparing the surface to change the gradients(weights)
            outputs = model(images)  # again, why we activate model fnc and not the forward that we wrote
            loss = loss_fn(outputs, labels)  # loss calculation - need in depth explanation (my idea is that it taked
            # the outputs that holds the werights and the labels maybe of the fully connected, and then compare
            # them to the real lables so it know if the prediction is wrong and by how )
            total_epoch_loss += loss.item()  # takes the loss of any individual image(or batch)
            loss.backward()  # what does that do
            optimizer.step()  # i know its the adam but it is still unclear
            i = i + 1  # counts number of batches or individual images?

        with torch.no_grad():  # use the no grad because we are not mean to change any weights as it is test data
            model.eval()  # what is this fnc
            correct = 0  # we want to check the accuracy
            total = 0  # total of samples sampled
            for batch_idx, (images, labels) in enumerate(test_loader):  # running upon test data
                images = images.to(device)  # to device
                labels = labels.to(device)  # to device
                outputs = model(images)  # get the results (in what format)?
                preds = torch.argmax(outputs, dim=1)  # dont understand this line
                correct += (preds == labels).sum().item()  # count the correct preds but how?
                total += labels.size(0)  # ???
            accuracy = correct / total
            print(f"Validation Accuracy: {accuracy * 100:.2f}%")

        print(f"avarage epoch loss is {total_epoch_loss / i}")
        print(f"Finished Epoch {epoch + 1}/15")

    torch.save(model.state_dict(), "cnn_dogs_cats_model.pth")
    print("model saved successfully")

