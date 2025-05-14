import torch.nn as nn


class CNN(nn.Module):  # Creating a CNN class to set the CNN flow and arrtributes, also it inherits the nn.Module
    # that holds important and usefull CNN utilities such as
    def __init__(self):  # The constructor to create a CNN object
        super().__init__()  # send it to the father initialization
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1)  # First conv_layer
        self.conv2 = nn.Conv2d(16, 32, 3, 1, 1)  # Second conv_layer
        self.conv3 = nn.Conv2d(32, 64, 3, 1, 1)
        self.relu = nn.ReLU()  # Activation to highlight important patterns
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)  # pool highest pixel value of 4x4 set to reduce size
        # and improve flow of the calculations.
        self.flatten = nn.Flatten()  # Flattening the tensor for the fc layer
        self.fc1 = nn.Linear(4096, 128)  # First fc_layer
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
        pool3_result = self.pool(relu3_result)
        x = self.flatten(pool3_result)  # Finished the conv, now we flatten the data as preparation for the fc layer
        x = self.fc1(x)  # apply fc layer
        x = self.relu(x)  # apply relu again to highlight important patterns even though we "finished the calculations"
        x = self.fc2(x)  # apply final fc layerר
        return x
