import torch  # a directory with a lot of data utilities(in this project i used raw torch just for GPU usage)
import torch.nn as nn  # has some learning functions like Loss etc
import torch.optim as optim  # holds different kinds of optimaizers
import torchvision  # incredible library that handle data manipulations and storage such as transforms or ImageFolder
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")  # config the nn calculation to run on GPU(if can)
path_to_data = "C:/Users/Matan/PycharmProjects/CNN_dogs_cats/CORRECT DATA"  # my data path

image_size = 64  # instance created for resize the images in future(in transforms)

train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.CenterCrop(image_size),
    torchvision.transforms.RandomHorizontalFlip(),
    torchvision.transforms.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1), shear=10),
    torchvision.transforms.ColorJitter(brightness=0.2, contrast=0.2),
    torchvision.transforms.ToTensor()  # converts from HxWxC to CxHxW to make math calculations
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

class_names = train_dataset.classes
train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=16, shuffle=True, num_workers=4)  # Final preparing
# before sending to the CNN
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, num_workers=4)  # Final preparing

# before  sending to the CNN


if __name__ == '__main__':
    num_epochs = 15
    model = CNN()  # Creating a CNN object named model
    model.to(device)  # configing the model to run on the gpu (if ther eis one) for faster calculations
    loss_fn = nn.CrossEntropyLoss()  # defining loss_fn to be a CrossEntropy loss
    optimizer = optim.Adam(model.parameters(), lr=0.001)  # ADAM uses AdaGrad and RMSProp, adjust parameter individualy
    epoch_train_losses = []
    epoch_val_accuracies = []
    for epoch in range(num_epochs):  # running in a loop 15 times (15 epoch that runs upon all the data)
        print(f"Starting Epoch {epoch + 1}/" + str(num_epochs))  # indication of where we are
        batch_count = 0  # meant to count the number of batches to calculate average loss
        total_epoch_loss = 0  # total of epoch loss to calculate average loss
        model.train()  # setting train mode

        for batch_idx, (images, labels) in enumerate(  # enumerate because it adds a counting for batches
                train_loader):  # run upon the whole train loader by separated batches
            images = images.to(device)  # what does that mean
            labels = labels.to(device)  # what does that mean
            optimizer.zero_grad()  # It's important to zeroing the gradients
            outputs = model.forward(images)  # applying forward fnc
            loss = loss_fn(outputs, labels)
            total_epoch_loss += loss.item()  # takes the loss of any individual image(or batch) for avg epoch loss
            loss.backward()  # calculate the gradients
            optimizer.step()  # uses the gradients to adjust weights for next batch (through adam algo)
            batch_count += 1  # counts number of batches for avg epoch loss

        avg_epoch_loss = total_epoch_loss / batch_count
        epoch_train_losses.append(avg_epoch_loss)
        print(f"Average training loss for Epoch {epoch + 1}: {avg_epoch_loss:.4f}")

        model.eval()  # evaluation mode (for testing) is mainly puts the dropout set to off.
        correct = 0
        total = 0

        with torch.no_grad():  # use the no grad because we are not mean to change any weights as it is test data
            for batch_idx, (images, labels) in enumerate(test_loader):  # running upon test data
                images = images.to(device)  # to device
                labels = labels.to(device)  # to device
                outputs = model(images)  # get the results in a 2d tensor in size (batch size, num_classes)
                # each row holds 2 columns of the classes, so it shows the score of a dor prediction and so of a cat
                preds = torch.argmax(outputs, dim=1)  # takes the highest prediction from each row(so it chooses
                # the higher prediction between two classes dim means row or column searching)
                correct += (preds == labels).sum().item()  # lables holds the ground true for the lables of the batch,
                # the preds are the preds and it basically do a "OR" gate of predictions and ground true, the .item
                # just simplifies the number of True and puuls it as a singe number instead of a tensor,
                # to then calculate accuracy.
                total += labels.size(0)  # just takes the tensor dimentions.
            accuracy = correct / total
            epoch_val_accuracies.append(accuracy)
            print(f"Validation Accuracy after Epoch {epoch + 1}: {accuracy * 100:.2f}%")
        print(f"Finished Epoch {epoch + 1}/{num_epochs}")
        print("-" * 30)

    print("finished training")

    torch.save(model.state_dict(), "cnn_dogs_cats_model.pth")
    print("model saved successfully")

    plt.figure(figsize=(10, 5))
    plt.plot(range(1, num_epochs + 1), epoch_train_losses, label='Training Loss')
    plt.plot(range(1, num_epochs + 1), epoch_val_accuracies, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Value')
    plt.title('Training Loss and Validation Accuracy over Epochs')
    plt.legend()
    plt.grid(True)
    plt.xticks(range(1, num_epochs + 1))  # Ensure ticks for each epoch
    plt.show()  # Display the plot

    # --- Calculate and Display Confusion Matrix (Requirement 1) ---
    print("\nCalculating final Confusion Matrix on the test set...")
    all_preds = []
    all_labels = []
    model.eval()  # Ensure model is in evaluation mode
    with torch.no_grad():
        for images, labels in test_loader:  # Iterate through test loader again for final eval
            images = images.to(device)
            labels = labels.to(device)
            outputs = model(images)
            preds = torch.argmax(outputs, dim=1)  #
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # Calculate confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # Display confusion matrix
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues)  # Use a color map
    plt.title("Confusion Matrix")
    plt.show()  # Display the plot

    print("Confusion Matrix calculation and display complete.")
