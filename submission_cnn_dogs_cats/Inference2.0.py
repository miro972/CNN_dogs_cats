import PIL.Image
from model import CNN
import sys
import torchvision
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

model = CNN()
model.to(device)
trained_weights = torch.load('cnn_dogs_cats_model.pth')
model.load_state_dict(trained_weights)
model.eval()

raw_path_to_image = sys.argv[1]
image_size = 64
transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.CenterCrop(image_size),
    torchvision.transforms.ToTensor()
])

image = PIL.Image.open(raw_path_to_image).convert('RGB')
image.show()
image = transform(image)
image = image.unsqueeze(0)
image = image.to(device)

result = model.forward(image)

predicted_idx = torch.argmax(result, dim=1).item()

idx_to_predict_class = {0: 'cat', 1: 'dog'}
predicted_animal = idx_to_predict_class[predicted_idx]

print("predicted animal:", predicted_animal)
