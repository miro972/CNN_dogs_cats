import torch
from torchvision import transforms
from PIL import Image
import sys
from model import CNN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = CNN().to(device)
model.load_state_dict(torch.load('cnn_dogs_cats_model.pth', map_location=device))
model.eval()

image_size = 64
test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.CenterCrop(image_size),
    transforms.ToTensor()
])

idx_to_class = {0: 'cat', 1: 'dog'}


def predict(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = test_transform(image).unsqueeze(0).to(device)
    with torch.no_grad():
        output = model(input_tensor)
        prediction = torch.argmax(output, dim=1).item()
    return idx_to_class[prediction]


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Usage: python inference.py <image_path>")
    else:
        image_path = sys.argv[1]
        result = predict(image_path)
        print(f"Predicted class: {result}")

