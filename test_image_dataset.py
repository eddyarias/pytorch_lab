import os
import numpy as np
import matplotlib.pyplot as plt
from dataloaders.Image_Dataset import Image_Dataset
from torchvision.transforms import v2
import torch

# Directory of the script file
script_dir = os.path.dirname(__file__)

# Path to the existing test list file
list_path = os.path.join(script_dir, 'test.txt')

# Arguments for Image_Dataset
class Args:
    img_size = 224
    da_library = 'torchvision'
args = Args()

# Transformation
transform = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

def denormalize(tensor):
    """
    Denormalize a tensor image.
    """
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    tensor = tensor.permute(1, 2, 0).numpy()  # Convert to HWC
    tensor = (tensor * std) + mean  # Denormalize
    tensor = np.clip(tensor, 0, 1)  # Clip values to [0, 1]
    return tensor

def show_image_with_treatment(treatment):
    dataset = Image_Dataset(list_path, args=args, transform=transform, treatment=treatment)
    image, label, _ = dataset[0]

    # Denormalize the image for visualization
    image_np = denormalize(image)

    plt.imshow(image_np)
    plt.title(f'{treatment.capitalize()} Treatment')
    plt.show()

# Show images with different treatments
show_image_with_treatment('black')
show_image_with_treatment('blur')