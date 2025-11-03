# import subprocess

# subprocess.run("pip install -r requirements.txt", shell=True)
# subprocess.run("git clone https://github.com/openvinotoolkit/anomalib.git")
# subprocess.run("cd anomalib && git checkout df50c0b && cd ..", shell=True)

import torchvision
from torchvision import transforms
#from torchvision.transforms import InterpolationMode
from PIL import Image
import os

data_transforms = transforms.Compose([
    transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5),  # Randomly change brightness, contrast, saturation, and hue
    transforms.RandomRotation(degrees=180, fill=175),          # Randomly rotate by -180 to 180 degrees 
    transforms.CenterCrop(size=(650, 650)),       # Center crop the image to 256x256
    transforms.Resize(size=(256, 256)),  # Resize the image to 256x256
    transforms.RandomHorizontalFlip(p=0.5),        # Randomly flip horizontally with 50% probability
    transforms.RandomVerticalFlip(p=0.5),          # Randomly flip vertically with 50% probability
    transforms.ToTensor(),                          # Convert PIL image to PyTorch tensor
    #transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize pixel values (ImageNet statistics)
])

good_image_folder = "dataset/good"
bad_hole_image_folder = "dataset/bad/hole"
bad_bent_image_folder = "dataset/bad/bent"

counter = 1
for img in os.listdir(good_image_folder):
    image = Image.open(os.path.join(good_image_folder, img))
    for _ in range(300):
        augmented_image = data_transforms(image)

        # Option 1: Basic Export (using a counter for filenames)
        filename = f"good_augmented_image_{counter}.png"  # Adjust format as needed (e.g., jpg)
        counter += 1  # Increment counter for unique filenames
        torchvision.utils.save_image(augmented_image, os.path.join("dataset/augmented_good", filename))

counter = 1
for img in os.listdir(bad_hole_image_folder):
    image = Image.open(os.path.join(bad_hole_image_folder, img))
    for _ in range(15):
        augmented_image = data_transforms(image)

        # Option 1: Basic Export (using a counter for filenames)
        filename = f"hole_augmented_image_{counter}.png"  # Adjust format as needed (e.g., jpg)
        counter += 1  # Increment counter for unique filenames
        torchvision.utils.save_image(augmented_image, os.path.join("dataset/augmented_bad/augmented_bad_hole", filename))

counter = 1
for img in os.listdir(bad_bent_image_folder):
    image = Image.open(os.path.join(bad_bent_image_folder, img))
    for _ in range(10):
        augmented_image = data_transforms(image)

        # Option 1: Basic Export (using a counter for filenames)
        filename = f"bent_augmented_image_{counter}.png"  # Adjust format as needed (e.g., jpg)
        counter += 1  # Increment counter for unique filenames
        torchvision.utils.save_image(augmented_image, os.path.join("dataset/augmented_bad/augmented_bad_bent", filename))
    