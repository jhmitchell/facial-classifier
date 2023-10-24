import os
import numpy as np
from PIL import Image
import torch
from torch.utils.data import TensorDataset

# Read images and their attractiveness scores from a text file
def read_img(root, filedir, transform=None):
    with open(filedir, 'r') as f:
        lines = f.readlines()
    
    images, targets = [], []
    for line in lines:
        addr, target = line.strip().split(' ')
        img = Image.open(os.path.join(root, addr)).convert('RGB')
        if transform:
            img = transform(img)
        
        images.append(img)
        targets.append(float(target))
    
    images = torch.stack(images)
    targets = torch.Tensor(targets).unsqueeze(-1)

    return TensorDataset(images, targets)

# Read images and their corresponding landmarks from text files
def read_img_landmarks(image_root, landmark_root, filedir, transform=None):
    with open(filedir, 'r') as f:
        lines = f.readlines()
    
    images, landmarks = [], []
    for line in lines:
        addr, _ = line.strip().split(' ')
        img = Image.open(os.path.join(image_root, addr)).convert('RGB')
        if transform:
            img = transform(img)

        # Read the corresponding landmarks
        landmark_file = os.path.splitext(addr)[0] + '.txt'
        with open(os.path.join(landmark_root, landmark_file), 'r') as f:
            landmark_data = np.array([list(map(float, line.strip().split())) for line in f])
        
        images.append(img)
        landmarks.append(landmark_data.flatten())  # Flatten the 2D landmarks into 1D array

    images = torch.stack(images)
    landmarks = torch.FloatTensor(landmarks)  # Convert to tensor

    return TensorDataset(images, landmarks)  # Return dataset containing images and landmarks
