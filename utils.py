import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import TensorDataset


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

        # Error checking (CM152.pts is missing from the dataset)
        if not os.path.exists(os.path.join(landmark_root, landmark_file)):
            print(f"Warning: {landmark_file} not found!")
            continue

        with open(os.path.join(landmark_root, landmark_file), 'r') as f:
            landmark_data = np.array(
                [list(map(float, line.strip().split())) for line in f])

        images.append(img)
        # Flatten the 2D landmarks into 1D array
        landmarks.append(landmark_data.flatten())

    images = torch.stack(images)
    landmarks = torch.FloatTensor(landmarks)  # Convert to tensor

    # Return dataset containing images and landmarks
    return TensorDataset(images, landmarks)


def display_landmarks(image, landmarks_tensor, image_name):
    """
    Display landmarks on top of the image using PIL.

    Parameters:
        image_tensor (torch.Tensor): A tensor of shape (C, H, W).
        landmarks_tensor (torch.Tensor): A tensor of shape (num_landmarks * 2,).
        image_name (str): The name to save the image with landmarks as.

    Returns:
        None
    """

    draw = ImageDraw.Draw(image)
    landmarks = landmarks_tensor.cpu().numpy().reshape(-1, 2)

    # Draw circles at the landmark locations
    for x, y in landmarks.astype(int):
        draw.ellipse([(x-3, y-3), (x+3, y+3)], outline='red', width=2)

    if not os.path.exists('results'):
        os.makedirs('results')
    image.save(os.path.join('results', image_name))
