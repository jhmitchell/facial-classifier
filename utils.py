import os
import numpy as np
from PIL import Image, ImageDraw
import torch
from torch.utils.data import TensorDataset
from torchvision import transforms


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


def rescale_landmarks(landmarks, original_size, crop_size=224, resize_size=256):
    # Scale factor between the original 350x350 and the required 224x224
    original_size_x, original_size_y = 350, 350
    scale_factor_x = crop_size / original_size_x
    scale_factor_y = crop_size / original_size_y

    landmarks[:, 0] *= scale_factor_x
    landmarks[:, 1] *= scale_factor_y

    return landmarks


def display_landmarks(image, landmarks_tensor, image_name):
    # Draw directly on the image
    draw = ImageDraw.Draw(image)
    landmarks = landmarks_tensor.cpu().numpy().reshape(-1, 2)

    # Draw the landmarks on the image
    for x, y in landmarks.astype(int):
        draw.ellipse([(x-3, y-3), (x+3, y+3)], outline='red', width=2)

    # Save the modified image with landmarks
    if not os.path.exists('results'):
        os.makedirs('results')
    image.save(os.path.join('results', image_name))
