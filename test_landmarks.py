import os
import numpy as np
from PIL import Image, ImageDraw
from torchvision import transforms
from utils import rescale_landmarks
import torch

def display_landmarks(image, landmarks, image_name):
    draw = ImageDraw.Draw(image)
    for x, y in landmarks.astype(int):
        draw.ellipse([(x-3, y-3), (x+3, y+3)], outline='red', width=2)

    if not os.path.exists('results'):
        os.makedirs('results')
    image.save(os.path.join('results', image_name))

def read_single_landmark(landmark_root, image_name):
    landmark_file = os.path.splitext(image_name)[0] + '.txt'
    landmark_path = os.path.join(landmark_root, landmark_file)
    with open(landmark_path, 'r') as f:
        landmark_data = np.array([list(map(float, line.strip().split())) for line in f])
    return landmark_data

if __name__ == '__main__':
    image_root = 'data/images'
    landmark_root = 'data/landmarks'
    image_name = 'CF7.jpg'

    image_path = os.path.join(image_root, image_name)

    # Define the transformation
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Read the image
    img = Image.open(image_path).convert('RGB')
    img_transformed = transform(img)

    # Read the corresponding landmarks
    landmarks = read_single_landmark(landmark_root, image_name)
    rescale_landmarks(landmarks, img.size)

    # Convert tensor to PIL for displaying
    img_pil = transforms.ToPILImage()(img_transformed)
    
    # Display landmarks on the image
    display_landmarks(img_pil, landmarks, f"{os.path.splitext(image_name)[0]}_landmarks.jpg")
