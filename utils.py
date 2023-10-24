import os
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
