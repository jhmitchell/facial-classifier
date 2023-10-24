import torch
import torchvision.models as models
import numpy as np
from PIL import Image
from torchvision.transforms import transforms

def load_model(model_path, device):
    """
    Load a pretrained ResNeXt-50 model from disk.

    Parameters:
        model_path (str): Path to the saved model.
        device (torch.device): Device on which to load the model.

    Returns:
        model (torch.nn.Module): The loaded model.
    """
    model = models.resnext50_32x4d(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)  # Assuming we are predicting a single score
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model

def infer(model, image_path, transform, device):
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    image = image.to(device)

    with torch.no_grad():
        output = model(image)
        score = output.item()

    return score

def infer_bootstrap(models, image_path, transform, device):
    scores = []
    image = Image.open(image_path).convert('RGB')
    image = transform(image)
    image = image.unsqueeze(0)  # Add a batch dimension
    image = image.to(device)

    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(image)
            score = output.item()
        scores.append(score)

    mean_score = np.mean(scores)
    error = 2*np.std(scores)  # Standard deviation as a measure of "error"
    
    return mean_score, error

# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    model_path = './models/resnext50_model.pth'
    image_path = './path/to/image.jpg'

    model = load_model(model_path, device)
    score = infer(model, image_path, transform, device)
    print(f"The predicted attractiveness score is {score:.4f}")
