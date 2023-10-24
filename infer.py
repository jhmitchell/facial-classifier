import torch
import torchvision.models as models
import numpy as np
import os
from PIL import Image
from torchvision.transforms import transforms
from utils import display_landmarks, rescale_landmarks

num_landmarks = 86  # Number of landmarks in the dataset


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
    # Assuming we are predicting a single score
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()  # Set to evaluation mode
    return model


def load_landmark_model(model_path, device):
    model = models.resnext50_32x4d(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, num_landmarks * 2)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
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
            print(f"  Score: {score:.4f}")
        scores.append(score)

    mean_score = np.mean(scores)
    error = 2*np.std(scores)  # Standard deviation as a measure of "error"

    return mean_score, error


def infer_landmarks(models, image_path, transform, device):
    # Load the original image and convert to RGB
    original_image = Image.open(image_path).convert('RGB')

    # Create a temporary transform to visualize the image that the model sees
    temp_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor()
    ])

    # Apply the temporary transform for display purposes
    display_image = temp_transform(original_image)
    display_image = transforms.ToPILImage()(display_image)

    # Apply the original transform to the image for model inference
    transformed_image = transform(original_image)
    image = transformed_image.unsqueeze(0).to(device)

    all_landmarks = []
    for model in models:
        model.eval()
        with torch.no_grad():
            output = model(image)
            landmarks = output.cpu().squeeze().view(-1, 2).numpy()
            all_landmarks.append(landmarks)

    # Average the landmarks
    avg_landmarks = np.mean(np.array(all_landmarks), axis=0)

    # Rescale the landmarks to the original image size
    avg_landmarks = rescale_landmarks(
        avg_landmarks, original_image.size)

    # Open the transformed image used for display and draw landmarks on it
    file_name, file_ext = os.path.splitext(os.path.basename(image_path))
    image_name = f"{file_name}_landmarks{file_ext}"
    display_landmarks(display_image, torch.tensor(avg_landmarks), image_name)

    return avg_landmarks


# Example usage:
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    model_path = './models/resnext50_model.pth'
    image_path = './path/to/image.jpg'

    model = load_model(model_path, device)
    score = infer(model, image_path, transform, device)
    print(f"The predicted attractiveness score is {score:.4f}")
