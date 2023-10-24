import numpy as np
import torch

def evaluate_model(model, val_loader, device):
    model.eval()
    labels, preds = [], []
    with torch.no_grad():
        for img, target in val_loader:
            img, target = img.to(device), target.to(device)
            output = model(img).squeeze(1)
            labels.append(target.cpu().item())
            preds.append(output.cpu().item())

    labels = np.array(labels)
    preds = np.array(preds)
    correlation = np.corrcoef(labels, preds)[0, 1]
    mae = np.mean(np.abs(labels - preds))
    rmse = np.sqrt(np.mean(np.square(labels - preds)))

    return correlation, mae, rmse

def evaluate_landmarks(model, val_loader, device):
    model.eval()
    actual_landmarks, predicted_landmarks = [], []
    
    with torch.no_grad():
        for img, target in val_loader:
            img, target = img.to(device), target.to(device)
            output = model(img)
            
            # Flatten the landmark coordinates for comparison
            actual_landmarks.extend(target.view(target.size(0), -1).cpu().numpy())
            predicted_landmarks.extend(output.view(output.size(0), -1).cpu().numpy())

    actual_landmarks = np.array(actual_landmarks)
    predicted_landmarks = np.array(predicted_landmarks)
    
    # Calculate metrics
    correlation_matrix = np.corrcoef(actual_landmarks.flatten(), predicted_landmarks.flatten())
    correlation = correlation_matrix[0, 1]
    
    mae = np.mean(np.abs(actual_landmarks - predicted_landmarks))
    rmse = np.sqrt(np.mean(np.square(actual_landmarks - predicted_landmarks)))

    return correlation, mae, rmse
