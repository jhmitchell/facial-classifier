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
