import os
import numpy as np
import argparse
import glob
import json
import torch
import torchvision.transforms as transforms
import streamlit as st
from PIL import Image
from torch.utils.data import DataLoader
from utils import read_img, read_img_landmarks, display_landmarks
from train import train_model, train_landmarks
from evaluate import evaluate_model, evaluate_landmarks
from infer import load_model, load_landmark_model, infer_bootstrap, infer_landmarks


def main(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[
                             0.229, 0.224, 0.225]),
    ])

    root = './data/images'

    metrics = {}

    if config["mode"] == 'train-images':
        # Metrics for each fold
        correlations, maes, rmses = [], [], []

        # Display training configuration
        print(f"Training Configuration:")
        print(f"  Root Directory: {root}")
        print(f"  Batch Size: 32")
        print(f"  Device: {device}")
        print("="*50)

        # Add training configuration to metrics
        metrics['root_directory'] = root
        metrics['batch_size'] = 32
        metrics['device'] = str(device)

        # Metrics for each fold
        fold_metrics = []

        # Loop over each fold for cross-validation
        for fold in range(1, 6):
            print(f"Starting Training for Fold {fold}...")
            traindir = f'./data/train_test_files/cross/cross_validation_{fold}/train_{fold}.txt'
            valdir = f'./data/train_test_files/cross/cross_validation_{fold}/test_{fold}.txt'

            train_dataset = read_img(root, traindir, transform=transform)
            val_dataset = read_img(root, valdir, transform=transform)

            train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            print(f"  Training...")
            model = train_model(train_loader, device, fold)

            print(f"  Evaluating...")
            correlation, mae, rmse = evaluate_model(model, val_loader, device)

            # Display fold summary
            print(f"  Completed Fold {fold}")
            print(f"    Correlation: {correlation:.4f}")
            print(f"    MAE: {mae:.4f}")
            print(f"    RMSE: {rmse:.4f}")
            print("="*50)

            correlations.append(correlation)
            maes.append(mae)
            rmses.append(rmse)

            # Update fold metrics
            fold_metrics.append({
                'fold': fold,
                'corr': correlation,
                'mae': mae,
                'rmse': rmse
            })

        metrics['folds'] = fold_metrics

        # Aggregate metrics across all folds
        avg_corr = np.mean(correlations)
        avg_mae = np.mean(maes)
        avg_rmse = np.mean(rmses)

        print("Training Summary:")
        print(f"  Average Correlation: {avg_corr:.4f}")
        print(f"  Average MAE: {avg_mae:.4f}")
        print(f"  Average RMSE: {avg_rmse:.4f}")

        # Update metrics
        metrics['avg_corr'] = avg_corr
        metrics['avg_mae'] = avg_mae
        metrics['avg_rmse'] = avg_rmse

        # Save metrics to disk
        print("Saving metrics to disk...")
        with open('training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    elif config["mode"] == 'train-landmarks':
        print(f"Training Landmarks Configuration:")
        print(f"  Root Directory: {root}")
        print(f"  Batch Size: 32")
        print(f"  Device: {device}")
        print("=" * 50)

        landmark_root = './data/landmarks'

        # Metrics for each fold
        fold_metrics = []

        for fold in range(1, 6):
            print(f"Starting Training for Landmarks for Fold {fold}...")
            traindir = f'./data/train_test_files/cross/cross_validation_{fold}/train_{fold}.txt'
            valdir = f'./data/train_test_files/cross/cross_validation_{fold}/test_{fold}.txt'

            # Implement this function to return datasets with both images and landmarks
            train_dataset = read_img_landmarks(
                root, landmark_root, traindir, transform=transform)
            val_dataset = read_img_landmarks(
                root, landmark_root, valdir, transform=transform)

            train_loader = DataLoader(
                train_dataset, batch_size=32, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

            print(f"  Training Landmarks...")
            # Implement train_landmarks
            model = train_landmarks(train_loader, device, fold)

            print(f"  Evaluating Landmarks...")
            # Implement evaluate_landmarks similar to evaluate_model but for landmarks
            correlation, mae, rmse = evaluate_landmarks(
                model, val_loader, device)

            # Display fold summary
            print(f"  Completed Fold {fold}")
            print(f"    Correlation: {correlation:.4f}")
            print(f"    MAE: {mae:.4f}")
            print(f"    RMSE: {rmse:.4f}")
            print("=" * 50)

            # Update fold metrics
            fold_metrics.append({
                'fold': int(fold),
                'corr': float(correlation),
                'mae': float(mae),
                'rmse': float(rmse)
            })

        # Save landmark training metrics
        metrics['landmark_folds'] = fold_metrics

        # Save metrics to disk
        print("Saving landmark training metrics to disk...")
        with open('landmark_training_metrics.json', 'w') as f:
            json.dump(metrics, f, indent=4)

    elif config["mode"] == 'infer':
        models_path = config["models_path"]
        image_path = config["image_path"]

        models_paths = glob.glob(os.path.join(
            models_path, 'resnext50_fold_*.pth'))
        if not models_paths:
            print("No models found in the specified directory.")
            return

        models = [load_model(models_path, device)
                  for models_path in models_paths]

        mean_score, error = infer_bootstrap(
            models, image_path, transform, device)
        print(
            f"The predicted attractiveness score is {mean_score:.3f} ± {error:.3f}")

    elif config["mode"] == 'infer-landmarks':
        models_path = config["models_path"]
        image_path = config["image_path"]

        models_paths = glob.glob(os.path.join(
            models_path, 'resnext50_landmarks_fold_*.pth'))
        if not models_paths:
            print("No landmark models found in the specified directory.")
            return

        models = [load_landmark_model(models_path, device)
                  for models_path in models_paths]
        landmarks = infer_landmarks(models, image_path, transform, device)


if __name__ == '__main__':
    st.title("Attractiveness Prediction")
    with st.sidebar:
        uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

    models_path = './models'
    mode = 'infer'

    if uploaded_file is not None and models_path is not None:
        with open("temp_uploaded_image.jpg", "wb") as f:
            f.write(uploaded_file.getbuffer())

        # produce 'args' from mode, models_path, and image_path
        config = {
            'mode': mode,
            'models_path': models_path,
            'image_path': 'temp_uploaded_image.jpg'
        }
        main(config)
