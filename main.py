import os
import sys
import numpy as np
import argparse
import glob
import json
import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from utils import read_img
from train import train_model
from evaluate import evaluate_model
from infer import load_model, infer_bootstrap


def main(args):
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

    if args.mode == 'train':
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

    elif args.mode == 'infer':
        models_path = args.models_path
        image_path = args.image_path
        
        models_paths = glob.glob(os.path.join(models_path, '*.pth'))
        if not models_paths:
            print("No models found in the specified directory.")
            return

        models = [load_model(models_path, device) for models_path in models_paths]
        
        mean_score, error = infer_bootstrap(models, image_path, transform, device)
        print(f"The predicted attractiveness score is {mean_score:.3f} ± {error:.3f}")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or infer the model.")
    parser.add_argument('--mode', type=str, required=True, choices=['train', 'infer'],
                        help='Mode to run the script in. Choices are "train" and "infer".')
    parser.add_argument('-m', '--models-path', dest='models_path', type=str, default=None,
                        help='Directory containing the pre-trained models. Required only in "infer" mode.')
    parser.add_argument('-i', '--image-path', dest='image_path', type=str, default=None,
                        help='Path to the image to be inferred. Required only in "infer" mode.')
    args = parser.parse_args()

    main(args)