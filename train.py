# train.py
import torch
import torch.nn as nn
import torch.optim as optim
import time
from torchvision import models

num_epochs = 20  # Number of training epochs
avg_window_size = 10  # Number of latest batches to average over for time

num_landmarks = 86 

time_list = []


def train_model(train_loader, device, fold):
    # Initialize model architecture and move it to device
    model = models.resnext50_32x4d(weights=None)
    num_ftrs = model.fc.in_features

    # Modify the last layer to have 1 output unit (for regression)
    model.fc = nn.Linear(num_ftrs, 1)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for i, (img, target) in enumerate(train_loader):
            batch_start_time = time.time()
            img, target = img.to(device), target.to(device).float().view(-1, 1)
            optimizer.zero_grad()

            # Ensure output shape is [batch_size, 1]
            output = model(img).view(-1, 1)

            loss = criterion(output, target)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute time taken for this batch and add to list for running average
            batch_time = time.time() - batch_start_time
            time_list.append(batch_time)
            if len(time_list) > avg_window_size:
                del time_list[0]

            avg_time = sum(time_list) / len(time_list)

            # Dynamic console line update
            print(
                f"\rEpoch: {epoch+1}/{num_epochs} | Batch: {i+1}/{len(train_loader)} | Avg time/batch: {avg_time:.4f} s", end="")

        epoch_time = time.time() - epoch_start_time
        print(
            f"\nEpoch {epoch+1} completed in {epoch_time:.2f} s, Loss: {running_loss / len(train_loader)}")

    # Save the trained model for inference
    torch.save(model.state_dict(), f"./models/resnext50_fold_{fold}.pth")

    return model


def train_landmarks(train_loader, device, fold):
    # Initialize model architecture and move it to device
    model = models.resnext50_32x4d(weights=None)
    num_ftrs = model.fc.in_features

    # Modify the last layer to output coordinates for landmarks
    model.fc = nn.Linear(num_ftrs, num_landmarks * 2)
    model = model.to(device)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        epoch_start_time = time.time()

        for i, (img, landmarks) in enumerate(train_loader):
            batch_start_time = time.time()
            img, landmarks = img.to(device), landmarks.to(device).float()
            optimizer.zero_grad()

            # Ensure output shape is [batch_size, num_landmarks * 2]
            output = model(img).view(-1, num_landmarks * 2)

            loss = criterion(output, landmarks)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()

            # Compute time taken for this batch and add to list for running average
            batch_time = time.time() - batch_start_time
            time_list.append(batch_time)
            if len(time_list) > avg_window_size:
                del time_list[0]

            avg_time = sum(time_list) / len(time_list)

            # Dynamic console line update
            print(
                f"\r Epoch: {epoch+1}/{num_epochs} | Batch: {i+1}/{len(train_loader)} | Avg time/batch: {avg_time:.4f} s", end="")

        epoch_time = time.time() - epoch_start_time
        print(
            f"\n Epoch {epoch+1} completed in {epoch_time:.2f} s, Loss: {running_loss / len(train_loader)}")

    # Save the trained landmark model
    torch.save(model.state_dict(),
               f"./models/resnext50_landmarks_fold_{fold}.pth")

    return model
