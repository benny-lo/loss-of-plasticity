import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

import matplotlib.pyplot as plt
import numpy as np
import time

class SimpleMLP(nn.Module):
    def __init__(self, input_size=28*28, hidden_size=256, output_size=10):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the images
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    



class SimpleVGG(nn.Module):
    def __init__(self, num_classes=10):
        super(SimpleVGG, self).__init__()
        
        self.features = nn.Sequential(
            # First Block
            nn.Conv2d(3, 64, kernel_size=3, padding=1),  # Conv layer (3 input channels, 64 output channels)
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # Max Pooling (2x2)
            
            # Second Block
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Third Block
            nn.Conv2d(128, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            
            # Fourth Block
            nn.Conv2d(256, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )
        
        self.classifier = nn.Sequential(
            nn.Dropout(),
            nn.Linear(512 * 2 * 2, 4096),
            nn.ReLU(inplace=True),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, num_classes)
        )
    
    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)  # Flatten the output from Conv layers
        x = self.classifier(x)
        return x




def train_model(model, train_loader, optimizer, criterion, device, num_epochs=1,mask = None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
            
    for epoch in range(num_epochs):

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device).to(torch.float), labels.to(device)

            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()


            if mask is not None:
                
                with torch.no_grad():
                    for idx, param in enumerate(model.parameters()):
                        param.grad[mask[idx] == 0] = 0

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            time.sleep(2)

        print(correct/total)

    accuracy = correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in tqdm(test_loader, desc="Evaluating"):
            images, labels = images.to(device).to(torch.float), labels.to(device)

            outputs = model(images)

            loss = criterion(outputs, labels)

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    accuracy = correct / total
    avg_loss = running_loss / len(test_loader)
    return avg_loss, accuracy


def smooth_signal(signal, window_size):

    if window_size % 2 == 0:
        raise ValueError("Window size must be odd to ensure symmetry.")
    
    half_window = window_size // 2
    padded_signal = np.pad(signal, (half_window, half_window), mode='edge')
    smoothed_signal = np.convolve(padded_signal, np.ones(window_size) / window_size, mode='valid')
    return smoothed_signal


def plot_results(task_performance,smoothing=True,window=11):
    

    # Plot training and testing accuracy across tasks
    plt.figure(figsize=(12, 5))

    train_acc = task_performance["train_acc"]
    test_acc = task_performance["test_acc"]

    if smoothing:
        train_acc = smooth_signal(train_acc,window)
        test_acc = smooth_signal(test_acc,window)


    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(test_acc, label="Test Accuracy")
    plt.xlabel("Task")
    plt.ylabel("Accuracy")
    plt.title("Model Performance Across Tasks")
    plt.legend()
    plt.show()




