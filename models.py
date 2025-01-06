import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

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
    

def train_model(model, train_loader, optimizer, criterion, device, mask = None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
            

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


