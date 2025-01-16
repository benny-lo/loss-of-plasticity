import torch
from tqdm import tqdm 


def mask_out(model, mask):
    for idx, param in enumerate(model.parameters()):
        param = param * mask[idx]

def set_grad_zero(model, mask):
    for idx, param in enumerate(model.parameters()):
        param.grad = param.grad * mask[idx]




def train_model(model, train_loader, optimizer, criterion, device='cpu', num_epochs=10, mask = None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
            
    if mask:
        mask_out(model, mask)


    for epoch in range(num_epochs):

        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device).to(torch.float), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if mask:
                set_grad_zero(model,mask)

            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        print(correct/total)

    accuracy = correct / total
    avg_loss = running_loss / len(train_loader)
    return avg_loss, accuracy


def evaluate_model(model, test_loader, criterion, device, mask=None):
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