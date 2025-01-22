import torch
from tqdm import tqdm 

def mask_out(model, mask):
    idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            param = param * mask[idx]
            idx += 1

def set_grad_zero(model, mask):
    idx = 0
    for name, param in model.named_parameters():
        if 'weight' in name:
            param.grad = param.grad * mask[idx]
            idx += 1

def train_model(cfg, model, train_loader, device, num_epochs, mask = None):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    if cfg.general.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.general.lr)
    
    else:
        raise NotImplementedError

    
    if cfg.general.criterion == 'cross entropy':
        criterion = torch.nn.CrossEntropyLoss()

    else:
        raise NotImplementedError

            
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


def evaluate_model(cfg, model, test_loader, device, mask=None):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0

    if cfg.general.criterion == 'cross entropy':
        criterion = torch.nn.CrossEntropyLoss()

    else:
        raise NotImplementedError

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

def train_model_gradient(cfg, model,train_loader, device='cpu', num_epochs=10, mask = None):
    model.train()


    if cfg.general.optimizer == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.general.lr)
    
    else:
        raise NotImplementedError

    
    if cfg.general.criterion == 'cross entropy':
        criterion = torch.nn.CrossEntropyLoss()

    else:
        raise NotImplementedError
            
    if mask:
        mask_out(model, mask)

    grad = []

    for name, param in model.named_parameters():
        if 'weight' in name:
            grad.append(torch.zeros_like(param))

    for epoch in range(num_epochs):
        for images, labels in tqdm(train_loader, desc="Training"):
            images, labels = images.to(device).to(torch.float), labels.to(device)

            optimizer.zero_grad()

            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            
            if mask:
                set_grad_zero(model,mask)

            idx = 0
            for name, param in model.named_parameters():
                if 'weight' in name:
                    grad[idx] += param.grad.abs()
                    idx += 1
            
            optimizer.step()
    
    grad = [x / (num_epochs * len(train_loader)) for x in grad]

    return grad