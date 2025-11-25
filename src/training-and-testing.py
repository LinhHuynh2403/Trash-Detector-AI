import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

image_size = (224, 224)
batch_size = 32

train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

train_dir = 'data/Garbage classification/training'
test_dir = 'data/Garbage classification/validation'   # now treated as test set

def load_data():
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    test_dataset = datasets.ImageFolder(root=test_dir, transform=test_transform)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=0)

    return train_loader, test_loader, train_dataset.classes

def build_model(num_classes):
    model = models.resnet18(pretrained=True)
    
    for param in model.parameters():
        param.requires_grad = False

    # unfreeze only last block + fc
    for param in model.layer4.parameters():
        param.requires_grad = True
    for param in model.fc.parameters():
        param.requires_grad = True

    model.fc = nn.Linear(model.fc.in_features, num_classes)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device), device

def define_optimizer(model):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(
        [{'params': model.layer4.parameters(), 'lr': 0.0001},
         {'params': model.fc.parameters(), 'lr': 0.001}]
    )
    return criterion, optimizer

def define_scheduler(optimizer):
    return optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)

# ---------------- TRAIN ONLY ----------------
def train_model(num_epochs, train_loader, model, criterion, optimizer, scheduler, device):
    train_loss_history = []
    train_accuracy_history = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

        epoch_loss = running_loss / len(train_loader)
        epoch_acc = correct / total

        train_loss_history.append(epoch_loss)
        train_accuracy_history.append(epoch_acc)

        print(f"Epoch [{epoch+1}/{num_epochs}] - "
              f"Train Loss: {epoch_loss:.4f}, Train Accuracy: {epoch_acc:.4f}")

        scheduler.step()

    return train_loss_history, train_accuracy_history

# ---------------- FINAL TEST (1-TIME EVAL) ----------------
def test_model(test_loader, model, criterion, device):
    model.eval()
    running_loss = 0.0
    total = 0
    correct = 0

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)

            loss = criterion(outputs, labels)
            running_loss += loss.item()

            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()

    final_loss = running_loss / len(test_loader)
    final_accuracy = correct / total

    print("\n===== FINAL TEST RESULTS =====")
    print(f"Test Loss: {final_loss:.4f}")
    print(f"Test Accuracy: {final_accuracy:.4f}\n")

    return final_loss, final_accuracy

# ---------------- CONFUSION MATRIX ----------------
def plot_confusion_matrix(test_loader, model, device, class_names):
    model.eval()
    all_labels = []
    all_preds = []

    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)

            all_labels.extend(labels.cpu().numpy())
            all_preds.extend(preds.cpu().numpy())

    cm = confusion_matrix(all_labels, all_preds)

    fig, ax = plt.subplots(figsize=(8, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, xticks_rotation=45, colorbar=True)
    plt.title("Confusion Matrix - Trash Classifier")
    plt.tight_layout()
    plt.show()

# ---------------- MAIN ----------------
if __name__ == "__main__":
    train_loader, test_loader, classes = load_data()
    num_classes = len(classes)

    model, device = build_model(num_classes)
    criterion, optimizer = define_optimizer(model)
    scheduler = define_scheduler(optimizer)

    num_epochs = 10
    train_loss, train_accuracy = train_model(
        num_epochs, train_loader, model, criterion, optimizer, scheduler, device
    )

    # 1) Final test metrics
    test_loss, test_acc = test_model(test_loader, model, criterion, device)

    # 2) Training curves
    epochs = range(1, num_epochs + 1)
    plt.figure(figsize=(14, 5))

    # Loss plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_loss, marker='o')
    plt.title("Training Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")

    # Accuracy plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracy, marker='o')
    plt.title("Training Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")

    plt.tight_layout()
    plt.show()

    # 3) Confusion matrix on test set
    plot_confusion_matrix(test_loader, model, device, classes)

    torch.save(model.state_dict(), "trash_sorting_model_final.pth")
