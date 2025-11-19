import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import json
import argparse
import csv 
import os 
from utils import build_model, image_size, val_transform

# Define the image size for resizing and batch size
batch_size = 32

# Define transformations for training data (with augmentation)
train_transform = transforms.Compose([
    transforms.Resize(image_size),
    transforms.RandomRotation(40),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths to the training and validation data directories
train_dir = 'data/Garbage classification/training'
val_dir = 'data/Garbage classification/validation'

# Define the function to load datasets (remains the same)
def load_data():
    train_dataset = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0
    )

    return train_loader, val_loader, train_dataset.classes

# ---- Define the Loss Function and Optimizer ----
def define_optimizer(model, lr):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr) 
    return criterion, optimizer

# ---- Training the model (logic remains the same) ----
def train_model(num_epochs, train_loader, val_loader, model, criterion, optimizer, device):
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()
        running_loss = 0.0
        correct_train = 0
        total_train = 0
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = correct_train / total_train
        train_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_train_accuracy)

        model.eval()
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0

        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_accuracy = correct_val / total_val
        val_loss.append(epoch_val_loss)
        val_accuracy.append(epoch_val_accuracy)

        print(f'Epoch [{epoch+1}/{num_epochs}], '
            f'Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, '
            f'Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}')

    return train_loss, val_loss, train_accuracy, val_accuracy

# Function to save metrics as JSON 
def save_metrics_json(train_loss, val_loss, train_accuracy, val_accuracy, num_epochs, filepath): # Path argument added
    metrics = {
        'num_epochs': num_epochs,
        'train_loss': train_loss,
        'val_loss': val_loss,
        'train_accuracy': train_accuracy,
        'val_accuracy': val_accuracy
    }
    with open(filepath, 'w') as f:
        json.dump(metrics, f)
    print(f"Training metrics saved to {filepath}")

# Function to save metrics as CSV 
def save_metrics_csv(train_loss, val_loss, train_accuracy, val_accuracy, filepath): # Path argument added
    
    header = ['Epoch', 'Train_Loss', 'Val_Loss', 'Train_Accuracy', 'Val_Accuracy']
    data = []
    for i in range(len(train_loss)):
        data.append([
            i + 1,
            train_loss[i],
            val_loss[i],
            train_accuracy[i],
            val_accuracy[i]
        ])

    with open(filepath, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(data)
        
    print(f"Training metrics saved to {filepath}")

# Function to save class names 
def save_class_names(classes, filepath): # Path argument added
    with open(filepath, 'w') as f:
        json.dump(classes, f)
    print(f"Class names saved to {filepath}")

# ---- Main function (Modified) ----
if __name__ == "__main__":
    
    # Define the results folder name
    RESULTS_FOLDER = 'results'  
    
    parser = argparse.ArgumentParser(description='Train a garbage classification model.')
    parser.add_argument('--epochs', type=int, default=10, help='Number of training epochs. (Default: 10)')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate for the optimizer. (Default: 0.001)')
    args = parser.parse_args()

    # Load the data
    train_loader, val_loader, classes = load_data()
    num_classes = len(classes)

    # Build the model
    model, device = build_model(num_classes)

    # Define the loss function and optimizer
    criterion, optimizer = define_optimizer(model, args.lr)
    
    # Train the model
    train_loss, val_loss, train_accuracy, val_accuracy = train_model(
        args.epochs, train_loader, val_loader, model, criterion, optimizer, device
    )

    # --- Save all files to the results folder ---
    
    # Define file paths using os.path.join for cross-platform compatibility
    model_path = os.path.join(RESULTS_FOLDER, 'trash_sorting_finetuned_model.pth')
    class_names_path = os.path.join(RESULTS_FOLDER, 'class_names.json')
    metrics_json_path = os.path.join(RESULTS_FOLDER, 'training_metrics.json')
    metrics_csv_path = os.path.join(RESULTS_FOLDER, 'training_metrics.csv')


    # Save the metrics to JSON 
    save_metrics_json(train_loss, val_loss, train_accuracy, val_accuracy, args.epochs, metrics_json_path)
    
    # Save the metrics to CSV 
    save_metrics_csv(train_loss, val_loss, train_accuracy, val_accuracy, metrics_csv_path)

    # Save the trained model
    torch.save(model.state_dict(), model_path)
    print(f"Model saved to {model_path}")
    
    # Save the class names
    save_class_names(classes, class_names_path)