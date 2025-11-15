import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms, models
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

# Define the image size for resizing and batch size
image_size = (224, 224)  # Resize images to 224x224 for the model
batch_size = 32

# Define transformations for training data (with increased augmentation)
train_transform = transforms.Compose([
    transforms.Resize(image_size),  
    transforms.RandomRotation(40),  # Random rotation
    transforms.RandomHorizontalFlip(),  # Horizontal flip
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),  # Color jitter
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),  # Random translation
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),  # Random crop
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
])

# Define transformations for validation data (no augmentation, just resizing and normalization)
val_transform = transforms.Compose([
    transforms.Resize(image_size),  
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Paths to the training and validation data directories
train_dir = 'data/Garbage classification/training'
val_dir = 'data/Garbage classification/validation'

# Define the function to load datasets
def load_data():
    # Load the training dataset using ImageFolder (with transforms)
    train_dataset = datasets.ImageFolder(
        root=train_dir,
        transform=train_transform  # Apply transformations for training
    )

    # Load the validation dataset using ImageFolder (with transforms)
    val_dataset = datasets.ImageFolder(
        root=val_dir,
        transform=val_transform  # Apply transformations for validation
    )

    # Create DataLoader for training and validation sets (with num_workers=0 to avoid multiprocessing error)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,        # Number of images per batch
        shuffle=True,                 # Shuffle the data to avoid bias
        num_workers=0                 # Disable multiprocessing (set to 0 for Windows compatibility)
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,        # Number of images per batch
        shuffle=False,                # No need to shuffle validation data
        num_workers=0                 # Disable multiprocessing (set to 0 for Windows compatibility)
    )

    return train_loader, val_loader, train_dataset.classes

# ---- Define the model (ResNet18) with unfreezing of layers ----
def build_model(num_classes):
    # Load a pretrained ResNet18 model
    model = models.resnet18(pretrained=True)

    # Unfreeze the last two blocks (layer4 and fc)
    for param in model.parameters():
        param.requires_grad = False  # Freeze all layers initially

    # Unfreeze the last two blocks (layer4)
    for param in model.layer4.parameters():
        param.requires_grad = True

    # Unfreeze the fully connected layer
    for param in model.fc.parameters():
        param.requires_grad = True

    # Modify the final fully connected layer to match the number of classes in your dataset
    model.fc = nn.Linear(model.fc.in_features, num_classes)

    # Move the model to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    
    return model, device

# ---- Define the Loss Function and Optimizer ----
def define_optimizer(model):
    criterion = nn.CrossEntropyLoss()  # Use CrossEntropyLoss for multi-class classification
    optimizer = optim.Adam([
        {'params': model.layer4.parameters(), 'lr': 0.0001},  # Lower learning rate for frozen layers
        {'params': model.fc.parameters(), 'lr': 0.001}  # Higher learning rate for final fully connected layer
    ], lr=0.0001)  # Base learning rate for frozen layers
    return criterion, optimizer

# ---- Add a Learning Rate Scheduler ----
def define_scheduler(optimizer):
    # Use StepLR to decrease the learning rate every 5 epochs
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
    return scheduler

# ---- Training the model ----
def train_model(num_epochs, train_loader, val_loader, model, criterion, optimizer, scheduler, device):
    train_loss = []
    val_loss = []
    train_accuracy = []
    val_accuracy = []

    for epoch in range(num_epochs):
        model.train()  # Set the model to training mode
        running_loss = 0.0
        correct_train = 0
        total_train = 0

        # Training loop
        for inputs, labels in train_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()  # Zero the parameter gradients
            
            # Forward pass
            outputs = model(inputs)
            
            # Compute loss
            loss = criterion(outputs, labels)
            
            # Backward pass and optimization
            loss.backward()
            optimizer.step()
            
            # Calculate the training loss and accuracy
            running_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += labels.size(0)
            correct_train += (predicted == labels).sum().item()

        epoch_train_loss = running_loss / len(train_loader)
        epoch_train_accuracy = correct_train / total_train
        train_loss.append(epoch_train_loss)
        train_accuracy.append(epoch_train_accuracy)

        # Validation loop
        model.eval()  # Set the model to evaluation mode
        correct_val = 0
        total_val = 0
        running_val_loss = 0.0

        with torch.no_grad():  # No gradients needed for validation
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(device), labels.to(device)
                
                # Forward pass
                outputs = model(inputs)
                
                # Compute validation loss
                loss = criterion(outputs, labels)
                running_val_loss += loss.item()
                
                # Calculate accuracy
                _, predicted = torch.max(outputs, 1)
                total_val += labels.size(0)
                correct_val += (predicted == labels).sum().item()

        epoch_val_loss = running_val_loss / len(val_loader)
        epoch_val_accuracy = correct_val / total_val
        val_loss.append(epoch_val_loss)
        val_accuracy.append(epoch_val_accuracy)

        # Print statistics for each epoch
        print(f'Epoch [{epoch+1}/{num_epochs}], '
              f'Train Loss: {epoch_train_loss:.4f}, Train Accuracy: {epoch_train_accuracy:.4f}, '
              f'Val Loss: {epoch_val_loss:.4f}, Val Accuracy: {epoch_val_accuracy:.4f}')

        # Update the learning rate using the scheduler
        scheduler.step()

    # Return the metrics for later visualization
    return train_loss, val_loss, train_accuracy, val_accuracy

# ---- Plot the training and validation loss/accuracy ----
def plot_metrics(train_loss, val_loss, train_accuracy, val_accuracy, num_epochs):
    # Plot training and validation loss
    plt.plot(range(num_epochs), train_loss, label='Training Loss')
    plt.plot(range(num_epochs), val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training and validation accuracy
    plt.plot(range(num_epochs), train_accuracy, label='Training Accuracy')
    plt.plot(range(num_epochs), val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# ---- Main function ----
if __name__ == "__main__":
    # Load the data
    train_loader, val_loader, classes = load_data()
    num_classes = len(classes)  # Number of categories (e.g., glass, plastic, etc.)

    # Build the model
    model, device = build_model(num_classes)

    # Define the loss function, optimizer, and learning rate scheduler
    criterion, optimizer = define_optimizer(model)
    scheduler = define_scheduler(optimizer)

    # Train the model
    num_epochs = 10  # Train for more epochs to improve performance
    train_loss, val_loss, train_accuracy, val_accuracy = train_model(
        num_epochs, train_loader, val_loader, model, criterion, optimizer, scheduler, device
    )

    # Plot the results
    plot_metrics(train_loss, val_loss, train_accuracy, val_accuracy, num_epochs)

    # Save the fine-tuned model
    torch.save(model.state_dict(), 'fine_tuned_trash_sorting_model.pth')

