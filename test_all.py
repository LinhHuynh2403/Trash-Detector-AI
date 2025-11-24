import torch
import json
import argparse
import os
from torchvision import datasets
from torch.utils.data import DataLoader

# Import components from utils.py and training.py for consistency
from src.utils import load_checkpoint_model, val_transform, image_size # Import components from utils

batch_size = 32
val_dir = 'data/Garbage classification/validation'
CLASS_NAMES_PATH = 'results/class_names.json' 

def load_validation_data(val_transform):
    """Loads the validation dataset and DataLoader."""
    print(f"Loading validation data from: {val_dir}")
    # We pass val_transform to ensure we use the correct transformation
    val_dataset = datasets.ImageFolder(root=val_dir, transform=val_transform)
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False, 
        num_workers=0
    )
    return val_loader, val_dataset.classes

def load_class_names(path):
    """Loads the list of class names from a JSON file."""
    try:
        with open(path, 'r') as f:
            class_names = json.load(f)
        return class_names
    except FileNotFoundError:
        print(f"Error: Class names file not found at {path}. Run training.py first.")
        return None

def evaluate_model_detailed(model, device, val_loader, class_names):
    """
    Evaluates the model and reports detailed per-class results (Total, Correct, Accuracy).
    """
    print("\n--- Starting Detailed Validation Evaluation ---")
    
    model.eval()
    
    # Initialize trackers for each class
    class_totals = {name: 0 for name in class_names}
    class_correct = {name: 0 for name in class_names}
    
    total_val = 0
    correct_val = 0

    with torch.no_grad():
        for inputs, labels in val_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            
            # Forward pass
            outputs = model(inputs)
            
            # Get predictions
            _, predicted = torch.max(outputs, 1)
            
            # Update overall totals
            total_val += labels.size(0)
            correct_val += (predicted == labels).sum().item()

            # Update per-class totals and correct counts
            for i in range(len(labels)):
                true_label_index = labels[i].item()
                predicted_label_index = predicted[i].item()
                class_name = class_names[true_label_index]
                
                class_totals[class_name] += 1
                
                if true_label_index == predicted_label_index:
                    class_correct[class_name] += 1

    # Print the detailed summary
    print("\n==============================================")
    print("**Detailed Validation Summary**")
    print("==============================================")
    
    for class_name in sorted(class_names):
        correct = class_correct[class_name]
        total = class_totals[class_name]
        
        # Avoid division by zero
        accuracy = (correct / total) if total > 0 else 0.0
        
        # Example format: Paper (14/24), Accuracy: 58.33%
        print(f"  {class_name.ljust(15)}: Correct: {correct}/{total}, Accuracy: {accuracy * 100:.2f}%")

    # Final Summary
    overall_accuracy = correct_val / total_val
    print("\n--- Overall Results ---")
    print(f"Total Validation Samples: {total_val}")
    print(f"Overall Accuracy: **{overall_accuracy:.4f}** ({overall_accuracy * 100:.2f}%)")
    print("==============================================")
    
    return overall_accuracy

# You'll need to include the CSV reading logic here or use a helper function 
# to find the model path based on run_id, as done in the previous response's script.

# --- Main Execution (The rest of the logic remains similar to your existing script) ---
if __name__ == "__main__":
    
    # Placeholder for CSV reading logic (assuming it's here for brevity)
    SUMMARY_CSV_PATH = 'results/training_metrics.csv'
    
    parser = argparse.ArgumentParser(description='Evaluate a trained model on the entire validation dataset.')
    parser.add_argument('--run_id', type=int, required=True, help='The Run ID from training_metrics.csv.')
    args = parser.parse_args()
    
    # Logic to retrieve num_epochs and lr from CSV using args.run_id (Omitted for brevity)
    # ... (You must include the CSV lookup logic here) ...
    
    # Assuming the lookup found these values for run_id 2:
    num_epochs = 40  
    lr = 0.000010
    
    # Dynamically construct the model path
    suffix = f"e{num_epochs}_lr{lr:.6f}".replace('.', '')
    model_path = os.path.join('results', f'model_{suffix}.pth')
    
    # Load class names
    class_names = load_class_names(CLASS_NAMES_PATH)
    if not class_names:
        exit()
        
    num_classes = len(class_names)
    
    # Load the trained model
    print(f"Attempting to load model from {model_path} for evaluation...")
    try:
        model, device = load_checkpoint_model(model_path, num_classes)
        print("Model loaded successfully.")
        
        # Load the data
        val_loader, _ = load_validation_data(val_transform)

        # Evaluate the model
        evaluate_model_detailed(model, device, val_loader, class_names)

    except FileNotFoundError:
        print(f"Error: Model file not found at {model_path}. Check the Run ID and if the training completed successfully.")
    except Exception as e:
        print(f"An unexpected error occurred: {e}")