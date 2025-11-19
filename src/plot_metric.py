# this script plots training and validation loss and accuracy from a JSON file

import matplotlib.pyplot as plt
import json
import argparse
import os

def plot_metrics(train_loss, val_loss, train_accuracy, val_accuracy, num_epochs, save_path='training_metrics_plot.png'):
    """
    Plots the training and validation loss and accuracy.
    """
    # Create the figure and subplots
    plt.figure(figsize=(10, 4))
    
    # Plot 1: Loss
    plt.subplot(1, 2, 1)
    # Ensure epochs range starts from 1 for plotting clarity
    epochs_range = range(1, num_epochs + 1)
    
    plt.plot(epochs_range, train_loss, label='Training Loss')
    plt.plot(epochs_range, val_loss, label='Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.title('Loss vs. Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    # Plot 2: Accuracy
    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, train_accuracy, label='Training Accuracy')
    plt.plot(epochs_range, val_accuracy, label='Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.title('Accuracy vs. Epochs')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.6)
    
    plt.tight_layout()
    plt.savefig(save_path)
    print(f"Plot saved to {save_path}")
    plt.show()


def load_metrics(filepath):
    """
    Loads metrics from a JSON file.
    """
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Error: Metrics file not found at {filepath}. Please run training.py first.")
    
    with open(filepath, 'r') as f:
        metrics = json.load(f)
        
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Plot training and validation metrics.')
    parser.add_argument(
        '--file', 
        type=str, 
        default='training_metrics.json', 
        help='Path to the JSON file containing the training metrics. (Default: training_metrics.json)'
    )
    args = parser.parse_args()
    
    try:
        metrics = load_metrics(args.file)
        
        # Unpack metrics
        num_epochs = metrics['num_epochs']
        train_loss = metrics['train_loss']
        val_loss = metrics['val_loss']
        train_accuracy = metrics['train_accuracy']
        val_accuracy = metrics['val_accuracy']
        
        # Generate the plot
        plot_metrics(train_loss, val_loss, train_accuracy, val_accuracy, num_epochs)
        
    except FileNotFoundError as e:
        print(e)
    except KeyError:
        print(f"Error: The file {args.file} seems corrupted or is missing required keys.")