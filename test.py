import torch
import timm
from torch.utils.data import DataLoader
from torchvision import transforms
import torch.nn as nn
from torchvision.datasets import ImageFolder
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, confusion_matrix
from tqdm import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

# Set the device for computation (GPU if available)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
map_location=torch.device('cpu')
# Define image transformations for validation dataset
# transform = transforms.Compose([
#     transforms.Resize((224, 224)),  # Resize images to 224x224
#     transforms.ToTensor(),          # Convert images to tensor
#     transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # Normalize using ImageNet stats
# ])

# Load validation dataset



# Number of classes in the dataset

# Initialize the model
model = timm.create_model('efficientformerv2_s0.snap_dist_in1k', pretrained=True, num_classes=7)


data_config = timm.data.resolve_model_data_config(model)
transform = timm.data.create_transform(**data_config, is_training=False)
val_dir = r'C:\Users\DELL\Downloads\assignment_ai\assignment\test'
val_dataset = ImageFolder(root=val_dir, transform=transform)
val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)  # Validation data loader
# Load saved checkpoint
checkpoint_path = r'D:\Assignment\checkpoints\best_model.pth'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'),weights_only=True)

# Remove 'module.' prefix from keys if the model was trained with DataParallel
state_dict = {k.replace('module.', ''): v for k, v in checkpoint.items()}

# Load the state_dict into the model
model.load_state_dict(state_dict)
model = model.to(device)  # Move model to the selected device

# Set the model to evaluation mode
model.eval()

# Define the loss function (optional, for computing loss during evaluation)
criterion = nn.CrossEntropyLoss()

# Function to calculate evaluation metrics
def calculate_metrics(labels, predictions):
    """
    Calculate evaluation metrics: accuracy, F1 score, precision, and recall.
    
    Args:
        labels (list): Ground truth labels.
        predictions (list): Predicted labels.
    
    Returns:
        Tuple containing accuracy, F1 score, precision, and recall.
    """
    f1 = f1_score(labels, predictions, average='macro')
    precision = precision_score(labels, predictions, average='macro')
    recall = recall_score(labels, predictions, average='macro')
    accuracy = accuracy_score(labels, predictions)
    return accuracy, f1, precision, recall

# Function to plot and save the confusion matrix
def plot_confusion_matrix(cm, class_names):
    """
    Plot the confusion matrix and save it as an image file.
    
    Args:
        cm (ndarray): Confusion matrix.
        class_names (list): List of class names.
    """
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('Actual')
    plt.title('Confusion Matrix')
    #plt.savefig('/hdd5/purbayan/Ayush/Content_moderation_experiment/output/Confusion_Matrix_effiecient_LS.png')
    plt.close()

# Function to evaluate the model
def evaluate(model, val_loader):
    """
    Evaluate the model on the validation dataset.
    
    Args:
        model (nn.Module): Trained model.
        val_loader (DataLoader): Validation data loader.
    """
    model.eval()  # Set the model to evaluation mode
    total_loss = 0.0
    all_labels = []
    all_predictions = []
    correct = 0
    total = 0

    # Disable gradient computation for validation
    with torch.no_grad():
        loop = tqdm(val_loader, total=len(val_loader), desc="Validating")
        for inputs, labels in loop:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)

            # Calculate loss
            loss = criterion(outputs, labels)
            total_loss += loss.item()

            # Get predictions
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

            all_labels.extend(labels.cpu().numpy())
            all_predictions.extend(predicted.cpu().numpy())

    # Calculate metrics
    accuracy, val_f1, val_precision, val_recall = calculate_metrics(all_labels, all_predictions)
    print(f'Validation Accuracy: {accuracy:.4f}, F1 Score: {val_f1:.4f}, Precision: {val_precision:.4f}, Recall: {val_recall:.4f}')

    # Calculate and print confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print(f'Confusion Matrix:\n{cm}')

    # Plot confusion matrix
    plot_confusion_matrix(cm, val_dataset.classes)

# Run the evaluation
evaluate(model, val_loader)
